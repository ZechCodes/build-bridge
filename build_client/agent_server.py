"""Local WebSocket server for the Build Agent Protocol.

Accepts agent connections on localhost, handles the BAP handshake and message
lifecycle, persists data to SQLite, and emits E2EE-encrypted notifications
to browser clients via a broadcast callback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import websockets
from websockets.asyncio.server import serve as ws_serve, ServerConnection

from build_client.agent_protocol import (
    ACTIVITY_DELTA,
    ACTIVITY_END,
    ACTIVITY_PING,
    AGENT_ERROR,
    AGENT_GOODBYE,
    AGENT_HELLO,
    AGENT_CONFIGURED,
    AGENT_TO_CLIENT,
    BIDIRECTIONAL,
    CHAT_RESPONSE,
    ERROR_CAPABILITY_MISMATCH,
    ERROR_INTERNAL_ERROR,
    ERROR_INVALID_MESSAGE,
    ERROR_PROTOCOL_VIOLATION,
    ERROR_RECONNECT_FAILED,
    ERROR_UNKNOWN_TYPE,
    PROTOCOL_VERSION,
    TOOL_RESULT,
    TOOL_USE,
    check_capability,
    generate_id,
    make_envelope,
    now_iso,
    validate_agent_hello,
    validate_envelope,
)
from build_client.agent_store import AgentStore

log = logging.getLogger(__name__)

# Default port for the local agent WS server.
DEFAULT_AGENT_PORT = 9783

# Ping interval for agent health checking (§2.3).
AGENT_PING_INTERVAL_S = 30
AGENT_PING_TIMEOUT_S = 10

# Type for the browser notification callback.
BroadcastCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


@dataclass
class AgentConnection:
    """State for a single connected agent."""

    ws: ServerConnection
    agent_id: str
    channel_id: str
    capabilities: set[str]
    harness: str
    model: str
    hello_received: bool = False


class AgentServer:
    """Local WebSocket server implementing the Build Agent Protocol.

    Responsibilities:
    - Accept agent connections on ws://localhost:{port}/agent
    - Handle agent.hello handshake and send agent.configured
    - Validate and dispatch all protocol messages
    - Persist chat messages, activity, and tool use to AgentStore
    - Broadcast agent events to browser sessions via E2EE
    """

    def __init__(
        self,
        store: AgentStore,
        broadcast: BroadcastCallback | None = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_AGENT_PORT,
    ) -> None:
        self.store = store
        self._broadcast = broadcast
        self._host = host
        self._port = port
        self._agents: dict[str, AgentConnection] = {}  # agent_id -> connection
        self._channel_to_agent: dict[str, str] = {}  # channel_id -> agent_id
        self._server: Any = None
        self._serve_task: asyncio.Task | None = None

    @property
    def port(self) -> int:
        return self._port

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Start the agent WebSocket server."""
        self._server = await ws_serve(
            self._handle_agent,
            self._host,
            self._port,
            ping_interval=AGENT_PING_INTERVAL_S,
            ping_timeout=AGENT_PING_TIMEOUT_S,
            max_size=10 * 1024 * 1024,  # 10 MB — localhost only, large history payloads.
        )
        log.info("Agent server listening on ws://%s:%s/agent", self._host, self._port)

    async def stop(self) -> None:
        """Gracefully shut down: send agent.shutdown to all agents, close server."""
        # Send shutdown to all connected agents.
        for agent in list(self._agents.values()):
            try:
                shutdown = make_envelope("agent.shutdown", {"reason": "client_shutdown"})
                await agent.ws.send(json.dumps(shutdown))
            except Exception:
                pass

        # Give agents a moment to send agent.goodbye.
        await asyncio.sleep(0.5)

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            log.info("Agent server stopped")

    # -----------------------------------------------------------------
    # Send a chat message from user/browser to an agent
    # -----------------------------------------------------------------

    async def send_chat_message(
        self,
        channel_id: str,
        content: str | list[dict[str, Any]],
        msg_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Forward a user chat message to the agent on a channel.

        Returns True if the message was sent, False if no agent is connected.
        """
        agent_id = self._channel_to_agent.get(channel_id)
        if not agent_id:
            return False

        agent = self._agents.get(agent_id)
        if not agent:
            return False

        msg_id = msg_id or generate_id()

        # Store the user message.
        content_str = content if isinstance(content, str) else json.dumps(content)
        self.store.store_chat_message(msg_id, channel_id, "user", content_str)

        # Send chat.message to agent.
        payload: dict[str, Any] = {"role": "user", "content": content}
        if attachments:
            payload["attachments"] = attachments

        envelope = make_envelope(
            "chat.message",
            payload,
            msg_id=msg_id,
        )
        try:
            await agent.ws.send(json.dumps(envelope))
            log.info("Sent chat.message to agent %s on channel %s", agent_id[:8], channel_id[:8])
            return True
        except Exception as exc:
            log.error("Failed to send chat.message to agent: %s", exc)
            return False

    async def send_cancel(self, channel_id: str) -> bool:
        """Send chat.cancel to the agent on a channel."""
        agent_id = self._channel_to_agent.get(channel_id)
        if not agent_id:
            return False
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        envelope = make_envelope("chat.cancel", {})
        try:
            await agent.ws.send(json.dumps(envelope))
            return True
        except Exception:
            return False

    def get_channel_for_agent(self, agent_id: str) -> str | None:
        """Get the channel_id for a connected agent."""
        agent = self._agents.get(agent_id)
        return agent.channel_id if agent else None

    def is_channel_active(self, channel_id: str) -> bool:
        """Check if an agent is connected and active on a channel."""
        return channel_id in self._channel_to_agent

    # -----------------------------------------------------------------
    # Connection handler
    # -----------------------------------------------------------------

    async def _handle_agent(self, ws: ServerConnection) -> None:
        """Handle a single agent WebSocket connection."""
        agent: AgentConnection | None = None

        try:
            # --- Handshake: wait for agent.hello ---
            agent = await self._handle_hello(ws)
            if not agent:
                return

            log.info(
                "Agent connected: %s (%s, %s) on channel %s",
                agent.agent_id[:8], agent.harness, agent.model, agent.channel_id[:8],
            )

            # --- Notify browser of agent connection ---
            await self._notify_browser(agent.channel_id, {
                "action": "agent_event",
                "channel_id": agent.channel_id,
                "event_type": "agent.connected",
                "event": {
                    "agent_id": agent.agent_id,
                    "harness": agent.harness,
                    "model": agent.model,
                    "capabilities": list(agent.capabilities),
                },
            })

            # --- Message loop ---
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(
                        ws, ERROR_INVALID_MESSAGE, "Invalid JSON", fatal=False,
                    )
                    continue

                await self._dispatch(agent, data)

        except websockets.ConnectionClosed:
            pass
        except Exception as exc:
            log.error("Agent handler error: %s", exc, exc_info=True)
        finally:
            if agent:
                self._cleanup_agent(agent)
                await self._notify_browser(agent.channel_id, {
                    "action": "agent_event",
                    "channel_id": agent.channel_id,
                    "event_type": "agent.disconnected",
                    "event": {"agent_id": agent.agent_id},
                })

    async def _handle_hello(self, ws: ServerConnection) -> AgentConnection | None:
        """Wait for agent.hello, validate, create/restore channel, send agent.configured."""
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            log.warning("Agent failed to send agent.hello within timeout")
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(ws, ERROR_INVALID_MESSAGE, "Invalid JSON", fatal=True)
            return None

        # Validate envelope.
        valid, err = validate_envelope(data)
        if not valid:
            await self._send_error(ws, ERROR_INVALID_MESSAGE, err, fatal=True)
            return None

        if data["type"] != AGENT_HELLO:
            await self._send_error(
                ws, ERROR_PROTOCOL_VIOLATION,
                f"First message must be agent.hello, got {data['type']}",
                fatal=True,
            )
            return None

        # Validate payload.
        payload = data["payload"]
        valid, err = validate_agent_hello(payload)
        if not valid:
            await self._send_error(ws, ERROR_INVALID_MESSAGE, err, fatal=True)
            return None

        agent_id = payload["agent_id"]
        harness = payload["harness"]
        model = payload["model"]
        capabilities = set(payload["capabilities"])
        reconnect = payload["reconnect"]

        # Create or restore channel.
        channel = None
        if reconnect:
            channel = self.store.get_channel_by_agent_id(agent_id)
            if not channel:
                await self._send_error(
                    ws, ERROR_RECONNECT_FAILED,
                    f"No channel found for agent_id {agent_id}",
                    fatal=True, ref=data["id"],
                )
                return None
            # Reactivate.
            self.store.update_channel_status(channel.id, "active")
            log.info("Agent %s reconnected to channel %s", agent_id[:8], channel.id[:8])
        else:
            # Check if the spawner pre-created a channel for this agent_id.
            channel = self.store.get_channel_by_agent_id(agent_id)
            if channel:
                self.store.update_channel_status(channel.id, "active")
                log.info("Agent %s connected to pre-created channel %s", agent_id[:8], channel.id[:8])
            else:
                channel_id = f"ch_{uuid.uuid4().hex[:8]}"
                channel = self.store.create_channel(
                    channel_id=channel_id,
                    agent_id=agent_id,
                    harness=harness,
                    model=model,
                )
                log.info("Created channel %s for agent %s", channel_id, agent_id[:8])

        # Build history for agent.configured.
        chat_history = [
            {"role": m.role, "content": m.content}
            for m in self.store.get_chat_history(channel.id)
        ]
        activity_history = [
            json.loads(e.data) for e in self.store.get_activity_history(channel.id)
        ]

        # Register the connection.
        agent = AgentConnection(
            ws=ws,
            agent_id=agent_id,
            channel_id=channel.id,
            capabilities=capabilities,
            harness=harness,
            model=model,
            hello_received=True,
        )
        self._agents[agent_id] = agent
        self._channel_to_agent[channel.id] = agent_id

        # Send agent.configured.
        configured = make_envelope(
            AGENT_CONFIGURED,
            {
                "channel_id": channel.id,
                "system_prompt": channel.system_prompt,
                "chat_instructions": (
                    "You have access to 'send' and 'read_unread' MCP tools for communicating "
                    "with the user. Use 'read_unread' to check for user messages and 'send' to "
                    "reply. Do not output user-facing text directly — always use the send tool."
                ),
                "history": {
                    "chat": chat_history,
                    "activity": activity_history,
                },
            },
            ref=data["id"],
        )
        await ws.send(json.dumps(configured))
        return agent

    # -----------------------------------------------------------------
    # Message dispatch
    # -----------------------------------------------------------------

    async def _dispatch(self, agent: AgentConnection, data: dict[str, Any]) -> None:
        """Validate and dispatch an incoming message from an agent."""
        # Validate envelope.
        valid, err = validate_envelope(data)
        if not valid:
            await self._send_error(
                agent.ws, ERROR_INVALID_MESSAGE, err, fatal=False, ref=data.get("id"),
            )
            return

        msg_type = data["type"]
        payload = data["payload"]
        msg_id = data["id"]
        ref = data.get("ref")

        # Direction check: only accept agent→client or bidirectional.
        if msg_type not in AGENT_TO_CLIENT and msg_type not in BIDIRECTIONAL:
            await self._send_error(
                agent.ws, ERROR_PROTOCOL_VIOLATION,
                f"Unexpected direction for message type: {msg_type}",
                fatal=False, ref=msg_id,
            )
            return

        # Capability check.
        allowed, err = check_capability(msg_type, agent.capabilities)
        if not allowed:
            await self._send_error(
                agent.ws, ERROR_CAPABILITY_MISMATCH, err, fatal=False, ref=msg_id,
            )
            return

        # Dispatch by type.
        handler = self._handlers.get(msg_type)
        if handler:
            try:
                await handler(self, agent, data)
            except Exception as exc:
                log.error("Error handling %s: %s", msg_type, exc, exc_info=True)
                await self._send_error(
                    agent.ws, ERROR_INTERNAL_ERROR, str(exc), fatal=False, ref=msg_id,
                )
        else:
            # Unknown type in a known namespace: silently ignore per §11.2.
            log.debug("Ignoring unrecognized message type: %s", msg_type)

    # -----------------------------------------------------------------
    # Message handlers
    # -----------------------------------------------------------------

    async def _handle_chat_response(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle chat.response — agent sends a message to the user."""
        payload = data["payload"]
        content = payload.get("content", "")

        # Store as assistant message.
        self.store.store_chat_message(
            data["id"], agent.channel_id, "assistant", content,
        )

        # Notify browser.
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "chat.response",
            "event": {
                "id": data["id"],
                "ref": data.get("ref"),
                "content": content,
            },
        })

    async def _handle_activity_delta(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle activity.delta — streaming harness output."""
        payload = data["payload"]
        delta = payload.get("delta", {})
        index = payload.get("index", 0)

        # Store activity entry.
        self.store.store_activity(
            agent.channel_id,
            delta.get("type", "text"),
            delta,
        )

        # Notify browser.
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "activity.delta",
            "event": {
                "delta": delta,
                "index": index,
            },
        })

    async def _handle_activity_ping(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle activity.ping — agent is working but no new output."""
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "activity.ping",
            "event": {},
        })

    async def _handle_activity_end(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle activity.end — agent completed a unit of work."""
        payload = data["payload"]
        reason = payload.get("reason", "complete")
        usage = payload.get("usage")

        # Store activity entry.
        self.store.store_activity(
            agent.channel_id, "end",
            {"reason": reason, "usage": usage},
        )

        # Update channel status based on reason.
        if reason in ("complete", "waiting"):
            self.store.update_channel_status(agent.channel_id, "idle")
        elif reason == "error":
            self.store.update_channel_status(agent.channel_id, "error")

        # Notify browser.
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "activity.end",
            "event": {"reason": reason, "usage": usage},
        })

    async def _handle_tool_use(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle tool.use — agent is about to execute a tool."""
        payload = data["payload"]
        tool_use_id = payload.get("tool_use_id", "")
        name = payload.get("name", "")
        tool_input = payload.get("input", {})

        # Store tool use.
        record = self.store.store_tool_use(tool_use_id, agent.channel_id, name, tool_input)

        # Store as activity entry.
        self.store.store_activity(
            agent.channel_id, "tool_use",
            {"id": tool_use_id, "name": name, "input": tool_input},
        )

        # Notify browser.
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "tool.use",
            "event": {
                "tool_use_id": tool_use_id,
                "name": name,
                "input": tool_input,
                "created_at": record.created_at,
            },
        })

    async def _handle_tool_result(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle tool.result — tool execution completed."""
        payload = data["payload"]
        tool_use_id = payload.get("tool_use_id", "")
        content = payload.get("content", "")
        is_error = payload.get("is_error", False)

        # Store tool result.
        completed_at = self.store.store_tool_result(tool_use_id, content, is_error)

        # Store as activity entry.
        self.store.store_activity(
            agent.channel_id, "tool_result",
            {"tool_use_id": tool_use_id, "content": content, "is_error": is_error},
        )

        # Notify browser.
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "tool.result",
            "event": {
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": is_error,
                "completed_at": completed_at,
            },
        })

    async def _handle_agent_goodbye(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle agent.goodbye — agent is disconnecting gracefully."""
        payload = data["payload"]
        reason = payload.get("reason", "shutdown_ack")
        log.info("Agent %s goodbye: %s", agent.agent_id[:8], reason)

        if reason == "completed":
            self.store.update_channel_status(agent.channel_id, "closed")
        elif reason == "error":
            self.store.update_channel_status(agent.channel_id, "error")

    async def _handle_agent_error(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle agent.error from the agent side."""
        payload = data["payload"]
        code = payload.get("code", "unknown")
        message = payload.get("message", "")
        fatal = payload.get("fatal", False)

        log.error("Agent %s error [%s]: %s (fatal=%s)", agent.agent_id[:8], code, message, fatal)

        # Notify browser.
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "agent.error",
            "event": {"code": code, "message": message, "fatal": fatal},
        })

        if fatal:
            self.store.update_channel_status(agent.channel_id, "error")

    # Handler dispatch table.
    _handlers: dict[str, Any] = {
        CHAT_RESPONSE: _handle_chat_response,
        ACTIVITY_DELTA: _handle_activity_delta,
        ACTIVITY_PING: _handle_activity_ping,
        ACTIVITY_END: _handle_activity_end,
        TOOL_USE: _handle_tool_use,
        TOOL_RESULT: _handle_tool_result,
        AGENT_GOODBYE: _handle_agent_goodbye,
        AGENT_ERROR: _handle_agent_error,
    }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _cleanup_agent(self, agent: AgentConnection) -> None:
        """Remove an agent from tracking and update channel status."""
        self._agents.pop(agent.agent_id, None)
        self._channel_to_agent.pop(agent.channel_id, None)

        # Mark channel as idle (not closed — agent may reconnect).
        channel = self.store.get_channel(agent.channel_id)
        if channel and channel.status == "active":
            self.store.update_channel_status(agent.channel_id, "idle")

        log.info("Agent %s disconnected from channel %s", agent.agent_id[:8], agent.channel_id[:8])

    async def _send_error(
        self,
        ws: ServerConnection,
        code: str,
        message: str,
        fatal: bool = False,
        ref: str | None = None,
    ) -> None:
        """Send an agent.error message."""
        envelope = make_envelope(
            AGENT_ERROR,
            {"code": code, "message": message, "fatal": fatal},
            ref=ref,
        )
        try:
            await ws.send(json.dumps(envelope))
        except Exception:
            pass

        if fatal:
            try:
                await ws.close()
            except Exception:
                pass

    async def _notify_browser(
        self,
        channel_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Send an E2EE-encrypted notification to all browser sessions."""
        if self._broadcast:
            try:
                await self._broadcast(channel_id, payload)
            except Exception as exc:
                log.error("Failed to broadcast to browser: %s", exc)
