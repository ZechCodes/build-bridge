"""Local WebSocket server for the Build Agent Protocol.

Accepts agent connections on localhost, handles the BAP handshake and message
lifecycle, persists data to SQLite, and emits E2EE-encrypted notifications
to browser clients via a broadcast callback.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

import websockets
from websockets.asyncio.server import serve as ws_serve, ServerConnection

from build_bridge.agent_protocol import (
    ACTIVITY_DELTA,
    ACTIVITY_END,
    ACTIVITY_PING,
    AGENT_ERROR,
    AGENT_GOODBYE,
    AGENT_HELLO,
    AGENT_CONFIGURED,
    AGENT_FILE_CHANGES,
    AGENT_STATE_UPDATE,
    AGENT_SYSTEM_MESSAGE,
    AGENT_TO_CLIENT,
    BIDIRECTIONAL,
    CHAT_RESPONSE,
    ERROR_CAPABILITY_MISMATCH,
    ERROR_INTERNAL_ERROR,
    ERROR_INVALID_MESSAGE,
    ERROR_PROTOCOL_VIOLATION,
    ERROR_RECONNECT_FAILED,
    ERROR_UNKNOWN_TYPE,
    INTERACTION_REQUEST,
    INTERACTION_RESPONSE,
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
from build_bridge.agent_store import AgentStore
from build_bridge.complications import find_git_repo, _run_git

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
        e2ee_store: Any = None,
        complications: Any = None,
    ) -> None:
        self.store = store
        self._broadcast = broadcast
        self._e2ee_store = e2ee_store  # MessageStore for checking unread messages
        self._complications = complications  # ComplicationRegistry
        self._host = host
        self._port = port
        self._agents: dict[str, AgentConnection] = {}  # agent_id -> connection
        self._channel_to_agent: dict[str, str] = {}  # channel_id -> agent_id
        self._compact_futures: dict[str, asyncio.Future[str]] = {}  # channel_id -> pending summary
        self._cancel_ack_events: dict[str, asyncio.Event] = {}  # channel_id -> one-shot event
        # Tracks the activity_log row id currently receiving streaming text deltas.
        # Consecutive text deltas UPDATE this row; any non-text event closes the run
        # so the next text delta starts a fresh row. Mirrors the live UI which
        # coalesces reasoning deltas until a tool use or other event interrupts.
        self._active_text_row: dict[str, str] = {}  # channel_id -> activity row id
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

    async def stop(self, *, notify_agents: bool = True) -> None:
        """Shut down the server.

        When *notify_agents* is True, each connected agent receives an
        ``agent.shutdown`` envelope so it exits cleanly. Pass False during a
        ``--keep-agents`` restart — the agent subprocesses will outlive this
        daemon and reconnect to its successor.
        """
        if notify_agents:
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
        model: str | None = None,
        effort: str | None = None,
        plan_mode: bool | None = None,
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
        if model:
            payload["model"] = model
        if effort:
            payload["effort"] = effort
        if plan_mode is not None:
            payload["plan_mode"] = plan_mode

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
            log.warning("send_cancel: no agent mapped for channel %s", channel_id[:8])
            return False
        agent = self._agents.get(agent_id)
        if not agent:
            log.warning("send_cancel: agent %s not found for channel %s", agent_id[:8], channel_id[:8])
            return False

        envelope = make_envelope("chat.cancel", {})
        try:
            await agent.ws.send(json.dumps(envelope))
            log.info("Sent chat.cancel to agent %s on channel %s", agent_id[:8], channel_id[:8])
            return True
        except Exception as exc:
            log.error("Failed to send chat.cancel to agent: %s", exc)
            return False

    async def wait_for_cancel_ack(self, channel_id: str, timeout: float = 3.0) -> bool:
        """Wait for the agent to acknowledge cancel via activity.end.

        Returns True if activity.end was received within the timeout.
        """
        event = asyncio.Event()
        self._cancel_ack_events[channel_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            self._cancel_ack_events.pop(channel_id, None)

    async def send_interaction_response(
        self,
        channel_id: str,
        interaction_id: str,
        selected_option: str | None,
        freeform_response: str | None,
        selected_options: list[str] | None = None,
    ) -> bool:
        """Forward an interaction response to the agent on a channel."""
        agent_id = self._channel_to_agent.get(channel_id)
        if not agent_id:
            return False
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        envelope = make_envelope(INTERACTION_RESPONSE, {
            "interaction_id": interaction_id,
            "selected_option": selected_option,
            "freeform_response": freeform_response,
            "selected_options": selected_options,
        })
        try:
            await agent.ws.send(json.dumps(envelope))
            log.info("Sent interaction.response to agent %s (interaction=%s)", agent_id[:8], interaction_id[:12])
            return True
        except Exception as exc:
            log.error("Failed to send interaction.response to agent: %s", exc)
            return False

    async def send_system_instruction(
        self,
        channel_id: str,
        content: str,
    ) -> bool:
        """Send an instruction to the agent WITHOUT storing in chat history.

        Used for system-level commands (like plan mode toggle) that should
        reach the agent but not appear in the user's chat.
        """
        agent_id = self._channel_to_agent.get(channel_id)
        if not agent_id:
            return False
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        envelope = make_envelope("chat.message", {"role": "system", "content": content})
        try:
            await agent.ws.send(json.dumps(envelope))
            log.info("Sent system instruction to agent %s on channel %s", agent_id[:8], channel_id[:8])
            return True
        except Exception as exc:
            log.error("Failed to send system instruction: %s", exc)
            return False

    async def request_summary(self, channel_id: str) -> str | None:
        """Ask the agent to summarize the session. Returns the summary text.

        Sends a system instruction and waits for the agent's chat.response.
        Returns None if no agent is connected.
        """
        if channel_id in self._compact_futures:
            log.warning("Compact already in progress for channel %s", channel_id[:8])
            return None

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._compact_futures[channel_id] = future

        prompt = (
            "Summarize this session for context continuity. Include:\n"
            "1. What work is being done (projects, files, features)\n"
            "2. Current progress and state\n"
            "3. Directions or preferences the user has given\n"
            "4. Planned next steps\n\n"
            "Be concise but thorough. This summary will be the starting context "
            "for a fresh session. Use the send tool to deliver your summary."
        )

        sent = await self.send_system_instruction(channel_id, prompt)
        if not sent:
            self._compact_futures.pop(channel_id, None)
            return None

        try:
            return await future
        except asyncio.CancelledError:
            return None
        finally:
            self._compact_futures.pop(channel_id, None)

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

        # Always start with plan mode off — it's opt-in per session.
        self.store.update_plan_mode(channel.id, False)

        # Build history for agent.configured.
        # Limit to recent entries to avoid exceeding the WebSocket frame limit.
        all_chat = self.store.get_chat_history(channel.id, since=channel.session_start_at)
        chat_history = [
            {"role": m.role, "content": m.content}
            for m in all_chat[-50:]
        ]
        all_activity = self.store.get_activity_history(channel.id, since=channel.session_start_at)
        activity_history = [
            json.loads(e.data) for e in all_activity[-200:]
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
                "working_directory": channel.working_directory or "",
                "auto_approve_tools": bool(channel.auto_approve_tools),
                "history": {
                    "chat": chat_history,
                    "activity": activity_history,
                },
            },
            ref=data["id"],
        )
        await ws.send(json.dumps(configured))

        # Forward any unread E2EE messages to the agent so it can process
        # messages that arrived while no agent was running.
        if self._e2ee_store:
            try:
                unread = self._e2ee_store.get_unread_messages(channel.id)
                if unread:
                    log.info("Forwarding %d unread message(s) to agent on channel %s", len(unread), channel.id[:8])
                    for msg in unread:
                        chat_msg = make_envelope("chat.message", {
                            "role": "user",
                            "content": msg.content,
                        }, msg_id=msg.id)
                        await ws.send(json.dumps(chat_msg))
            except Exception as exc:
                log.warning("Failed to forward unread messages: %s", exc)

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

    @staticmethod
    def _agent_display_name(agent: AgentConnection) -> str:
        """Get the display name for an agent (e.g., 'Claude Code')."""
        from build_bridge.harness_registry import get_harness
        info = get_harness(agent.harness)
        return info.name if info else "Device"

    async def _handle_chat_response(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle chat.response — agent sends a message to the user."""
        payload = data["payload"]
        content = payload.get("content", "")

        self._close_text_run(agent.channel_id)

        # If a compact is pending for this channel, intercept the response.
        future = self._compact_futures.get(agent.channel_id)
        if future and not future.done():
            future.set_result(content)
            return  # Don't store or notify browser — compact handler will manage it.

        # Resolve [[file]] and [[diff]] markers into embedded content.
        content = await self._resolve_file_embeds(agent.channel_id, content)

        # Store as assistant message.
        suggested_actions = payload.get("suggested_actions")
        self.store.store_chat_message(
            data["id"], agent.channel_id, "assistant", content,
            suggested_actions=suggested_actions or None,
        )

        # Notify browser.
        event: dict[str, Any] = {
            "id": data["id"],
            "ref": data.get("ref"),
            "content": content,
            "sender": self._agent_display_name(agent),
        }
        if suggested_actions:
            event["suggested_actions"] = suggested_actions
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "chat.response",
            "event": event,
        })

    # ---- File / diff embed resolution ----

    _EMBED_FILE_RE = re.compile(r"\[\[file(?:\s+(\d+):(\d+))?\]\]\(([^)]+)\)")
    _EMBED_DIFF_RE = re.compile(r"\[\[diff\]\]\(([^)]+)\)")
    _MAX_EMBED_SIZE = 50 * 1024  # 50 KB per embed
    _MAX_EMBEDS = 10
    _MAX_IMAGE_EMBED_BYTES = 2 * 1024 * 1024  # 2 MiB per image embed
    _IMAGE_MIME = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "avif": "image/avif",
        "bmp": "image/bmp",
        "svg": "image/svg+xml",
    }

    def _resolve_safe_path(self, base_dir: str, relative_path: str) -> str | None:
        """Resolve relative_path under base_dir, or accept absolute paths."""
        if os.path.isabs(relative_path):
            return str(Path(relative_path).resolve())
        base = Path(base_dir).resolve()
        target = (base / relative_path).resolve()
        if target != base and not str(target).startswith(str(base) + os.sep):
            return None
        return str(target)

    async def _resolve_file_embeds(
        self, channel_id: str, content: str,
    ) -> str:
        """Replace [[file]] and [[diff]] markers with embedded content tags."""
        if "[[file" not in content and "[[diff" not in content:
            return content

        ch = self.store.get_channel(channel_id)
        cwd = os.path.expanduser(ch.working_directory) if ch and ch.working_directory else os.getcwd()

        embed_count = 0

        async def _replace_file(m: re.Match) -> str:
            nonlocal embed_count
            if embed_count >= self._MAX_EMBEDS:
                return m.group(0)
            embed_count += 1

            start_line = int(m.group(1)) if m.group(1) else None
            end_line = int(m.group(2)) if m.group(2) else None
            rel_path = m.group(3).strip()

            resolved = self._resolve_safe_path(cwd, rel_path)
            if resolved is None or not os.path.isfile(resolved):
                return f"[Could not read: {rel_path}]"

            ext = os.path.splitext(rel_path)[1].lstrip(".").lower()

            # Images: read binary, base64-encode, ship in a <build-image>
            # tag the frontend can render directly as a data-URI. Line
            # ranges are meaningless for images — the `start_line`/
            # `end_line` capture groups are ignored on this branch.
            if ext in self._IMAGE_MIME:
                try:
                    with open(resolved, "rb") as f:
                        raw = f.read(self._MAX_IMAGE_EMBED_BYTES + 1)
                except OSError:
                    return f"[Could not read: {rel_path}]"
                if len(raw) > self._MAX_IMAGE_EMBED_BYTES:
                    return f"[Image too large: {rel_path}]"
                mime = self._IMAGE_MIME[ext]
                b64 = base64.b64encode(raw).decode("ascii")
                return (
                    f'<build-image path="{rel_path}" mime="{mime}">\n'
                    f"{b64}\n"
                    f"</build-image>"
                )

            try:
                with open(resolved, "r", errors="replace") as f:
                    file_content = f.read(self._MAX_EMBED_SIZE + 1)
            except OSError:
                return f"[Could not read: {rel_path}]"

            truncated = len(file_content) > self._MAX_EMBED_SIZE
            if truncated:
                file_content = file_content[: self._MAX_EMBED_SIZE]

            if start_line is not None and end_line is not None:
                lines = file_content.split("\n")
                # Convert to 0-indexed, clamp to bounds.
                s = max(0, start_line - 1)
                e = min(len(lines), end_line)
                file_content = "\n".join(lines[s:e])
                line_attr = f' lines="{start_line}-{end_line}"'
            else:
                line_attr = ""

            lang = ext

            return (
                f'<build-file path="{rel_path}" lang="{lang}"{line_attr}>\n'
                f"{file_content}\n"
                f"</build-file>"
            )

        async def _replace_diff(m: re.Match) -> str:
            nonlocal embed_count
            if embed_count >= self._MAX_EMBEDS:
                return m.group(0)
            embed_count += 1

            raw = m.group(1).strip()
            if "|" in raw:
                parts = raw.split("|", 1)
                path1, path2 = parts[0].strip(), parts[1].strip()
                resolved1 = self._resolve_safe_path(cwd, path1)
                resolved2 = self._resolve_safe_path(cwd, path2)
                if not resolved1 or not resolved2:
                    return f"[Could not diff: {raw}]"
                repo = find_git_repo(resolved1) or cwd
                diff_out, _ = await _run_git(repo, ["diff", "--no-index", "--", resolved1, resolved2])
                display_path = f"{path1} vs {path2}"
            else:
                rel_path = raw
                resolved = self._resolve_safe_path(cwd, rel_path)
                if not resolved:
                    return f"[Could not diff: {rel_path}]"
                repo = find_git_repo(resolved)
                if not repo:
                    return f"[Not in a git repo: {rel_path}]"
                try:
                    file_repo_rel = str(Path(resolved).resolve().relative_to(Path(repo).resolve()))
                except ValueError:
                    file_repo_rel = rel_path
                diff_out, _ = await _run_git(repo, ["diff", "--", file_repo_rel])
                if not diff_out:
                    status_out, _ = await _run_git(repo, ["status", "--porcelain=v1", "--", file_repo_rel])
                    if status_out.startswith("??"):
                        diff_out, _ = await _run_git(repo, ["diff", "--no-index", "/dev/null", file_repo_rel])
                if not diff_out:
                    diff_out, _ = await _run_git(repo, ["diff", "--cached", "--", file_repo_rel])
                if not diff_out:
                    # Fall back to diff between last two commits that touched this file.
                    log_out, _ = await _run_git(repo, ["log", "--pretty=format:%H", "-2", "--", file_repo_rel])
                    commits = log_out.strip().splitlines()
                    if len(commits) == 2:
                        diff_out, _ = await _run_git(repo, ["diff", commits[1], commits[0], "--", file_repo_rel])
                    elif len(commits) == 1:
                        diff_out, _ = await _run_git(repo, ["diff", f"{commits[0]}~1", commits[0], "--", file_repo_rel])
                display_path = rel_path

            if not diff_out:
                return f"[No diff: {display_path}]"

            if len(diff_out) > self._MAX_EMBED_SIZE:
                diff_out = diff_out[: self._MAX_EMBED_SIZE]

            return (
                f'<build-diff path="{display_path}">\n'
                f"{diff_out}\n"
                f"</build-diff>"
            )

        # Process file embeds.
        parts = []
        last_end = 0
        for m in self._EMBED_FILE_RE.finditer(content):
            parts.append(content[last_end:m.start()])
            parts.append(await _replace_file(m))
            last_end = m.end()
        parts.append(content[last_end:])
        content = "".join(parts)

        # Process diff embeds.
        parts = []
        last_end = 0
        for m in self._EMBED_DIFF_RE.finditer(content):
            parts.append(content[last_end:m.start()])
            parts.append(await _replace_diff(m))
            last_end = m.end()
        parts.append(content[last_end:])
        return "".join(parts)

    async def _handle_activity_delta(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle activity.delta — streaming harness output."""
        payload = data["payload"]
        delta = payload.get("delta", {})
        index = payload.get("index", 0)
        delta_type = delta.get("type", "text")

        # Coalesce consecutive text deltas into a single activity row so history
        # replay matches the live UI (which merges consecutive reasoning deltas
        # into one entry). The first text delta inserts a row; subsequent deltas
        # append to it until _close_text_run is called by a non-text event.
        if delta_type == "text":
            active_id = self._active_text_row.get(agent.channel_id)
            if active_id:
                self.store.append_text_activity(active_id, delta.get("text", ""))
            else:
                entry = self.store.store_activity(agent.channel_id, "text", delta)
                self._active_text_row[agent.channel_id] = entry.id
        else:
            self._close_text_run(agent.channel_id)
            self.store.store_activity(agent.channel_id, delta_type, delta)

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

        self._close_text_run(agent.channel_id)

        # Signal any pending cancel acknowledgement waiter.
        ack_event = self._cancel_ack_events.get(agent.channel_id)
        if ack_event:
            ack_event.set()

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

        self._close_text_run(agent.channel_id)

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

        # Trigger complications evaluation (debounced).
        if self._complications:
            channel = self.store.get_channel(agent.channel_id)
            working_dir = channel.working_directory if channel else ""
            asyncio.create_task(
                self._complications.on_tool_event(
                    agent.channel_id, name, tool_input, working_dir,
                )
            )

    async def _handle_tool_result(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle tool.result — tool execution completed."""
        payload = data["payload"]
        tool_use_id = payload.get("tool_use_id", "")
        content = payload.get("content", "")
        is_error = payload.get("is_error", False)

        self._close_text_run(agent.channel_id)

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

    async def _handle_interaction_request(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle interaction.request — agent is asking the user a question."""
        payload = data["payload"]
        interaction_id = payload.get("interaction_id", "")
        kind = payload.get("kind", "question")
        question = payload.get("question", "")
        options = payload.get("options", [])
        allow_freeform = payload.get("allow_freeform", True)
        plan = payload.get("plan")
        multiselect = payload.get("multiselect", False)

        self._close_text_run(agent.channel_id)

        # Persist as a chat message with metadata.
        self.store.store_interaction(
            interaction_id, agent.channel_id, question, kind, options, allow_freeform, plan,
            multiselect=multiselect,
        )

        # Broadcast to browser (include sender name).
        event = {**payload, "sender": self._agent_display_name(agent)}
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "interaction.request",
            "event": event,
        })

    async def _handle_state_update(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle agent.state_update — agent reports its state."""
        payload = data["payload"]
        plan_mode = payload.get("plan_mode")
        if plan_mode is not None:
            self.store.update_plan_mode(agent.channel_id, plan_mode)

        # Persist resume cursor for session resume on restart.
        resume_cursor = payload.get("resume_cursor")
        if resume_cursor is not None:
            self.store.update_resume_cursor(agent.channel_id, resume_cursor)

        # Persist read status in E2EE message store.
        read_ids = payload.get("read_message_ids")
        if read_ids and self._e2ee_store:
            for mid in read_ids:
                try:
                    self._e2ee_store.mark_read(mid)
                except Exception:
                    pass

        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "agent.state_update",
            "event": payload,
        })

    async def _handle_system_message(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Handle agent.system_message — broadcast as ephemeral system notification."""
        payload = data["payload"]
        await self._notify_browser(agent.channel_id, {
            "action": "system_message",
            "channel_id": agent.channel_id,
            "text": payload.get("text", ""),
        })

    async def _handle_file_changes(
        self, agent: AgentConnection, data: dict[str, Any],
    ) -> None:
        """Forward agent.file_changes to the browser for live diff refresh."""
        payload = data["payload"] or {}
        paths = payload.get("paths") or []
        if not isinstance(paths, list):
            return
        await self._notify_browser(agent.channel_id, {
            "action": "agent_event",
            "channel_id": agent.channel_id,
            "event_type": "agent.file_changes",
            "event": {"paths": paths},
        })

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
        INTERACTION_REQUEST: _handle_interaction_request,
        AGENT_STATE_UPDATE: _handle_state_update,
        AGENT_SYSTEM_MESSAGE: _handle_system_message,
        AGENT_FILE_CHANGES: _handle_file_changes,
    }

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _close_text_run(self, channel_id: str) -> None:
        """End the current coalesced-text activity row so the next text delta
        starts a fresh row. See ``_handle_activity_delta`` for the rationale.
        """
        self._active_text_row.pop(channel_id, None)

    def _cleanup_agent(self, agent: AgentConnection) -> None:
        """Remove an agent from tracking and update channel status."""
        self._close_text_run(agent.channel_id)
        self._agents.pop(agent.agent_id, None)
        self._channel_to_agent.pop(agent.channel_id, None)

        # Cancel any pending compact summary request.
        future = self._compact_futures.pop(agent.channel_id, None)
        if future and not future.done():
            future.cancel()

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
