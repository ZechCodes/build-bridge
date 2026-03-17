"""Agent wrapper — bridges a coding harness to the device client via BAP.

The wrapper connects to the device client's local WebSocket server, performs
the BAP handshake, and provides:

1. Chat MCP tools (read_unread, send) for the harness to communicate with the user.
2. Methods for harness-specific code to report activity and tool use events.
3. Automatic handling of incoming chat.message, chat.cancel, and agent.shutdown.

Usage (harness-agnostic):

    wrapper = AgentWrapper(
        port=9783,
        harness="claude-code",
        model="claude-sonnet-4-20250514",
    )

    # Connect and perform handshake.
    await wrapper.connect()

    # Access chat MCP handlers for your harness integration.
    wrapper.chat_mcp.handle_read_unread  # MCP tool handler
    wrapper.chat_mcp.handle_send         # MCP tool handler

    # Report events from the harness.
    await wrapper.emit_activity_delta("text", "Analyzing code...")
    await wrapper.emit_tool_use("tu_1", "read_file", {"path": "/src/main.py"})
    await wrapper.emit_tool_result("tu_1", "file contents...", is_error=False)
    await wrapper.emit_activity_end("complete", usage={...})

    # Run the receive loop (blocks, handles incoming messages).
    await wrapper.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from build_client.agent_protocol import (
    ACTIVITY_DELTA,
    ACTIVITY_END,
    ACTIVITY_PING,
    AGENT_CONFIGURED,
    AGENT_ERROR,
    AGENT_GOODBYE,
    AGENT_HELLO,
    AGENT_SHUTDOWN,
    CHAT_CANCEL,
    CHAT_MESSAGE,
    CHAT_RESPONSE,
    PROTOCOL_VERSION,
    TOOL_RESULT,
    TOOL_USE,
    make_envelope,
    now_iso,
    validate_envelope,
)
from build_client.chat_mcp import ChatMCP

log = logging.getLogger(__name__)

# Chat MCP tool names — filtered from tool.* namespace (§8.4).
CHAT_MCP_TOOLS = frozenset({"read_unread", "send"})

# Reconnect backoff parameters (§2.4).
INITIAL_BACKOFF_MS = 500
MAX_BACKOFF_MS = 30_000
BACKOFF_FACTOR = 2
JITTER_PCT = 0.25


@dataclass
class WrapperConfig:
    """Configuration received from agent.configured."""

    channel_id: str
    system_prompt: str
    chat_instructions: str
    chat_history: list[dict[str, Any]]
    activity_history: list[dict[str, Any]]


# Callback types.
CancelCallback = Callable[[], Coroutine[Any, Any, None]]
ShutdownCallback = Callable[[str], Coroutine[Any, Any, None]]


class AgentWrapper:
    """BAP agent wrapper — connects to the device client and manages Chat MCP.

    Provides the Chat MCP tools for the harness and methods for the harness
    to report activity and tool use events to the device client.
    """

    def __init__(
        self,
        port: int = 9783,
        host: str = "127.0.0.1",
        harness: str = "claude-code",
        model: str = "claude-sonnet-4-20250514",
        capabilities: list[str] | None = None,
        agent_id: str | None = None,
        reconnect: bool = False,
        on_cancel: CancelCallback | None = None,
        on_shutdown: ShutdownCallback | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._harness = harness
        self._model = model
        self._capabilities = capabilities or ["chat", "activity", "tools"]
        self._agent_id = agent_id or f"agt_{uuid.uuid4().hex[:8]}"
        self._reconnect = reconnect
        self._on_cancel = on_cancel
        self._on_shutdown = on_shutdown

        # Chat MCP instance — tools share this state.
        self.chat_mcp = ChatMCP(on_send=self._emit_chat_response)

        # Connection state.
        self._ws: Any = None
        self._config: WrapperConfig | None = None
        self._connected = asyncio.Event()
        self._shutdown_requested = asyncio.Event()
        self._activity_index = 0

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def channel_id(self) -> str | None:
        return self._config.channel_id if self._config else None

    @property
    def config(self) -> WrapperConfig | None:
        return self._config

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    def mark_reconnect(self) -> None:
        """Mark subsequent connections as reconnections."""
        self._reconnect = True

    # -----------------------------------------------------------------
    # Connection
    # -----------------------------------------------------------------

    async def connect(self) -> WrapperConfig:
        """Connect to the device client WS and perform BAP handshake.

        Returns the WrapperConfig from agent.configured.
        Raises ConnectionError on failure.
        """
        url = f"ws://{self._host}:{self._port}"
        log.info("Connecting to device client at %s ...", url)

        self._ws = await ws_connect(url)

        # Send agent.hello.
        hello = make_envelope(AGENT_HELLO, {
            "agent_id": self._agent_id,
            "harness": self._harness,
            "capabilities": self._capabilities,
            "model": self._model,
            "reconnect": self._reconnect,
        })
        await self._ws.send(json.dumps(hello))
        log.debug("Sent agent.hello (agent_id=%s)", self._agent_id[:8])

        # Wait for agent.configured.
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
        except asyncio.TimeoutError:
            raise ConnectionError("Timeout waiting for agent.configured")

        data = json.loads(raw)
        valid, err = validate_envelope(data)
        if not valid:
            raise ConnectionError(f"Invalid agent.configured response: {err}")

        if data["type"] == AGENT_ERROR:
            payload = data["payload"]
            raise ConnectionError(
                f"Device client rejected connection: [{payload.get('code')}] {payload.get('message')}"
            )

        if data["type"] != AGENT_CONFIGURED:
            raise ConnectionError(f"Expected agent.configured, got {data['type']}")

        payload = data["payload"]
        self._config = WrapperConfig(
            channel_id=payload["channel_id"],
            system_prompt=payload.get("system_prompt", ""),
            chat_instructions=payload.get("chat_instructions", ""),
            chat_history=payload.get("history", {}).get("chat", []),
            activity_history=payload.get("history", {}).get("activity", []),
        )

        self._connected.set()
        log.info(
            "Connected to channel %s (harness=%s, model=%s)",
            self._config.channel_id[:8], self._harness, self._model,
        )
        return self._config

    async def disconnect(self, reason: str = "shutdown_ack") -> None:
        """Send agent.goodbye and close the connection."""
        if self._ws:
            try:
                goodbye = make_envelope(AGENT_GOODBYE, {"reason": reason})
                await self._ws.send(json.dumps(goodbye))
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected.clear()
        log.info("Disconnected (reason=%s)", reason)

    # -----------------------------------------------------------------
    # Receive loop
    # -----------------------------------------------------------------

    async def run(self) -> None:
        """Run the message receive loop. Blocks until shutdown or disconnect.

        Handles incoming messages from the device client:
        - chat.message → queues in ChatMCP
        - chat.cancel → calls on_cancel callback
        - agent.shutdown → calls on_shutdown callback, sends goodbye
        - agent.error → logs the error
        """
        if not self._ws:
            raise RuntimeError("Not connected — call connect() first")

        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("Received invalid JSON from device client")
                    continue

                valid, err = validate_envelope(data)
                if not valid:
                    log.warning("Invalid envelope from device client: %s", err)
                    continue

                await self._handle_message(data)

                if self._shutdown_requested.is_set():
                    break

        except ConnectionClosed:
            log.info("Connection to device client closed")
        finally:
            self._connected.clear()

    async def _handle_message(self, data: dict[str, Any]) -> None:
        """Dispatch an incoming message from the device client."""
        msg_type = data["type"]
        payload = data["payload"]

        if msg_type == CHAT_MESSAGE:
            # Queue user message for Chat MCP read_unread.
            content = payload.get("content", "")
            await self.chat_mcp.queue_message(content)
            log.debug("Queued user message (%d unread)", self.chat_mcp.unread_count)

        elif msg_type == CHAT_CANCEL:
            log.info("Received chat.cancel")
            if self._on_cancel:
                await self._on_cancel()
            # Emit activity.end with cancelled reason.
            await self.emit_activity_end("cancelled")

        elif msg_type == AGENT_SHUTDOWN:
            reason = payload.get("reason", "client_shutdown")
            log.info("Received agent.shutdown (reason=%s)", reason)
            if self._on_shutdown:
                await self._on_shutdown(reason)
            await self.disconnect("shutdown_ack")
            self._shutdown_requested.set()

        elif msg_type == AGENT_ERROR:
            code = payload.get("code", "unknown")
            message = payload.get("message", "")
            fatal = payload.get("fatal", False)
            log.error("Device client error [%s]: %s (fatal=%s)", code, message, fatal)
            if fatal:
                self._shutdown_requested.set()

        else:
            log.debug("Ignoring message type from device client: %s", msg_type)

    # -----------------------------------------------------------------
    # Emit events (called by harness-specific code)
    # -----------------------------------------------------------------

    async def emit_activity_delta(
        self,
        delta_type: str,
        text: str,
        index: int | None = None,
    ) -> None:
        """Emit activity.delta — streaming harness output."""
        if index is None:
            index = self._activity_index
            self._activity_index += 1

        envelope = make_envelope(ACTIVITY_DELTA, {
            "delta": {"type": delta_type, "text": text},
            "index": index,
        })
        await self._send(envelope)

    async def emit_activity_ping(self) -> None:
        """Emit activity.ping — agent is working but no new output."""
        envelope = make_envelope(ACTIVITY_PING, {})
        await self._send(envelope)

    async def emit_activity_end(
        self,
        reason: str = "complete",
        usage: dict[str, int] | None = None,
    ) -> None:
        """Emit activity.end — work unit completed."""
        payload: dict[str, Any] = {"reason": reason}
        if usage:
            payload["usage"] = usage

        envelope = make_envelope(ACTIVITY_END, payload)
        await self._send(envelope)

        # Reset activity index for next turn.
        self._activity_index = 0

    async def emit_tool_use(
        self,
        tool_use_id: str,
        name: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Emit tool.use — agent is about to execute a tool.

        Chat MCP tools (read_unread, send) are automatically filtered (§8.4).
        """
        if name in CHAT_MCP_TOOLS:
            return  # Filtered per §8.4.

        envelope = make_envelope(TOOL_USE, {
            "tool_use_id": tool_use_id,
            "name": name,
            "input": tool_input,
        })
        await self._send(envelope)

    async def emit_tool_result(
        self,
        tool_use_id: str,
        content: str | list[dict[str, Any]],
        is_error: bool = False,
        *,
        tool_name: str,
    ) -> None:
        """Emit tool.result — tool execution completed.

        Chat MCP tools are automatically filtered (§8.4).
        tool_name is required to ensure filtering is never bypassed.
        """
        if tool_name in CHAT_MCP_TOOLS:
            return  # Filtered per §8.4.

        envelope = make_envelope(TOOL_RESULT, {
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        })
        await self._send(envelope)

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    async def _emit_chat_response(self, message: str) -> None:
        """Handle Chat MCP send tool — emit chat.response to device client."""
        envelope = make_envelope(CHAT_RESPONSE, {"content": message})
        await self._send(envelope)
        log.debug("Emitted chat.response (%d chars)", len(message))

    async def _send(self, envelope: dict[str, Any]) -> None:
        """Send an envelope to the device client."""
        if not self._ws:
            log.warning("Cannot send — not connected")
            return
        try:
            await self._ws.send(json.dumps(envelope))
        except ConnectionClosed:
            log.warning("Connection lost while sending %s", envelope.get("type"))
            self._connected.clear()

    async def _send_error(
        self,
        code: str,
        message: str,
        fatal: bool = False,
        ref: str | None = None,
    ) -> None:
        """Send agent.error to the device client."""
        envelope = make_envelope(
            AGENT_ERROR,
            {"code": code, "message": message, "fatal": fatal},
            ref=ref,
        )
        await self._send(envelope)


# ---------------------------------------------------------------------------
# Convenience: connect + run with auto-reconnect (§2.4)
# ---------------------------------------------------------------------------


async def run_agent(
    wrapper: AgentWrapper,
    auto_reconnect: bool = True,
) -> None:
    """Connect the wrapper and run the receive loop with auto-reconnect.

    Implements §2.4 exponential backoff with jitter.
    """
    import random

    backoff_ms = INITIAL_BACKOFF_MS

    while True:
        try:
            await wrapper.connect()
            backoff_ms = INITIAL_BACKOFF_MS  # Reset on success.
            await wrapper.run()

            if wrapper.shutdown_requested:
                log.info("Shutdown requested — exiting")
                return

        except (ConnectionError, ConnectionClosed, OSError) as exc:
            log.warning("Connection error: %s", exc)

        if not auto_reconnect:
            return

        # Exponential backoff with jitter.
        jitter = random.uniform(1 - JITTER_PCT, 1 + JITTER_PCT)
        delay_ms = backoff_ms * jitter
        delay_s = delay_ms / 1000

        log.info("Reconnecting in %.1fs...", delay_s)
        await asyncio.sleep(delay_s)

        backoff_ms = min(backoff_ms * BACKOFF_FACTOR, MAX_BACKOFF_MS)
        wrapper.mark_reconnect()  # Subsequent connects are reconnections.
