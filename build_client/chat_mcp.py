"""Chat MCP server — read_unread and send tools for agent↔user communication.

Implements §8 of the Build Agent Protocol. The Chat MCP server runs inside
the agent process, managed by the wrapper. The harness connects to it via
stdio transport and uses the tools to communicate with the user.

Chat MCP calls are NOT reported in the tool.* namespace (§8.4).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

log = logging.getLogger(__name__)

# Type for the send callback.
SendCallback = Callable[[str], Coroutine[Any, Any, None]]


@dataclass
class UnreadMessage:
    """A queued user message waiting to be read by the agent."""

    role: str
    content: str
    timestamp: str


class ChatMCP:
    """Chat MCP tool logic — message queue and tool handlers.

    Framework-agnostic: provides async handler functions that can be wrapped
    for different MCP transports (mcp library stdio, Agent SDK in-process, etc.).

    Usage:
        chat = ChatMCP(on_send=my_send_callback)

        # Queue an incoming user message (from device client WS).
        await chat.queue_message("Fix the auth bug")

        # Agent calls read_unread via MCP.
        result = await chat.handle_read_unread()
        # → {"messages": [{"role": "user", "content": "Fix the auth bug", "timestamp": "..."}]}

        # Agent calls send via MCP.
        result = await chat.handle_send("Done — auth module refactored.")
        # → {"status": "sent"}  (and on_send callback fires chat.response)
    """

    def __init__(self, on_send: SendCallback | None = None) -> None:
        self._unread: list[UnreadMessage] = []
        self._lock = asyncio.Lock()
        self._on_send = on_send
        self._unread_event = asyncio.Event()

    # -----------------------------------------------------------------
    # Message queue (called by wrapper when chat.message arrives)
    # -----------------------------------------------------------------

    async def queue_message(
        self,
        content: str,
        role: str = "user",
        timestamp: str | None = None,
    ) -> None:
        """Queue a user message for the agent to read via read_unread.

        Must be called from the same event loop as the other ChatMCP methods.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        async with self._lock:
            self._unread.append(UnreadMessage(role=role, content=content, timestamp=ts))
            self._unread_event.set()
        log.debug("Queued message for agent (%d unread)", len(self._unread))

    @property
    def unread_count(self) -> int:
        """Number of unread messages in the queue."""
        return len(self._unread)

    @property
    def has_unread(self) -> bool:
        return len(self._unread) > 0

    async def wait_for_unread(self, timeout: float | None = None) -> bool:
        """Wait until there is at least one unread message. Returns True if message arrived."""
        if self._unread:
            return True
        try:
            await asyncio.wait_for(self._unread_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # -----------------------------------------------------------------
    # MCP tool handlers
    # -----------------------------------------------------------------

    async def handle_read_unread(self) -> dict[str, Any]:
        """MCP tool: read_unread — return queued user messages.

        Returns all unread messages and clears the queue. Subsequent calls
        return only new messages (§8.1).
        """
        async with self._lock:
            messages = [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self._unread
            ]
            self._unread.clear()
            self._unread_event.clear()

        if messages:
            log.debug("Agent read %d unread messages", len(messages))

        return {"messages": messages}

    async def handle_send(self, message: str) -> dict[str, Any]:
        """MCP tool: send — send a message to the user.

        The wrapper translates this into a chat.response protocol message (§8.2).
        The on_send callback is responsible for emitting the BAP message.
        """
        if self._on_send:
            await self._on_send(message)
        else:
            log.warning("send called but no on_send callback registered")

        return {"status": "sent"}

    # -----------------------------------------------------------------
    # Notification injection helpers (§8.3)
    # -----------------------------------------------------------------

    def build_unread_notification(self) -> str | None:
        """Build an unread notification string for harness injection.

        Returns None if no unread messages.

        Note: prefer ``drain_unread_notification`` for atomic check-and-build
        to avoid races between ``wait_for_unread`` and ``handle_read_unread``.
        """
        count = len(self._unread)
        if count == 0:
            return None

        if count == 1:
            return (
                "You have 1 unread message from the user. "
                "Use the read_unread tool to read it."
            )

        return (
            f"You have {count} unread messages from the user. "
            "Use the read_unread tool to read them."
        )

    async def drain_unread_notification(self) -> str | None:
        """Atomically check for unread messages and build a notification.

        Returns the notification string if there were unread messages,
        or None if the queue was empty. The queue is NOT drained — the agent
        still reads messages via ``handle_read_unread``. This just ensures the
        notification is consistent with the queue state under the lock.
        """
        async with self._lock:
            count = len(self._unread)
            if count == 0:
                return None
            if count == 1:
                return (
                    "You have 1 unread message from the user. "
                    "Use the read_unread tool to read it."
                )
            return (
                f"You have {count} unread messages from the user. "
                "Use the read_unread tool to read them."
            )

    # -----------------------------------------------------------------
    # MCP server creation (requires `mcp` package)
    # -----------------------------------------------------------------

    def create_stdio_server(self) -> Any:
        """Create a FastMCP server for stdio transport.

        Requires the `mcp` package to be installed.
        The returned server can be run with server.run(transport="stdio").
        """
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError:
            raise ImportError(
                "The 'mcp' package is required for stdio server mode. "
                "Install it with: uv add mcp"
            )

        mcp = FastMCP(name="build-chat")

        # Capture self for closures.
        chat = self

        @mcp.tool(
            name="read_unread",
            description=(
                "Read unread messages from the user. Returns any queued messages "
                "and clears the queue. Call this when notified of unread messages."
            ),
        )
        async def read_unread() -> dict:
            return await chat.handle_read_unread()

        @mcp.tool(
            name="send",
            description=(
                "Send a message to the user. Use this to communicate with the user "
                "instead of outputting text directly. The message will be delivered "
                "to the user's browser."
            ),
        )
        async def send(message: str) -> dict:
            return await chat.handle_send(message)

        return mcp
