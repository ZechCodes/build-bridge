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
from pathlib import Path
from typing import Annotated, Any, Callable, Coroutine

log = logging.getLogger(__name__)

# Type for the send callback.
SendCallback = Callable[[str, list[str] | None], Coroutine[Any, Any, None]]

# Supported image MIME types for inline content blocks.
_IMAGE_MIME_TYPES = frozenset({
    "image/jpeg", "image/png", "image/gif", "image/webp",
})


@dataclass
class UnreadMessage:
    """A queued user message waiting to be read by the agent."""

    role: str
    content: str
    timestamp: str
    attachments: list[dict[str, Any]] | None = None
    msg_id: str | None = None  # E2EE message ID for read tracking


# Callback fired when messages are read by the agent.
ReadCallback = Callable[[list[str]], Coroutine[Any, Any, None]]


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

    def __init__(
        self,
        on_send: SendCallback | None = None,
        on_read: ReadCallback | None = None,
        harness: str = "claude-code",
    ) -> None:
        self._unread: list[UnreadMessage] = []
        self._lock = asyncio.Lock()
        self._on_send = on_send
        self._on_read = on_read
        self._unread_event = asyncio.Event()
        self._harness = harness

    # -----------------------------------------------------------------
    # Message queue (called by wrapper when chat.message arrives)
    # -----------------------------------------------------------------

    async def queue_message(
        self,
        content: str,
        role: str = "user",
        timestamp: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        msg_id: str | None = None,
    ) -> None:
        """Queue a user message for the agent to read via read_unread.

        Must be called from the same event loop as the other ChatMCP methods.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        async with self._lock:
            self._unread.append(UnreadMessage(
                role=role, content=content, timestamp=ts,
                attachments=attachments, msg_id=msg_id,
            ))
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
        return only new messages (§8.1). Fires on_read callback with message
        IDs so the browser can update read status.
        """
        async with self._lock:
            messages = [
                {
                    "role": m.role,
                    "content": self._format_message_content(m),
                    "timestamp": m.timestamp,
                }
                for m in self._unread
            ]
            read_ids = [m.msg_id for m in self._unread if m.msg_id]
            self._unread.clear()
            self._unread_event.clear()

        if messages:
            log.debug("Agent read %d unread messages", len(messages))

        # Notify that messages were read.
        if read_ids and self._on_read:
            try:
                await self._on_read(read_ids)
            except Exception as exc:
                log.warning("on_read callback failed: %s", exc)

        return {"messages": messages}

    async def handle_send(
        self,
        message: str,
        suggested_actions: list[str] | None = None,
    ) -> dict[str, Any]:
        """MCP tool: send — send a message to the user.

        The wrapper translates this into a chat.response protocol message (§8.2).
        The on_send callback is responsible for emitting the BAP message.
        """
        if self._on_send:
            await self._on_send(message, suggested_actions)
        else:
            log.warning("send called but no on_send callback registered")

        return {"status": "sent"}

    # -----------------------------------------------------------------
    # Multimodal content formatting
    # -----------------------------------------------------------------

    def _format_message_content(
        self, msg: UnreadMessage,
    ) -> str | list[dict[str, Any]]:
        """Format a message's content with attachment references.

        If the message has no attachments, returns the plain text content.
        Otherwise returns a list of content blocks with file path references
        so the agent can read attachments directly (avoiding base64 inlining
        which can exceed SDK buffer limits).
        """
        if not msg.attachments:
            return msg.content

        blocks: list[dict[str, Any]] = []

        # Add text content first.
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})

        for att in msg.attachments:
            file_path = att.get("path")
            mime_type = att.get("mime_type", "application/octet-stream")
            filename = att.get("filename", "unknown")

            if not file_path:
                blocks.append({
                    "type": "text",
                    "text": f"[Attachment missing path: {filename}]",
                })
                continue

            path = Path(file_path)
            if not path.exists():
                blocks.append({
                    "type": "text",
                    "text": f"[Attachment not found: {filename}]",
                })
                continue

            if mime_type in _IMAGE_MIME_TYPES:
                blocks.append({
                    "type": "text",
                    "text": f"[Image: {filename} — read it with: Read {file_path}]",
                })
            else:
                blocks.append({
                    "type": "text",
                    "text": f"[Attached file: {filename} ({mime_type}) — path: {file_path}]",
                })

        # If only text blocks resulted, return plain string.
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            return blocks[0]["text"]

        return blocks

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
                "to the user's browser.\n\n"
                "You can embed file snippets and diffs in messages:\n"
                "- [[file]](path/to/file) — embed entire file with syntax highlighting\n"
                "- [[file 8:18]](path/to/file) — embed lines 8-18\n"
                "- [[diff]](path/to/file) — embed git diff for file\n"
                "- [[diff]](file1|file2) — diff two files\n"
                "Paths are resolved relative to the channel working directory "
                "(not your shell cwd). Use absolute paths if you've cd'd elsewhere. "
                "Embeds render with syntax highlighting in the browser."
            ),
        )
        async def send(
            message: str,
            suggested_actions: Annotated[
                list[str] | None,
                "Optional 2-3 short action labels shown as clickable buttons below "
                "the message (e.g. ['yes', 'no']). Clicking one sends it as a user message.",
            ] = None,
        ) -> dict:
            return await chat.handle_send(message, suggested_actions)

        return mcp
