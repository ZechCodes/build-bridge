"""Relay protocol action handlers.

Per-namespace handler modules that implement the browser → device wire
defined in build-bridge/RELAY_PROTOCOL.md §5.

Handler shape:

    async def handle_<action>(
        ctx: HandlerContext,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None

A few handlers (chat-message, retry-message) take the full decrypted
frame rather than just the payload because they read the outer
`message_id` for correlation; those are wired through
`E2EEHandler._handle_chat_message` / `_retry_message` directly rather
than via the dispatch table.

The dispatch table itself is built in `build_bridge.e2ee.E2EEHandler._build_dispatch_table`.
"""

from build_bridge.relay_handlers.context import HandlerContext

__all__ = ["HandlerContext"]
