"""Complications namespace — RELAY_PROTOCOL.md §5.7.

Browser-facing handlers for complications:

- `get_complications`: snapshot the current complications for a channel.
- `complication:action`: trigger a user-initiated action (e.g. `git push`)
  on a specific complication.

The complications registry itself (git-status evaluation, debounced
broadcasts to all sessions) lives in `build_bridge/complications.py`.
"""

from __future__ import annotations

import logging
from typing import Any

from build_bridge import relay_protocol as proto
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


async def handle_get_complications(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Send the current complications for a channel."""
    channel_id = payload.get("channel_id", "")
    complications: list[dict[str, Any]] = []

    agent_server = ctx.agent_server
    if agent_server and agent_server._complications:
        try:
            all_comps = await agent_server._complications.get_current_complications(
                agent_store=agent_server.store,
            )
            complications = [c for c in all_comps if c.get("channel_id") == channel_id]
        except Exception as exc:
            log.debug("Failed to get complications for %s: %s", channel_id[:8], exc)

    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.COMPLICATIONS,
            "channel_id": channel_id,
            "complications": complications,
        },
    )


async def handle_complication_action(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Handle a user action on a complication (e.g. `git push`).

    Fire-and-forget at the relay layer; the result manifests as a
    subsequent `complication:update` broadcast when git state changes.
    """
    agent_server = ctx.agent_server
    if not agent_server or not agent_server._complications:
        log.warning("Complication action received but no complications registry")
        return

    # validator: channel_id + complication_id + option_id required.
    channel_id = payload["channel_id"]
    complication_id = payload["complication_id"]
    option_id = payload["option_id"]
    await agent_server._complications.handle_action(
        channel_id, complication_id, option_id,
    )
