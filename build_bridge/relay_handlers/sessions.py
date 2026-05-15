"""Sessions namespace — RELAY_PROTOCOL.md §5.6.

Session-history operations:

- `reset_session`: discard history at the current channel and respawn the
  agent so it picks up a clean session.
- `compact_session`: ask the agent for a summary, persist it as the first
  message in a new session, restart the agent against the compacted state.
"""

from __future__ import annotations

import logging
from typing import Any

from build_bridge import relay_protocol as proto
from build_bridge.agent_protocol import generate_id
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


async def handle_reset_session(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    channel_id = payload["channel_id"]  # validator: required
    agent_server = ctx.agent_server
    agent_spawner = ctx.agent_spawner
    if not agent_server or not agent_spawner:
        await ctx.send_frame(
            session, ws,
            payload={"action": proto.ERROR, "error": "agent not available"},
        )
        return

    timestamp = agent_server.store.reset_session(channel_id)
    await agent_spawner.restart(channel_id)
    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.SESSION_RESET,
            "channel_id": channel_id,
            "session_start_at": timestamp,
        },
    )


async def handle_compact_session(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Summarize → reset → restart, carrying the summary into the new session."""
    channel_id = payload["channel_id"]  # validator: required
    agent_server = ctx.agent_server
    agent_spawner = ctx.agent_spawner
    if not agent_server or not agent_spawner:
        await ctx.send_frame(
            session, ws,
            payload={"action": proto.ERROR, "error": "agent not available"},
        )
        return

    # Phase 1: notify browser that compaction has started.
    await ctx.send_frame(
        session, ws,
        payload={"action": proto.COMPACT_STARTED, "channel_id": channel_id},
    )

    # Ask the agent to summarize. This awaits until the agent responds.
    summary = await agent_server.request_summary(channel_id)

    if not summary:
        # No agent connected or already compacting — fall back to plain reset.
        timestamp = agent_server.store.reset_session(channel_id)
        await agent_spawner.restart(channel_id)
        await ctx.send_frame(
            session, ws,
            payload={
                "action": proto.SESSION_RESET,
                "channel_id": channel_id,
                "session_start_at": timestamp,
            },
        )
        return

    # Reset session boundary.
    timestamp = agent_server.store.reset_session(channel_id)

    # Store summary as the first assistant message in the new session.
    agent_server.store.store_chat_message(
        generate_id(), channel_id, "assistant", summary,
    )

    # Restart agent — it will pick up the summary in its history.
    await agent_spawner.restart(channel_id)

    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.SESSION_RESET,
            "channel_id": channel_id,
            "session_start_at": timestamp,
        },
    )
