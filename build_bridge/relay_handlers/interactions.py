"""Interactions namespace — RELAY_PROTOCOL.md §5.5.

Forwards the user's answer to a prior `interaction.request` from the agent.
"""

from __future__ import annotations

import logging
from typing import Any

from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


async def handle_interaction_response(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Persist + forward an interaction response to the agent."""
    # validator: channel_id + interaction_id required.
    channel_id = payload["channel_id"]
    interaction_id = payload["interaction_id"]
    selected_option = payload.get("selected_option")
    freeform_response = payload.get("freeform_response")
    selected_options = payload.get("selected_options")
    step_answers = payload.get("step_answers")

    agent_server = ctx.agent_server
    if not agent_server:
        return

    # Persist the response.
    agent_server.store.resolve_interaction(
        interaction_id, channel_id, selected_option, freeform_response,
        selected_options=selected_options,
    )

    # Forward to agent.
    sent = await agent_server.send_interaction_response(
        channel_id, interaction_id, selected_option, freeform_response,
        selected_options=selected_options,
        step_answers=step_answers,
    )
    if sent:
        log.info("Forwarded interaction.response to agent on channel %s", channel_id[:8])
    else:
        log.error("Failed to forward interaction.response on channel %s", channel_id[:8])
        await ctx.send_error_message(
            session, ws, channel_id,
            "Failed to send response to agent. The agent may have disconnected.",
        )
