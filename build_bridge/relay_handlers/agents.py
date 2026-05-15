"""Agents namespace — RELAY_PROTOCOL.md §5.4.

Handlers for: list_harnesses, list_workers, start_agent, stop_agent,
cancel, restart_agent.
"""

from __future__ import annotations

import logging
from typing import Any

from build_bridge import relay_protocol as proto
from build_bridge.harness_registry import detect_installed, get_harness, serialize_harnesses
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


async def handle_list_harnesses(
    ctx: HandlerContext,
    session: ActiveSession,
    ws: Any,
) -> None:
    harnesses = detect_installed()
    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.HARNESS_LIST,
            "harnesses": serialize_harnesses(harnesses),
        },
    )


async def handle_list_workers(
    ctx: HandlerContext,
    session: ActiveSession,
    ws: Any,
) -> None:
    spawner = ctx.agent_spawner
    if not spawner:
        await ctx.send_frame(
            session, ws,
            payload={"action": proto.WORKER_LIST, "workers": []},
        )
        return

    workers = spawner.list_workers()
    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.WORKER_LIST,
            "workers": workers,
        },
    )


async def handle_start_agent(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    spawner = ctx.agent_spawner
    if not spawner:
        await ctx.send_frame(
            session, ws,
            payload={"action": proto.ERROR, "error": "agent spawner not available"},
        )
        return

    # validator: channel_id + harness required.
    channel_id = payload["channel_id"]
    harness = payload["harness"]
    model = payload.get("model", "")
    system_prompt = payload.get("system_prompt", "")
    working_directory = payload.get("working_directory", "")

    harness_info = get_harness(harness)
    if not harness_info:
        await ctx.send_frame(
            session, ws,
            payload={"action": proto.ERROR, "error": f"unknown harness: {harness}"},
        )
        return

    if not model:
        model = harness_info.default_model

    try:
        worker = await spawner.spawn(
            channel_id=channel_id,
            harness=harness,
            model=model,
            system_prompt=system_prompt,
            working_directory=working_directory,
        )
        await ctx.send_frame(
            session, ws,
            payload={
                "action": proto.AGENT_STARTED,
                "channel_id": channel_id,
                "agent_id": worker.agent_id,
                "harness": harness,
                "model": model,
                "pid": worker.pid,
            },
        )
    except Exception as exc:
        log.error("Failed to start agent on channel %s: %s", channel_id[:8], exc)
        await ctx.send_frame(
            session, ws,
            payload={"action": proto.ERROR, "error": f"failed to start agent: {exc}"},
        )


async def handle_stop_agent(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Two-phase stop: graceful chat.cancel, then force-kill after 3s timeout."""
    spawner = ctx.agent_spawner
    if not spawner:
        await ctx.send_frame(
            session, ws,
            payload={"action": proto.ERROR, "error": "agent spawner not available"},
        )
        return

    channel_id = payload["channel_id"]  # validator: required
    was_running = spawner.is_running(channel_id)
    killed = False
    log.info("stop_agent: channel=%s was_running=%s", channel_id[:8], was_running)

    agent_server = ctx.agent_server
    if was_running and agent_server:
        sent = await agent_server.send_cancel(channel_id)
        log.info("stop_agent: send_cancel returned %s for %s", sent, channel_id[:8])
        if sent:
            acked = await agent_server.wait_for_cancel_ack(channel_id, timeout=3.0)
            log.info("stop_agent: cancel ack=%s for %s", acked, channel_id[:8])
            if not acked:
                log.info("Agent on %s didn't ack cancel, killing", channel_id[:8])
                await spawner.stop(channel_id, resumable=True)
                killed = True
        else:
            log.warning("stop_agent: could not send cancel to %s", channel_id[:8])

    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.AGENT_STOPPED,
            "channel_id": channel_id,
            "was_running": was_running,
            "killed": killed,
        },
    )


async def handle_cancel(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Send chat.cancel to the agent without waiting for ack."""
    channel_id = payload["channel_id"]  # validator: required
    sent = await ctx.agent_server.send_cancel(channel_id)
    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.CANCEL_ACK,
            "channel_id": channel_id,
            "sent": sent,
        },
    )


async def handle_restart_agent(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    spawner = ctx.agent_spawner
    if not spawner:
        await ctx.send_frame(
            session, ws,
            payload={"action": proto.ERROR, "error": "agent spawner not available"},
        )
        return

    channel_id = payload["channel_id"]  # validator: required
    try:
        worker = await spawner.restart(channel_id)
        if not worker:
            await ctx.send_frame(
                session, ws,
                payload={
                    "action": proto.ERROR,
                    "error": f"no agent config found for channel {channel_id[:8]}",
                },
            )
            return

        await ctx.send_frame(
            session, ws,
            payload={
                "action": proto.AGENT_RESTARTED,
                "channel_id": channel_id,
                "agent_id": worker.agent_id,
                "pid": worker.pid,
            },
        )
    except Exception as exc:
        log.error("Failed to restart agent on channel %s: %s", channel_id[:8], exc)
        await ctx.send_frame(
            session, ws,
            payload={"action": proto.ERROR, "error": f"failed to restart agent: {exc}"},
        )
