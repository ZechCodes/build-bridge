"""Channels namespace — RELAY_PROTOCOL.md §5.1.

Channel CRUD: list, create, rename, update, delete. The "channel" concept
is shared between the E2EE MessageStore (which holds the chat history)
and the AgentStore (which holds the agent configuration); these handlers
keep both stores in sync.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from build_bridge import relay_protocol as proto
from build_bridge.harness_registry import get_harness
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


async def handle_list_channels(
    ctx: HandlerContext,
    session: ActiveSession,
    ws: Any,
) -> None:
    """Send the list of channels to the browser.

    Followed by zero or more `complication:update` frames seeded from the
    current complications registry, so the browser has state immediately.
    """
    channels = ctx.e2ee_store.list_channels()
    channel_list: list[dict[str, Any]] = []
    agent_server = ctx.agent_server
    agent_spawner = ctx.agent_spawner

    for c in channels:
        entry: dict[str, Any] = {"id": c.id, "name": c.name, "created_at": c.created_at}
        if agent_server:
            agent_ch = agent_server.store.get_channel(c.id)
            if agent_ch:
                entry["harness"] = agent_ch.harness
                entry["model"] = agent_ch.model
                info = get_harness(agent_ch.harness)
                entry["agent_name"] = info.name if info else "Device"
                entry["working_directory"] = agent_ch.working_directory
                entry["effort"] = agent_ch.effort
                entry["plan_mode"] = agent_ch.plan_mode
                entry["auto_approve_tools"] = agent_ch.auto_approve_tools
                entry["last_seen_at"] = agent_ch.last_seen_at
                # Authoritative running flag: worker subprocess is alive AND
                # the agent is mid-turn (status flips to 'idle' on
                # activity.end{complete|waiting}, 'error' on
                # activity.end{error}). The browser reconciles
                # presenceStore from this on every channel_list response —
                # covers SSE blips and tab-focus resyncs.
                entry["is_running"] = bool(
                    agent_spawner
                    and agent_spawner.is_running(c.id)
                    and agent_ch.status == "active"
                )
        channel_list.append(entry)

    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.CHANNEL_LIST,
            "channels": channel_list,
            "agent_cwd": os.getcwd(),
        },
    )

    # Send current complications so the browser has state immediately.
    if agent_server and agent_server._complications:
        try:
            comps = await agent_server._complications.get_current_complications(
                agent_store=agent_server.store,
            )
            for comp in comps:
                await ctx.send_frame(session, ws, payload=comp)
        except Exception as exc:
            log.debug("Failed to send initial complications: %s", exc)


async def handle_create_channel(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Create a new channel and confirm to the browser.

    If a harness was provided, auto-spawn an agent on the new channel.
    """
    name = payload["name"].strip()  # validator: required, non-empty
    channel_id = str(uuid.uuid4())
    channel = ctx.e2ee_store.create_channel(channel_id, name)
    log.info("Created channel: %s (%s)", name, channel_id[:8])

    # Ensure an agent_channels stub exists from the moment the channel is
    # born, so later settings edits always have a row to UPDATE.
    agent_server = ctx.agent_server
    if agent_server:
        agent_server.store.ensure_channel_row(channel_id)

    response: dict[str, Any] = {
        "action": proto.CHANNEL_CREATED,
        "channel": {
            "id": channel.id,
            "name": channel.name,
            "created_at": channel.created_at,
        },
    }

    # Auto-spawn agent if harness config provided.
    harness = payload.get("harness", "")
    agent_spawner = ctx.agent_spawner
    if harness and agent_spawner:
        model = payload.get("model", "")
        effort = payload.get("effort", "")
        system_prompt = payload.get("system_prompt", "")
        working_directory = payload.get("working_directory", "")
        auto_approve_tools = bool(payload.get("auto_approve_tools", False))

        log.info(
            "create_channel payload: harness=%r, model=%r, effort=%r, "
            "system_prompt=%r, working_directory=%r, auto_approve_tools=%r, all_keys=%s",
            harness, model, effort, system_prompt, working_directory,
            auto_approve_tools, list(payload.keys()),
        )

        harness_info = get_harness(harness)
        if harness_info:
            if not model:
                model = harness_info.default_model
            try:
                worker = await agent_spawner.spawn(
                    channel_id=channel_id,
                    harness=harness,
                    model=model,
                    effort=effort,
                    system_prompt=system_prompt,
                    working_directory=working_directory,
                    auto_approve_tools=auto_approve_tools,
                )
                response["agent"] = {
                    "agent_id": worker.agent_id,
                    "harness": harness,
                    "model": model,
                    "pid": worker.pid,
                }
            except Exception as exc:
                log.error("Auto-spawn failed for channel %s: %s", channel_id[:8], exc)
                response["agent_error"] = str(exc)

    await ctx.send_frame(session, ws, payload=response)


async def handle_rename_channel(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    channel_id = payload["channel_id"]  # validator: required
    name = payload["name"].strip()       # validator: required, non-empty
    ctx.e2ee_store.rename_channel(channel_id, name)
    log.info("Renamed channel %s to %r", channel_id[:8], name)
    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.CHANNEL_RENAMED,
            "channel_id": channel_id,
            "name": name,
        },
    )


async def handle_update_channel(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Update channel settings. Optionally restarts agent for spawn-time settings."""
    channel_id = payload["channel_id"]  # validator: required
    agent_server = ctx.agent_server

    # Guarantee an agent_channels row exists before any UPDATE runs.
    # Channels created before the stub-on-create change (or created via
    # paths that skipped auto-spawn) would otherwise silently no-op.
    if agent_server:
        agent_server.store.ensure_channel_row(channel_id)

    working_directory = payload.get("working_directory")
    if working_directory is not None and agent_server:
        agent_server.store.update_working_directory(channel_id, working_directory)
        log.info("Updated working_directory for channel %s", channel_id[:8])

    model = payload.get("model")
    if model and agent_server:
        agent_server.store.update_model(channel_id, model)
        log.info("Updated model for channel %s to %s", channel_id[:8], model)

    effort = payload.get("effort")
    if effort is not None and agent_server:
        agent_server.store.update_effort(channel_id, effort)
        log.info("Updated effort for channel %s to %s", channel_id[:8], effort)

    harness_changed = False
    harness = payload.get("harness")
    if harness and agent_server:
        current = agent_server.store.get_channel(channel_id)
        if current is not None and current.harness != harness:
            agent_server.store.update_harness(channel_id, harness)
            harness_changed = True
            log.info("Updated harness for channel %s to %s", channel_id[:8], harness)

    auto_approve_changed = False
    auto_approve_tools = payload.get("auto_approve_tools")
    if auto_approve_tools is not None and agent_server:
        flag = bool(auto_approve_tools)
        current = agent_server.store.get_channel(channel_id)
        if current is not None and current.auto_approve_tools != flag:
            agent_server.store.update_auto_approve_tools(channel_id, flag)
            auto_approve_changed = True
            log.info(
                "Updated auto_approve_tools for channel %s to %s",
                channel_id[:8], flag,
            )

    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.CHANNEL_UPDATED,
            "channel_id": channel_id,
            "working_directory": working_directory,
            "model": model,
            "effort": effort,
            "harness": harness,
            "auto_approve_tools": auto_approve_tools,
        },
    )

    # Emit setting-change markers so the browser shows what changed.
    changes: list[str] = []
    if harness_changed:
        changes.append(f"Harness → {harness}")
    if model:
        changes.append(f"Model → {model}")
    if effort is not None:
        changes.append(f"Effort → {effort or 'default'}")
    if auto_approve_changed:
        changes.append(
            "Auto-approve tools → on" if auto_approve_tools else "Auto-approve tools → off"
        )
    if changes:
        await ctx.send_system_message(session, ws, channel_id, " · ".join(changes))

    # Restart the agent whenever a setting that only takes effect at spawn
    # time is changed. harness controls which binary to launch;
    # auto_approve_tools is read once at spawn for approvalPolicy (Codex) /
    # permission_mode (Claude).
    agent_spawner = ctx.agent_spawner
    if (harness_changed or auto_approve_changed) and agent_spawner:
        try:
            await agent_spawner.restart(channel_id)
        except Exception:
            log.exception("Failed to restart agent after channel update")


async def handle_delete_channel(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    channel_id = payload["channel_id"]  # validator: required
    agent_spawner = ctx.agent_spawner
    if agent_spawner:
        try:
            await agent_spawner.stop(channel_id)
        except Exception as exc:
            log.debug("Error stopping agent for deleted channel %s: %s", channel_id[:8], exc)

    ctx.e2ee_store.delete_channel(channel_id)

    agent_server = ctx.agent_server
    if agent_server and agent_server.store:
        try:
            agent_server.store.delete_channel(channel_id)
        except Exception as exc:
            log.debug("Error cleaning agent store for channel %s: %s", channel_id[:8], exc)
    log.info("Deleted channel %s", channel_id[:8])

    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.CHANNEL_DELETED,
            "channel_id": channel_id,
        },
    )
