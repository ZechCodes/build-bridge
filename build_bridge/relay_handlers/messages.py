"""Messages namespace — RELAY_PROTOCOL.md §5.2.

Handlers for: get_messages, message (chat from user), retry_message,
mark_read, mark_seen. The chat-message flow also covers attachment
enrichment with upload final paths, auto-restart of a downed agent, and
the per-message plan_mode / model / effort overrides.

`handle_chat_message` and `handle_retry_message` take the full decrypted
`frame` (rather than just `payload`) because they read the outer
`message_id` for delivery correlation. The dispatcher in
`E2EESession._handle_data_frame` routes those two actions through
`E2EEHandler._handle_chat_message` / `_retry_message` to preserve the
frame-shape contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from build_bridge import relay_protocol as proto
from build_bridge.relay_handlers import uploads
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


async def handle_get_messages(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Send merged E2EE + AgentStore chat history, chronologically sorted."""
    channel_id = payload.get("channel_id", "")
    limit = payload.get("limit", 50)
    before = payload.get("before")

    # E2EE messages (user messages, device error messages).
    e2ee_msgs = ctx.e2ee_store.get_messages(channel_id, limit=limit, before=before)

    # Agent chat messages (user + assistant from AgentStore).
    agent_msgs: list[dict[str, Any]] = []
    agent_server = ctx.agent_server
    if agent_server:
        try:
            agent_name = ctx.get_agent_name(channel_id)
            chat_history = agent_server.store.get_chat_history(channel_id)
            # Avoid duplicates — user messages appear in both stores.
            e2ee_ids = {m.id for m in e2ee_msgs}
            for cm in chat_history:
                if cm.id in e2ee_ids:
                    continue
                # Parse ISO created_at to float for consistent sorting.
                try:
                    dt = datetime.fromisoformat(cm.created_at)
                    ts = dt.timestamp()
                except (ValueError, TypeError):
                    ts = 0.0
                msg_dict = {
                    "id": cm.id,
                    "channel_id": cm.channel_id,
                    "sender": agent_name if cm.role == "assistant" else "client",
                    "content": cm.content,
                    "created_at": ts,
                    "delivered_at": ts,
                    "read_at": ts,
                }
                if cm.metadata:
                    msg_dict["metadata"] = cm.metadata
                if cm.suggested_actions:
                    msg_dict["suggested_actions"] = json.loads(cm.suggested_actions)
                agent_msgs.append(msg_dict)
        except Exception as exc:
            log.error("Failed to merge agent chat messages: %s", exc)

    # Merge and sort chronologically.
    combined: list[dict[str, Any]] = []
    for m in e2ee_msgs:
        msg_dict: dict[str, Any] = {
            "id": m.id,
            "channel_id": m.channel_id,
            "sender": m.sender,
            "content": m.content,
            "created_at": m.created_at,
            "delivered_at": m.delivered_at,
            "read_at": m.read_at,
        }
        if m.attachments:
            msg_dict["attachments"] = m.attachments
        combined.append(msg_dict)
    combined += agent_msgs
    combined.sort(key=lambda mm: mm.get("created_at") or 0)

    if len(combined) > limit:
        combined = combined[-limit:]

    log.info(
        "Sending %d messages for channel %s (e2ee=%d, agent=%d)",
        len(combined), channel_id[:8], len(e2ee_msgs), len(agent_msgs),
    )

    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.MESSAGES,
            "channel_id": channel_id,
            "messages": combined,
        },
    )


async def handle_chat_message(
    ctx: HandlerContext,
    session: ActiveSession,
    frame: dict[str, Any],
    ws: Any,
) -> None:
    """Handle an incoming chat message — store, broadcast, forward to agent."""
    payload = frame.get("payload") or {}
    message_id = frame.get("message_id")
    # validator: channel_id required; content or attachments required.
    channel_id = payload["channel_id"]
    content = payload.get("content", "")
    plan_mode = payload.get("plan_mode")
    model_override = payload.get("model")
    effort_override = payload.get("effort")
    attachments = payload.get("attachments")

    # Store the incoming message from the client.
    stored = ctx.e2ee_store.store_message(
        message_id=message_id,
        channel_id=channel_id,
        session_id=session.session_id,
        sender="client",
        content=content,
        attachments=attachments,
    )

    # Mark delivered the moment we have it on-device; the dual delivery
    # contract (delivered now, delivery_failed later if agent forward
    # fails) is a known wire quirk — see RELAY_PROTOCOL.md §12.
    ctx.e2ee_store.mark_delivered(message_id)
    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.DELIVERED,
            "message_id": message_id,
            "channel_id": channel_id,
        },
    )

    # Broadcast to all sessions so other browsers see the message.
    # The originating browser deduplicates by message ID.
    await ctx.broadcast(channel_id, {
        "action": proto.MESSAGE,
        "message": {
            "id": message_id,
            "channel_id": channel_id,
            "sender": "client",
            "content": content,
            "created_at": stored.created_at,
            "attachments": attachments,
        },
    })

    # NOTE: We do NOT mark as read here. The message is marked read when
    # the agent calls read_unread via the Chat MCP tool.

    # Try to forward to a connected agent.
    agent_server = ctx.agent_server
    if not agent_server:
        log.error(
            "No agent server registered — cannot forward message on channel %s",
            channel_id[:8],
        )
        await _send_delivery_failed(ctx, session, ws, channel_id, message_id)
        return

    if not agent_server.is_channel_active(channel_id):
        # Agent is down — auto-restart it and notify the user.
        log.warning("No active agent on channel %s, attempting auto-restart", channel_id[:8])
        await ctx.send_system_message(
            session, ws, channel_id,
            "Agent is restarting, please standby...",
        )
        restarted = False
        agent_spawner = ctx.agent_spawner
        if agent_spawner:
            try:
                worker = await agent_spawner.restart(channel_id)
                if worker:
                    log.info(
                        "Auto-restarted agent on channel %s (pid=%s)",
                        channel_id[:8], worker.pid,
                    )
                    restarted = True
            except Exception as exc:
                log.error("Failed to auto-restart agent on channel %s: %s", channel_id[:8], exc)
        if not restarted:
            await _send_delivery_failed(ctx, session, ws, channel_id, message_id)
            return
        # Wait briefly for the agent to connect, then retry forwarding.
        for _ in range(10):
            await asyncio.sleep(0.5)
            if agent_server.is_channel_active(channel_id):
                break
        else:
            await _send_delivery_failed(ctx, session, ws, channel_id, message_id)
            return

    # Enrich attachment metadata with local file paths so agents can read files.
    enriched_attachments = None
    if attachments:
        enriched_attachments = []
        for att in attachments:
            file_path = (
                uploads.upload_final_dir(ctx.facade, channel_id, att["file_id"])
                / uploads.sanitize_filename(att.get("filename", "upload"))
            )
            enriched_attachments.append({**att, "path": str(file_path)})

    # Prepend plan mode instruction if the browser sent plan_mode flag.
    forwarded_content = content
    if plan_mode:
        forwarded_content = (
            "[System: The user has planning mode active. Plan before executing. "
            "Use EnterPlanMode if not already in plan mode.]\n\n" + content
        )
        agent_server.store.update_plan_mode(channel_id, True)

    # Persist model/effort changes to the store if provided.
    if model_override:
        agent_server.store.update_model(channel_id, model_override)
    if effort_override:
        agent_server.store.update_effort(channel_id, effort_override)

    sent = await agent_server.send_chat_message(
        channel_id, forwarded_content, msg_id=message_id,
        attachments=enriched_attachments,
        model=model_override,
        effort=effort_override,
        plan_mode=plan_mode,
    )
    if sent:
        log.info("Forwarded chat message to agent on channel %s", channel_id[:8])
        # NOTE: Do NOT mark as read here. The message is only "read" when
        # the agent actually processes it via read_unread.
    else:
        log.error(
            "Failed to send chat message to agent on channel %s "
            "(agent lookup succeeded but send failed)",
            channel_id[:8],
        )
        await _send_delivery_failed(ctx, session, ws, channel_id, message_id)


async def handle_retry_message(
    ctx: HandlerContext,
    session: ActiveSession,
    frame: dict[str, Any],
    ws: Any,
) -> None:
    """Re-attempt forwarding a previously stored message to the agent."""
    payload = frame.get("payload") or {}
    # validator: channel_id + message_id required.
    channel_id = payload["channel_id"]
    message_id = payload["message_id"]
    msg = ctx.e2ee_store.get_message(message_id)
    if not msg:
        log.warning("Retry: message %s not found in store", message_id)
        await _send_delivery_failed(ctx, session, ws, channel_id, message_id)
        return

    # Re-run the delivery flow by constructing a synthetic frame.
    retry_frame = {
        "message_id": message_id,
        "payload": {
            "action": proto.MESSAGE,
            "channel_id": channel_id,
            "content": msg.content,
        },
    }
    await handle_chat_message(ctx, session, retry_frame, ws)


async def _send_delivery_failed(
    ctx: HandlerContext,
    session: ActiveSession,
    ws: Any,
    channel_id: str,
    message_id: str | None,
) -> None:
    """Notify the browser that a message failed to reach the agent."""
    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.DELIVERY_FAILED,
            "message_id": message_id,
            "channel_id": channel_id,
        },
    )


async def handle_mark_read(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Mark messages as read; ack only when message_ids is non-empty."""
    message_ids = payload.get("message_ids", [])
    channel_id = payload.get("channel_id", "")
    for mid in message_ids:
        ctx.e2ee_store.mark_read(mid)
    if message_ids:
        await ctx.send_frame(
            session, ws,
            payload={
                "action": proto.READ,
                "message_ids": message_ids,
                "channel_id": channel_id,
            },
        )


async def handle_mark_seen(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Mark a channel as seen by the user. Fire-and-forget; no response."""
    channel_id = payload["channel_id"]  # validator: required
    agent_server = ctx.agent_server
    if not agent_server:
        return
    agent_server.store.mark_channel_seen(channel_id)
