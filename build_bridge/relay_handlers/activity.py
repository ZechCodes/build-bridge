"""Activity namespace — RELAY_PROTOCOL.md §5.3.

Reads the agent activity log (tool uses, results, reasoning text) for a
channel and returns it as an `activity_history` frame, with aggressive
trimming to stay under the relay's 256 KB encrypted-envelope limit.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from build_bridge import relay_protocol as proto
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


# Max activity entries to send in a single activity_history response.
# The relay enforces a 256 KB envelope limit; large histories (hundreds of
# tool uses with big inputs) easily exceed that. We keep the last N
# *tool_use* entries plus their matching tool_result entries so the console
# shows recent activity without blowing the size budget.
# After encryption + base64 overhead (~33%), 150 KB payload ≈ 200 KB envelope.
_MAX_ACTIVITY_TOOL_USES = 50
_MAX_RESULT_CONTENT_LEN = 1000


def _trim_activity_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last N tool_use entries and their matching tool_result entries."""
    tool_use_indices: list[int] = []
    for i, e in enumerate(entries):
        if e["type"] == "tool_use":
            tool_use_indices.append(i)

    if len(tool_use_indices) <= _MAX_ACTIVITY_TOOL_USES:
        return entries

    cutoff_idx = tool_use_indices[-_MAX_ACTIVITY_TOOL_USES]
    kept = entries[cutoff_idx:]

    # Also keep any tool_result entries before the cutoff that reference
    # a kept tool_use (shouldn't normally happen, but be safe).
    kept_ids = {
        e["data"].get("id") for e in kept if e["type"] == "tool_use" and "data" in e
    }
    for e in entries[:cutoff_idx]:
        if (
            e["type"] == "tool_result"
            and e.get("data", {}).get("tool_use_id") in kept_ids
        ):
            kept.insert(0, e)

    return kept


async def handle_get_activity(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Return tool_use / tool_result / text entries for a channel.

    Returns an `activity_history` frame trimmed to the last
    `_MAX_ACTIVITY_TOOL_USES` (50) tool uses plus their matching results.
    Per-entry content is also truncated (200 chars for tool_use inputs,
    1000 chars for tool_result content and reasoning text).
    """
    channel_id = payload.get("channel_id", "")

    entries: list[dict[str, Any]] = []
    agent_server = ctx.agent_server
    if agent_server:
        try:
            activity = agent_server.store.get_activity_history(channel_id)
            for entry in activity:
                if entry.type in ("tool_use", "tool_result", "text"):
                    try:
                        data = json.loads(entry.data)
                    except (ValueError, TypeError):
                        data = {}
                    # Trim large tool inputs to keep envelope size down.
                    if entry.type == "tool_use" and isinstance(data.get("input"), dict):
                        data["input"] = {
                            k: (v[:200] + "…" if isinstance(v, str) and len(v) > 200 else v)
                            for k, v in data["input"].items()
                        }
                    # Trim large tool result content for history view.
                    if entry.type == "tool_result":
                        c = data.get("content", "")
                        if isinstance(c, str) and len(c) > _MAX_RESULT_CONTENT_LEN:
                            data["content"] = c[:_MAX_RESULT_CONTENT_LEN] + "…"
                    # Trim reasoning text for history view.
                    if entry.type == "text":
                        c = data.get("content", "")
                        if isinstance(c, str) and len(c) > _MAX_RESULT_CONTENT_LEN:
                            data["content"] = c[:_MAX_RESULT_CONTENT_LEN] + "…"
                    entries.append({
                        "type": entry.type,
                        "data": data,
                        "created_at": entry.created_at,
                    })
        except Exception as exc:
            log.error("Failed to fetch activity history: %s", exc)

    # Trim to most recent N tool_use entries (plus their results).
    if entries:
        entries = _trim_activity_entries(entries)

    # Safety: halve until estimated encrypted size is under 200 KB.
    payload_json = json.dumps(entries)
    estimated_encrypted = len(payload_json) * 1.4
    while estimated_encrypted > 200_000 and len(entries) > 10:
        entries = entries[len(entries) // 2:]
        payload_json = json.dumps(entries)
        estimated_encrypted = len(payload_json) * 1.4
        log.warning(
            "Activity payload too large, reduced to %d entries (%.0f KB est.)",
            len(entries), estimated_encrypted / 1024,
        )

    log.info(
        "Sending %d activity entries for channel %s (%.1f KB)",
        len(entries), channel_id[:8], len(payload_json) / 1024,
    )

    await ctx.send_frame(
        session, ws,
        payload={
            "action": proto.ACTIVITY_HISTORY,
            "channel_id": channel_id,
            "entries": entries,
        },
    )
