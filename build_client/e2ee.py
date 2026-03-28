"""Device-side E2EE handler — session management, decryption, and agent relay."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from build_secure_transport import (
    build_session_accept,
    decrypt_envelope,
    encrypt_frame,
    open_session_init,
)

from build_client.config import DeviceConfig
from build_client.harness_registry import detect_installed, get_harness, serialize_harnesses
from build_client.storage import MessageStore

if TYPE_CHECKING:
    from build_client.agent_server import AgentServer
    from build_client.agent_spawner import AgentSpawner

log = logging.getLogger(__name__)


@dataclass
class ActiveSession:
    """An active E2EE session with a browser client."""
    session_id: str
    session_key_b64: str


class E2EEHandler:
    """Handles E2EE protocol on the device side.

    Manages sessions, decrypts incoming envelopes, stores messages,
    and relays chat messages to/from connected agents.
    """

    def __init__(self, config: DeviceConfig, store: MessageStore) -> None:
        self.config = config
        self.store = store
        self._sessions: dict[str, ActiveSession] = {}
        self._relay_ws: Any = None  # Current relay WS connection
        self._agent_server: AgentServer | None = None
        self._agent_spawner: AgentSpawner | None = None

    def _get_agent_name(self, channel_id: str) -> str:
        """Get the display name for the agent on a channel (e.g., 'Claude Code').

        Falls back to 'device' if the harness cannot be determined.
        """
        if self._agent_server:
            channel = self._agent_server.store.get_channel(channel_id)
            if channel:
                harness_info = get_harness(channel.harness)
                if harness_info:
                    return harness_info.name
        return "device"

    def set_agent_server(self, server: AgentServer) -> None:
        """Register the agent server for forwarding chat messages."""
        self._agent_server = server

    def set_agent_spawner(self, spawner: AgentSpawner) -> None:
        """Register the agent spawner for starting/stopping agents."""
        self._agent_spawner = spawner

    async def broadcast_to_sessions(
        self,
        channel_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Broadcast an E2EE-encrypted payload to ALL active browser sessions.

        Called by AgentServer when agent events need to reach the browser.
        The payload includes channel_id so the browser can filter.
        """
        if not self._relay_ws:
            log.warning("No relay WS connection — skipping broadcast for %s", payload.get("event_type", "?"))
            return

        sessions = list(self._sessions.values())
        if not sessions:
            log.warning("No active sessions — skipping broadcast for %s", payload.get("event_type", "?"))
            return

        log.info(
            "Broadcasting %s to %d session(s) on channel %s",
            payload.get("event_type", "?"), len(sessions), channel_id[:8],
        )

        for session in sessions:
            try:
                await self._send_frame(session, self._relay_ws, payload=payload)
            except Exception as exc:
                log.error(
                    "Failed to broadcast to session %s: %s",
                    session.session_id[:12], exc,
                )

    async def handle_message(self, msg: dict[str, Any], ws: Any) -> None:
        """Dispatch incoming WS messages from the server."""
        # Track the relay WS for broadcasting.
        self._relay_ws = ws

        msg_type = msg.get("type")

        if msg_type == "session_init":
            await self._handle_session_init(msg, ws)
        elif msg_type == "e2ee_envelope":
            await self._handle_envelope(msg, ws)
        else:
            log.warning("E2EE handler got unknown type: %s", msg_type)

    async def _handle_session_init(self, msg: dict[str, Any], ws: Any) -> None:
        """Process a session_init from a browser — unwrap session key, send accept."""
        session_init = msg.get("session_init")
        if not session_init:
            log.error("session_init missing payload")
            return

        try:
            result = open_session_init(
                self.config.transport_private_key_b64,
                session_init,
            )
        except Exception as exc:
            log.error("Failed to open session_init: %s", exc)
            return

        session_id = result["session_id"]
        session_key_b64 = result["session_key_b64"]

        # Store the session.
        self._sessions[session_id] = ActiveSession(
            session_id=session_id,
            session_key_b64=session_key_b64,
        )

        log.info("E2EE session established: %s", session_id[:12])

        # Build and send session_accept back through the relay.
        accept_envelope = build_session_accept(
            session_key_b64=session_key_b64,
            session_id=session_id,
            route_to="client",
        )

        await ws.send(json.dumps({
            "type": "session_accept",
            "session_id": session_id,
            "envelope": accept_envelope,
        }))
        log.info("Sent session_accept for %s", session_id[:12])

    async def _handle_envelope(self, msg: dict[str, Any], ws: Any) -> None:
        """Decrypt an incoming envelope and process the inner frame."""
        session_id = msg.get("session_id")
        envelope = msg.get("envelope")

        if not session_id or not envelope:
            log.error("e2ee_envelope missing session_id or envelope")
            return

        session = self._sessions.get(session_id)
        if not session:
            log.warning("Envelope for unknown session: %s", session_id[:12] if session_id else "?")
            return

        try:
            frame = decrypt_envelope(session.session_key_b64, envelope)
        except Exception as exc:
            log.error("Failed to decrypt envelope: %s", exc)
            # Session may be compromised — close it.
            self._sessions.pop(session_id, None)
            return

        frame_type = frame.get("frame_type")
        sender = frame.get("sender")
        payload = frame.get("payload")
        message_id = frame.get("message_id")

        if frame_type == "close":
            log.info("Session %s closed by %s", session_id[:12], sender)
            self._sessions.pop(session_id, None)
            return

        if frame_type == "data":
            await self._handle_data_frame(session, frame, ws)

    async def _handle_data_frame(
        self,
        session: ActiveSession,
        frame: dict[str, Any],
        ws: Any,
    ) -> None:
        """Process a decrypted data frame — dispatch by payload type."""
        payload = frame.get("payload") or {}
        action = payload.get("action", "message")

        if action == "list_channels":
            await self._send_channel_list(session, ws)
        elif action == "create_channel":
            await self._create_channel(session, payload, ws)
        elif action == "get_messages":
            await self._send_messages(session, payload, ws)
        elif action == "get_activity":
            await self._send_activity(session, payload, ws)
        elif action == "message":
            await self._handle_chat_message(session, frame, ws)
        elif action == "mark_read":
            await self._mark_read(session, payload, ws)
        elif action == "list_harnesses":
            await self._send_harness_list(session, ws)
        elif action == "start_agent":
            await self._start_agent(session, payload, ws)
        elif action == "stop_agent":
            await self._stop_agent(session, payload, ws)
        elif action == "restart_agent":
            await self._restart_agent(session, payload, ws)
        elif action == "rename_channel":
            await self._rename_channel(session, payload, ws)
        elif action == "delete_channel":
            await self._delete_channel(session, payload, ws)
        elif action == "list_workers":
            await self._send_worker_list(session, ws)
        elif action == "interaction_response":
            await self._handle_interaction_response(session, payload, ws)
        elif action == "upload_chunk":
            await self._handle_upload_chunk(session, payload, ws)
        elif action == "upload_complete":
            await self._handle_upload_complete(session, payload, ws)
        elif action == "complication:action":
            await self._handle_complication_action(session, payload, ws)
        elif action == "get_complications":
            await self._send_complications(session, payload, ws)
        else:
            log.warning("Unknown action: %s", action)

    async def _send_channel_list(self, session: ActiveSession, ws: Any) -> None:
        """Send the list of channels to the browser."""
        channels = self.store.list_channels()
        channel_list = []
        for c in channels:
            entry: dict[str, Any] = {"id": c.id, "name": c.name, "created_at": c.created_at}
            # Enrich with agent info if available.
            if self._agent_server:
                agent_ch = self._agent_server.store.get_channel(c.id)
                if agent_ch:
                    from build_client.harness_registry import get_harness
                    entry["harness"] = agent_ch.harness
                    entry["model"] = agent_ch.model
                    info = get_harness(agent_ch.harness)
                    entry["agent_name"] = info.name if info else "Device"
                    entry["plan_mode"] = agent_ch.plan_mode
            channel_list.append(entry)
        await self._send_frame(
            session, ws,
            payload={"action": "channel_list", "channels": channel_list},
        )

        # Send current complications so the browser has state immediately.
        if self._agent_server and self._agent_server._complications:
            try:
                comps = await self._agent_server._complications.get_current_complications(
                    agent_store=self._agent_server.store,
                )
                for comp in comps:
                    await self._send_frame(session, ws, payload=comp)
            except Exception as exc:
                log.debug("Failed to send initial complications: %s", exc)

    async def _create_channel(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Create a new channel and confirm to browser.

        If harness config is provided, auto-spawn an agent on the new channel.
        """
        name = payload.get("name", "").strip()
        if not name:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": "channel name required"},
            )
            return

        channel_id = str(uuid.uuid4())
        channel = self.store.create_channel(channel_id, name)
        log.info("Created channel: %s (%s)", name, channel_id[:8])

        response: dict[str, Any] = {
            "action": "channel_created",
            "channel": {
                "id": channel.id,
                "name": channel.name,
                "created_at": channel.created_at,
            },
        }

        # Auto-spawn agent if harness config provided.
        harness = payload.get("harness", "")
        if harness and self._agent_spawner:
            model = payload.get("model", "")
            system_prompt = payload.get("system_prompt", "")
            working_directory = payload.get("working_directory", "")

            log.info(
                "create_channel payload: harness=%r, model=%r, "
                "system_prompt=%r, working_directory=%r, all_keys=%s",
                harness, model, system_prompt, working_directory,
                list(payload.keys()),
            )

            harness_info = get_harness(harness)
            if harness_info:
                if not model:
                    model = harness_info.default_model
                try:
                    worker = await self._agent_spawner.spawn(
                        channel_id=channel_id,
                        harness=harness,
                        model=model,
                        system_prompt=system_prompt,
                        working_directory=working_directory,
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

        await self._send_frame(session, ws, payload=response)

    async def _rename_channel(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Rename an existing channel."""
        channel_id = payload.get("channel_id", "")
        name = payload.get("name", "").strip()
        if not channel_id or not name:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": "channel_id and name required"},
            )
            return

        self.store.rename_channel(channel_id, name)
        log.info("Renamed channel %s to %r", channel_id[:8], name)
        await self._send_frame(
            session, ws,
            payload={
                "action": "channel_renamed",
                "channel_id": channel_id,
                "name": name,
            },
        )

    async def _delete_channel(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Delete a channel, stopping its agent first if running."""
        channel_id = payload.get("channel_id", "")
        if not channel_id:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": "channel_id required"},
            )
            return

        # Stop agent if running.
        if self._agent_spawner:
            try:
                await self._agent_spawner.stop(channel_id)
            except Exception as exc:
                log.debug("Error stopping agent for deleted channel %s: %s", channel_id[:8], exc)

        # Clean up message store.
        self.store.delete_channel(channel_id)
        # Clean up agent store.
        if self._agent_server and self._agent_server.store:
            try:
                self._agent_server.store.delete_channel(channel_id)
            except Exception as exc:
                log.debug("Error cleaning agent store for channel %s: %s", channel_id[:8], exc)
        log.info("Deleted channel %s", channel_id[:8])
        await self._send_frame(
            session, ws,
            payload={
                "action": "channel_deleted",
                "channel_id": channel_id,
            },
        )

    async def _send_messages(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Send message history for a channel.

        Merges E2EE messages (user + device error messages) with agent chat
        messages (from the AgentStore) into a single chronological list.
        """
        channel_id = payload.get("channel_id", "")
        limit = payload.get("limit", 50)
        before = payload.get("before")

        # E2EE messages (user messages, device error messages).
        e2ee_msgs = self.store.get_messages(channel_id, limit=limit, before=before)

        # Agent chat messages (user + assistant from AgentStore).
        agent_msgs: list[dict[str, Any]] = []
        if self._agent_server:
            try:
                from datetime import datetime, timezone
                agent_name = self._get_agent_name(channel_id)
                chat_history = self._agent_server.store.get_chat_history(channel_id)
                # Collect IDs from E2EE messages to avoid duplicates (user messages
                # appear in both stores).
                e2ee_ids = {m.id for m in e2ee_msgs}
                for cm in chat_history:
                    if cm.id in e2ee_ids:
                        continue  # Already in E2EE messages.
                    # Parse ISO created_at to float timestamp for consistent sorting.
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
                    agent_msgs.append(msg_dict)
            except Exception as exc:
                log.error("Failed to merge agent chat messages: %s", exc)

        # Merge and sort chronologically.
        combined = []
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

        combined.sort(key=lambda m: m.get("created_at") or 0)

        # Apply limit after merge.
        if len(combined) > limit:
            combined = combined[-limit:]

        log.info(
            "Sending %d messages for channel %s (e2ee=%d, agent=%d)",
            len(combined), channel_id[:8], len(e2ee_msgs), len(agent_msgs),
        )

        await self._send_frame(
            session, ws,
            payload={
                "action": "messages",
                "channel_id": channel_id,
                "messages": combined,
            },
        )

    # Max activity entries to send in a single activity_history response.
    # The relay enforces a 256 KB envelope limit; large histories (hundreds of
    # tool uses with big inputs) easily exceed that.  We keep the last N
    # *tool_use* entries plus their matching tool_result entries so the console
    # shows recent activity without blowing the size budget.
    # After encryption + base64 overhead (~33%), 150 KB payload ≈ 200 KB envelope.
    _MAX_ACTIVITY_TOOL_USES = 50
    _MAX_RESULT_CONTENT_LEN = 1000  # Truncate tool_result content for history

    async def _send_activity(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Send tool use history for a channel.

        Returns tool_use and tool_result entries from the AgentStore activity
        log so the browser can populate the console on page load / reconnect.
        """
        channel_id = payload.get("channel_id", "")

        entries: list[dict[str, Any]] = []
        if self._agent_server:
            try:
                import json as _json
                activity = self._agent_server.store.get_activity_history(channel_id)
                for entry in activity:
                    if entry.type in ("tool_use", "tool_result", "text"):
                        try:
                            data = _json.loads(entry.data)
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
                            if isinstance(c, str) and len(c) > self._MAX_RESULT_CONTENT_LEN:
                                data["content"] = c[:self._MAX_RESULT_CONTENT_LEN] + "…"
                        # Trim reasoning text for history view.
                        if entry.type == "text":
                            c = data.get("content", "")
                            if isinstance(c, str) and len(c) > self._MAX_RESULT_CONTENT_LEN:
                                data["content"] = c[:self._MAX_RESULT_CONTENT_LEN] + "…"
                        entries.append({
                            "type": entry.type,
                            "data": data,
                            "created_at": entry.created_at,
                        })
            except Exception as exc:
                log.error("Failed to fetch activity history: %s", exc)

        # Count total tool_uses before trimming for the console counter.
        total_tool_uses = sum(1 for e in entries if e["type"] == "tool_use")

        # Trim to the most recent N tool_use entries (plus their results) so
        # the encrypted envelope stays under the relay's 256 KB size limit.
        if entries:
            entries = self._trim_activity_entries(entries)

        # Safety: check estimated size and halve entries if still too large.
        import json as _json
        payload_json = _json.dumps(entries)
        estimated_encrypted = len(payload_json) * 1.4  # encryption + base64 overhead
        while estimated_encrypted > 200_000 and len(entries) > 10:
            entries = entries[len(entries) // 2:]
            payload_json = _json.dumps(entries)
            estimated_encrypted = len(payload_json) * 1.4
            log.warning(
                "Activity payload too large, reduced to %d entries (%.0f KB est.)",
                len(entries), estimated_encrypted / 1024,
            )

        log.info(
            "Sending %d activity entries for channel %s (%.1f KB)",
            len(entries), channel_id[:8], len(payload_json) / 1024,
        )

        await self._send_frame(
            session, ws,
            payload={
                "action": "activity_history",
                "channel_id": channel_id,
                "entries": entries,
                "total_tool_uses": total_tool_uses,
            },
        )

    async def _send_complications(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Send current complications for a channel."""
        channel_id = payload.get("channel_id", "")
        complications: list[dict[str, Any]] = []

        if self._agent_server and self._agent_server._complications:
            try:
                all_comps = await self._agent_server._complications.get_current_complications(
                    agent_store=self._agent_server.store,
                )
                complications = [c for c in all_comps if c.get("channel_id") == channel_id]
            except Exception as exc:
                log.debug("Failed to get complications for %s: %s", channel_id[:8], exc)

        await self._send_frame(
            session, ws,
            payload={
                "action": "complications",
                "channel_id": channel_id,
                "complications": complications,
            },
        )

    @classmethod
    def _trim_activity_entries(
        cls, entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep the last N tool_use entries and their matching tool_result entries."""
        # Collect tool_use entries and find the cutoff point.
        tool_use_indices: list[int] = []
        for i, e in enumerate(entries):
            if e["type"] == "tool_use":
                tool_use_indices.append(i)

        if len(tool_use_indices) <= cls._MAX_ACTIVITY_TOOL_USES:
            return entries

        # Keep entries from the Nth-from-last tool_use onward.
        cutoff_idx = tool_use_indices[-cls._MAX_ACTIVITY_TOOL_USES]
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

    async def _handle_chat_message(
        self,
        session: ActiveSession,
        frame: dict[str, Any],
        ws: Any,
    ) -> None:
        """Handle an incoming chat message — forward to agent."""
        payload = frame.get("payload") or {}
        message_id = frame.get("message_id")
        channel_id = payload.get("channel_id", "")
        content = payload.get("content", "")
        plan_mode = payload.get("plan_mode")
        attachments = payload.get("attachments")  # list[{file_id, filename, size, mime_type}] or None

        if not channel_id or (not content and not attachments):
            log.warning("Chat message missing channel_id or content: channel_id=%r, content=%r, attachments=%r",
                        channel_id, bool(content), bool(attachments))
            return

        # Store the incoming message from the client.
        self.store.store_message(
            message_id=message_id,
            channel_id=channel_id,
            session_id=session.session_id,
            sender="client",
            content=content,
            attachments=attachments,
        )

        # Send delivered status (message reached device, but agent hasn't read it yet).
        self.store.mark_delivered(message_id)
        await self._send_frame(
            session, ws,
            payload={
                "action": "delivered",
                "message_id": message_id,
                "channel_id": channel_id,
            },
        )

        # NOTE: We do NOT mark as read here. The message is marked read when
        # the agent calls read_unread via the Chat MCP tool. See
        # _on_agent_read_messages() for the read notification flow.

        # Try to forward to a connected agent.
        if not self._agent_server:
            log.error(
                "No agent server registered — cannot forward message on channel %s",
                channel_id[:8],
            )
            await self._send_error_message(session, ws, channel_id, "No agent server available")
            return

        if not self._agent_server.is_channel_active(channel_id):
            # Agent is down — auto-restart it and notify the user.
            log.warning("No active agent on channel %s, attempting auto-restart", channel_id[:8])
            await self._send_system_message(
                session, ws, channel_id,
                "Agent is restarting, please standby...",
            )
            restarted = False
            if self._agent_spawner:
                try:
                    worker = await self._agent_spawner.restart(channel_id)
                    if worker:
                        log.info("Auto-restarted agent on channel %s (pid=%s)", channel_id[:8], worker.pid)
                        restarted = True
                except Exception as exc:
                    log.error("Failed to auto-restart agent on channel %s: %s", channel_id[:8], exc)
            if not restarted:
                await self._send_system_message(
                    session, ws, channel_id,
                    "Failed to restart agent. Try restarting manually.",
                )
                return
            # Wait briefly for the agent to connect, then retry forwarding.
            for _ in range(10):
                await asyncio.sleep(0.5)
                if self._agent_server.is_channel_active(channel_id):
                    break
            else:
                await self._send_system_message(
                    session, ws, channel_id,
                    "Agent is still starting up. Your message will be delivered when it connects.",
                )
                return

        # Enrich attachment metadata with local file paths so agents can read files.
        enriched_attachments = None
        if attachments:
            enriched_attachments = []
            for att in attachments:
                file_path = (
                    self._upload_final_dir(channel_id, att["file_id"])
                    / self._sanitize_filename(att.get("filename", "upload"))
                )
                enriched_attachments.append({**att, "path": str(file_path)})

        # Prepend plan mode instruction if the browser sent plan_mode flag.
        forwarded_content = content
        if plan_mode:
            forwarded_content = (
                "[System: The user has planning mode active. Plan before executing. "
                "Use EnterPlanMode if not already in plan mode.]\n\n" + content
            )
            if self._agent_server:
                self._agent_server.store.update_plan_mode(channel_id, True)

        sent = await self._agent_server.send_chat_message(
            channel_id, forwarded_content, msg_id=message_id,
            attachments=enriched_attachments,
        )
        if sent:
            log.info("Forwarded chat message to agent on channel %s", channel_id[:8])
            # NOTE: Do NOT mark as read here. The message is only "read" when
            # the agent actually processes it via read_unread. Forwarding to
            # the agent WebSocket just queues it.
        else:
            log.error(
                "Failed to send chat message to agent on channel %s "
                "(agent lookup succeeded but send failed)",
                channel_id[:8],
            )
            await self._send_system_message(
                session, ws, channel_id,
                "Message delivery failed. The agent may have disconnected.",
            )

    async def _send_system_message(
        self,
        session: ActiveSession,
        ws: Any,
        channel_id: str,
        text: str,
    ) -> None:
        """Send an ephemeral system message to the browser.

        Not stored in the database — only shown in the current session.
        The browser renders it as a transient notification inline in chat.
        """
        await self._send_frame(
            session, ws,
            payload={
                "action": "system_message",
                "channel_id": channel_id,
                "text": text,
            },
        )

    async def _send_error_message(
        self,
        session: ActiveSession,
        ws: Any,
        channel_id: str,
        error_text: str,
    ) -> None:
        """Send an error as an agent message to the browser."""
        agent_name = self._get_agent_name(channel_id)
        error_id = str(uuid.uuid4())
        self.store.store_message(
            message_id=error_id,
            channel_id=channel_id,
            session_id=session.session_id,
            sender=agent_name,
            content=f"⚠ {error_text}",
        )
        recent = self.store.get_messages(channel_id, limit=1)
        created_at = recent[-1].created_at if recent else None
        await self._send_frame(
            session, ws,
            payload={
                "action": "message",
                "message": {
                    "id": error_id,
                    "channel_id": channel_id,
                    "sender": agent_name,
                    "content": f"⚠ {error_text}",
                    "created_at": created_at,
                },
            },
        )

    async def _mark_read(self, session: ActiveSession, payload: dict[str, Any], ws: Any) -> None:
        """Mark messages as read and confirm to browser."""
        message_ids = payload.get("message_ids", [])
        channel_id = payload.get("channel_id", "")
        for mid in message_ids:
            self.store.mark_read(mid)
        if message_ids:
            await self._send_frame(
                session, ws,
                payload={
                    "action": "read",
                    "message_ids": message_ids,
                    "channel_id": channel_id,
                },
            )

    # ------------------------------------------------------------------
    # Harness & Agent management actions
    # ------------------------------------------------------------------

    async def _send_harness_list(self, session: ActiveSession, ws: Any) -> None:
        """Send list of detected harnesses to the browser."""
        harnesses = detect_installed()
        await self._send_frame(
            session, ws,
            payload={
                "action": "harness_list",
                "harnesses": serialize_harnesses(harnesses),
            },
        )

    async def _start_agent(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Start an agent on a channel."""
        if not self._agent_spawner:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": "agent spawner not available"},
            )
            return

        channel_id = payload.get("channel_id", "")
        harness = payload.get("harness", "")
        model = payload.get("model", "")
        system_prompt = payload.get("system_prompt", "")
        working_directory = payload.get("working_directory", "")

        if not channel_id or not harness:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": "channel_id and harness required"},
            )
            return

        # Validate harness exists.
        harness_info = get_harness(harness)
        if not harness_info:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": f"unknown harness: {harness}"},
            )
            return

        # Default to harness default model if none specified.
        if not model:
            model = harness_info.default_model

        try:
            worker = await self._agent_spawner.spawn(
                channel_id=channel_id,
                harness=harness,
                model=model,
                system_prompt=system_prompt,
                working_directory=working_directory,
            )
            await self._send_frame(
                session, ws,
                payload={
                    "action": "agent_started",
                    "channel_id": channel_id,
                    "agent_id": worker.agent_id,
                    "harness": harness,
                    "model": model,
                    "pid": worker.pid,
                },
            )
        except Exception as exc:
            log.error("Failed to start agent on channel %s: %s", channel_id[:8], exc)
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": f"failed to start agent: {exc}"},
            )

    async def _stop_agent(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Stop the agent on a channel."""
        if not self._agent_spawner:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": "agent spawner not available"},
            )
            return

        channel_id = payload.get("channel_id", "")
        if not channel_id:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": "channel_id required"},
            )
            return

        stopped = await self._agent_spawner.stop(channel_id)
        await self._send_frame(
            session, ws,
            payload={
                "action": "agent_stopped",
                "channel_id": channel_id,
                "was_running": stopped,
            },
        )

    async def _restart_agent(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Restart the agent on a channel."""
        if not self._agent_spawner:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": "agent spawner not available"},
            )
            return

        channel_id = payload.get("channel_id", "")
        if not channel_id:
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": "channel_id required"},
            )
            return

        try:
            worker = await self._agent_spawner.restart(channel_id)
            if not worker:
                await self._send_frame(
                    session, ws,
                    payload={"action": "error", "error": f"no agent config found for channel {channel_id[:8]}"},
                )
                return

            await self._send_frame(
                session, ws,
                payload={
                    "action": "agent_restarted",
                    "channel_id": channel_id,
                    "agent_id": worker.agent_id,
                    "pid": worker.pid,
                },
            )
        except Exception as exc:
            log.error("Failed to restart agent on channel %s: %s", channel_id[:8], exc)
            await self._send_frame(
                session, ws,
                payload={"action": "error", "error": f"failed to restart agent: {exc}"},
            )

    async def _send_worker_list(self, session: ActiveSession, ws: Any) -> None:
        """Send list of running agent workers to the browser."""
        if not self._agent_spawner:
            await self._send_frame(
                session, ws,
                payload={"action": "worker_list", "workers": []},
            )
            return

        workers = self._agent_spawner.list_workers()
        await self._send_frame(
            session, ws,
            payload={
                "action": "worker_list",
                "workers": workers,
            },
        )

    # ------------------------------------------------------------------
    # Interaction & plan mode actions
    # ------------------------------------------------------------------

    async def _handle_interaction_response(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Forward user's interaction response to the agent."""
        channel_id = payload.get("channel_id", "")
        interaction_id = payload.get("interaction_id", "")
        selected_option = payload.get("selected_option")
        freeform_response = payload.get("freeform_response")

        if not channel_id or not interaction_id:
            log.warning("interaction_response missing channel_id or interaction_id")
            return

        # Persist the response.
        if self._agent_server:
            self._agent_server.store.resolve_interaction(
                interaction_id, channel_id, selected_option, freeform_response,
            )

        # Forward to agent.
        if self._agent_server:
            sent = await self._agent_server.send_interaction_response(
                channel_id, interaction_id, selected_option, freeform_response,
            )
            if sent:
                log.info("Forwarded interaction.response to agent on channel %s", channel_id[:8])
            else:
                log.error("Failed to forward interaction.response on channel %s", channel_id[:8])
                await self._send_error_message(
                    session, ws, channel_id,
                    "Failed to send response to agent. The agent may have disconnected.",
                )

    # ---- File Upload Handling ----

    _UPLOADS_BASE = Path.home() / ".config" / "build" / "uploads"
    _MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

    def _upload_tmp_dir(self, file_id: str) -> Path:
        return self._UPLOADS_BASE / "tmp" / file_id

    def _upload_final_dir(self, channel_id: str, file_id: str) -> Path:
        return self._UPLOADS_BASE / channel_id / file_id

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove path separators and dangerous characters from a filename."""
        name = os.path.basename(name)
        # Strip leading dots to prevent hidden files.
        name = name.lstrip(".")
        return name or "upload"

    async def _handle_upload_chunk(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Receive and store a single file chunk, then ack."""
        file_id = payload.get("file_id", "")
        channel_id = payload.get("channel_id", "")
        chunk_index = payload.get("chunk_index", 0)
        total_size = payload.get("total_size", 0)
        data_b64 = payload.get("data", "")

        if not file_id or not channel_id or not data_b64:
            await self._send_frame(session, ws, payload={
                "action": "upload_error",
                "file_id": file_id,
                "error": "missing required fields",
            })
            return

        if total_size > self._MAX_FILE_SIZE:
            await self._send_frame(session, ws, payload={
                "action": "upload_error",
                "file_id": file_id,
                "error": f"file too large (max {self._MAX_FILE_SIZE // (1024*1024)} MB)",
            })
            return

        try:
            import base64
            chunk_data = base64.b64decode(data_b64 + "==")  # pad for standard base64
        except Exception:
            # Try libsodium-style no-padding base64
            try:
                import base64 as _b64
                # Add padding
                padded = data_b64 + "=" * (-len(data_b64) % 4)
                chunk_data = _b64.b64decode(padded)
            except Exception as exc:
                await self._send_frame(session, ws, payload={
                    "action": "upload_error",
                    "file_id": file_id,
                    "error": f"invalid chunk data: {exc}",
                })
                return

        tmp_dir = self._upload_tmp_dir(file_id)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Write chunk file.
        chunk_path = tmp_dir / f"chunk_{chunk_index}"
        chunk_path.write_bytes(chunk_data)

        # Store metadata on first chunk.
        meta_path = tmp_dir / "meta.json"
        if chunk_index == 0:
            meta = {
                "filename": self._sanitize_filename(payload.get("filename", "upload")),
                "mime_type": payload.get("mime_type", "application/octet-stream"),
                "total_size": total_size,
                "total_chunks": payload.get("total_chunks", 1),
                "channel_id": channel_id,
            }
            meta_path.write_text(json.dumps(meta))

        log.info(
            "Received chunk %d for upload %s (%d bytes)",
            chunk_index, file_id[:8], len(chunk_data),
        )

        await self._send_frame(session, ws, payload={
            "action": "chunk_ack",
            "file_id": file_id,
            "chunk_index": chunk_index,
        })

    async def _handle_upload_complete(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Assemble chunks, verify hash, and move to final location."""
        file_id = payload.get("file_id", "")
        channel_id = payload.get("channel_id", "")
        expected_sha256 = payload.get("sha256", "")

        tmp_dir = self._upload_tmp_dir(file_id)
        meta_path = tmp_dir / "meta.json"

        if not meta_path.exists():
            await self._send_frame(session, ws, payload={
                "action": "upload_error",
                "file_id": file_id,
                "error": "no chunks received for this upload",
            })
            return

        meta = json.loads(meta_path.read_text())
        total_chunks = meta["total_chunks"]
        filename = meta["filename"]

        # Assemble chunks.
        assembled = bytearray()
        for i in range(total_chunks):
            chunk_path = tmp_dir / f"chunk_{i}"
            if not chunk_path.exists():
                await self._send_frame(session, ws, payload={
                    "action": "upload_error",
                    "file_id": file_id,
                    "error": f"missing chunk {i}",
                })
                return
            assembled.extend(chunk_path.read_bytes())

        # Verify SHA-256.
        actual_sha256 = hashlib.sha256(assembled).hexdigest()
        if expected_sha256 and actual_sha256 != expected_sha256:
            log.error(
                "Upload %s hash mismatch: expected %s, got %s",
                file_id[:8], expected_sha256[:12], actual_sha256[:12],
            )
            # Clean up temp files.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            await self._send_frame(session, ws, payload={
                "action": "upload_error",
                "file_id": file_id,
                "error": "SHA-256 mismatch — file corrupted in transit",
            })
            return

        # Write assembled file to final location.
        final_dir = self._upload_final_dir(channel_id, file_id)
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / filename
        final_path.write_bytes(assembled)

        # Clean up temp.
        shutil.rmtree(tmp_dir, ignore_errors=True)

        log.info(
            "Upload complete: %s → %s (%d bytes, sha256=%s)",
            file_id[:8], final_path, len(assembled), actual_sha256[:12],
        )

        await self._send_frame(session, ws, payload={
            "action": "upload_accepted",
            "file_id": file_id,
            "filename": filename,
            "size": len(assembled),
            "path": str(final_path),
        })

    async def _handle_complication_action(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Handle a user action on a complication (e.g. git push)."""
        if not self._agent_server or not self._agent_server._complications:
            log.warning("Complication action received but no complications registry")
            return
        channel_id = payload.get("channel_id", "")
        complication_id = payload.get("complication_id", "")
        option_id = payload.get("option_id", "")
        if not channel_id or not complication_id or not option_id:
            log.warning("Complication action missing required fields")
            return
        await self._agent_server._complications.handle_action(
            channel_id, complication_id, option_id,
        )

    async def _send_frame(
        self,
        session: ActiveSession,
        ws: Any,
        payload: dict[str, Any],
    ) -> None:
        """Encrypt and send a frame to the browser via the relay."""
        envelope = encrypt_frame(
            session_key_b64=session.session_key_b64,
            outer_fields={
                "session_id": session.session_id,
                "route_to": "client",
            },
            frame_fields={
                "frame_type": "data",
                "sender": "device",
                "payload": payload,
            },
        )

        await ws.send(json.dumps({
            "type": "e2ee_envelope",
            "session_id": session.session_id,
            "envelope": envelope,
        }))
