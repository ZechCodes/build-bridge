"""Device-side E2EE handler — session management, decryption, and agent relay."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
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
            log.debug("No relay WS connection — skipping broadcast")
            return

        for session in list(self._sessions.values()):
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
        elif action == "list_workers":
            await self._send_worker_list(session, ws)
        else:
            log.warning("Unknown action: %s", action)

    async def _send_channel_list(self, session: ActiveSession, ws: Any) -> None:
        """Send the list of channels to the browser."""
        channels = self.store.list_channels()
        await self._send_frame(
            session, ws,
            payload={
                "action": "channel_list",
                "channels": [
                    {"id": c.id, "name": c.name, "created_at": c.created_at}
                    for c in channels
                ],
            },
        )

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

    async def _send_messages(
        self,
        session: ActiveSession,
        payload: dict[str, Any],
        ws: Any,
    ) -> None:
        """Send message history for a channel."""
        channel_id = payload.get("channel_id", "")
        limit = payload.get("limit", 50)
        before = payload.get("before")

        messages = self.store.get_messages(channel_id, limit=limit, before=before)
        await self._send_frame(
            session, ws,
            payload={
                "action": "messages",
                "channel_id": channel_id,
                "messages": [
                    {
                        "id": m.id,
                        "channel_id": m.channel_id,
                        "sender": m.sender,
                        "content": m.content,
                        "created_at": m.created_at,
                        "delivered_at": m.delivered_at,
                        "read_at": m.read_at,
                    }
                    for m in messages
                ],
            },
        )

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

        if not channel_id or not content:
            log.warning("Chat message missing channel_id or content: channel_id=%r, content=%r", channel_id, bool(content))
            return

        # Store the incoming message from the client.
        self.store.store_message(
            message_id=message_id,
            channel_id=channel_id,
            session_id=session.session_id,
            sender="client",
            content=content,
        )

        # Send delivered status.
        await self._send_frame(
            session, ws,
            payload={
                "action": "delivered",
                "message_id": message_id,
                "channel_id": channel_id,
            },
        )

        # Mark as read (device has processed it).
        self.store.mark_read(message_id)
        await self._send_frame(
            session, ws,
            payload={
                "action": "read",
                "message_ids": [message_id],
                "channel_id": channel_id,
            },
        )

        # Try to forward to a connected agent.
        if not self._agent_server:
            log.error(
                "No agent server registered — cannot forward message on channel %s",
                channel_id[:8],
            )
            await self._send_error_message(session, ws, channel_id, "No agent server available")
            return

        if not self._agent_server.is_channel_active(channel_id):
            # Log detailed diagnostics.
            known_channels = list(self._agent_server._channel_to_agent.keys())
            spawner_workers = list(self._agent_spawner.workers.keys()) if self._agent_spawner else []
            log.error(
                "No active agent on channel %s. "
                "Agent server active channels: %s. "
                "Spawner tracked workers: %s",
                channel_id[:8],
                [c[:8] for c in known_channels],
                [w[:8] for w in spawner_workers],
            )
            await self._send_error_message(
                session, ws, channel_id,
                "No agent connected on this channel. Try restarting the agent.",
            )
            return

        sent = await self._agent_server.send_chat_message(
            channel_id, content, msg_id=message_id,
        )
        if sent:
            log.info("Forwarded chat message to agent on channel %s", channel_id[:8])
        else:
            log.error(
                "Failed to send chat message to agent on channel %s "
                "(agent lookup succeeded but send failed)",
                channel_id[:8],
            )
            await self._send_error_message(
                session, ws, channel_id,
                "Failed to deliver message to agent. The agent may have disconnected.",
            )

    async def _send_error_message(
        self,
        session: ActiveSession,
        ws: Any,
        channel_id: str,
        error_text: str,
    ) -> None:
        """Send an error as a device message to the browser."""
        error_id = str(uuid.uuid4())
        self.store.store_message(
            message_id=error_id,
            channel_id=channel_id,
            session_id=session.session_id,
            sender="device",
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
                    "sender": "device",
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
