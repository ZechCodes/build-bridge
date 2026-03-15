"""Device-side E2EE handler — session management, decryption, and echo chat."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from build_secure_transport import (
    build_session_accept,
    decrypt_envelope,
    encrypt_frame,
    open_session_init,
)

from build_client.config import DeviceConfig
from build_client.storage import MessageStore

log = logging.getLogger(__name__)


@dataclass
class ActiveSession:
    """An active E2EE session with a browser client."""
    session_id: str
    session_key_b64: str


class E2EEHandler:
    """Handles E2EE protocol on the device side.

    Manages sessions, decrypts incoming envelopes, stores messages,
    and implements echo-chat for MVP.
    """

    def __init__(self, config: DeviceConfig, store: MessageStore) -> None:
        self.config = config
        self.store = store
        self._sessions: dict[str, ActiveSession] = {}

    async def handle_message(self, msg: dict[str, Any], ws: Any) -> None:
        """Dispatch incoming WS messages from the server."""
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
            self._mark_read(payload)
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
        """Create a new channel and confirm to browser."""
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

        await self._send_frame(
            session, ws,
            payload={
                "action": "channel_created",
                "channel": {
                    "id": channel.id,
                    "name": channel.name,
                    "created_at": channel.created_at,
                },
            },
        )

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
        """Handle an incoming chat message — store it and echo back."""
        payload = frame.get("payload") or {}
        message_id = frame.get("message_id")
        channel_id = payload.get("channel_id", "")
        content = payload.get("content", "")

        if not channel_id or not content:
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

        # Echo the message back as a device response.
        echo_id = str(uuid.uuid4())
        echo_content = f"Echo: {content}"

        self.store.store_message(
            message_id=echo_id,
            channel_id=channel_id,
            session_id=session.session_id,
            sender="device",
            content=echo_content,
        )

        await self._send_frame(
            session, ws,
            payload={
                "action": "message",
                "message": {
                    "id": echo_id,
                    "channel_id": channel_id,
                    "sender": "device",
                    "content": echo_content,
                    "created_at": self.store.get_messages(channel_id, limit=1)[-1].created_at
                    if self.store.get_messages(channel_id, limit=1) else None,
                },
            },
        )

    def _mark_read(self, payload: dict[str, Any]) -> None:
        """Mark messages as read."""
        message_ids = payload.get("message_ids", [])
        for mid in message_ids:
            self.store.mark_read(mid)

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
