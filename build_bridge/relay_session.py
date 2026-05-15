"""E2EE session lifecycle and frame routing.

Owns the crypto and the dispatcher. The high-level shape:

1. `session_init` from the browser → derive the session key, send back a
   `session_accept`.
2. `e2ee_envelope` from either side → decrypt, then either close the
   session (frame_type=close) or dispatch the data frame by
   `payload.action` to the handler registered in
   `relay_handlers.build_dispatch_table`.
3. Handlers call back into `send_frame` / `broadcast_to_sessions` when
   they need to emit a frame.

Session state (`_sessions`, `_relay_ws`) lives on the E2EEHandler facade,
not on this class, so tests that reach into `handler._sessions` keep
working. We mutate it through the facade reference held in `_facade`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from build_secure_transport import (
    build_session_accept,
    decrypt_envelope,
    encrypt_frame,
    open_session_init,
)

from build_bridge import relay_protocol as proto

if TYPE_CHECKING:
    from build_bridge.e2ee import E2EEHandler

log = logging.getLogger(__name__)


@dataclass
class ActiveSession:
    """An active E2EE session with a browser client."""
    session_id: str
    session_key_b64: str
    protocol_version: int = 0


class E2EESession:
    """Frame-level transport for one browser-facing E2EE channel.

    One instance per E2EEHandler. The handler owns the dict of active
    sessions and the relay WebSocket; we read/mutate those through the
    facade reference so test-time monkeypatches remain effective.
    """

    def __init__(self, facade: "E2EEHandler") -> None:
        self._facade = facade
        # Set lazily by the facade once handlers are wired (avoids a
        # circular dependency at construction time — the dispatch table
        # imports handler modules that import this module's ActiveSession).
        self._dispatch: dict[str, Any] = {}

    def set_dispatch_table(self, table: dict[str, Any]) -> None:
        """Late binding: the facade builds the dispatch table after wiring."""
        self._dispatch = table

    # ------------------------------------------------------------------
    # Entry point — wired into ws.py via E2EEHandler.handle_message
    # ------------------------------------------------------------------

    async def handle_message(self, msg: dict[str, Any], ws: Any) -> None:
        """Dispatch incoming WS messages from the relay."""
        self._facade._relay_ws = ws  # Track for broadcasts.

        msg_type = msg.get("type")
        if msg_type == proto.WS_SESSION_INIT:
            await self._handle_session_init(msg, ws)
        elif msg_type == proto.WS_E2EE_ENVELOPE:
            await self._handle_envelope(msg, ws)
        else:
            log.warning("E2EE handler got unknown WS type: %s", msg_type)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _handle_session_init(self, msg: dict[str, Any], ws: Any) -> None:
        """Process a session_init from a browser — unwrap session key, send accept."""
        session_init = msg.get("session_init")
        if not session_init:
            log.error("session_init missing payload")
            return

        try:
            result = open_session_init(
                self._facade.config.transport_private_key_b64,
                session_init,
            )
        except Exception as exc:
            log.error("Failed to open session_init: %s", exc)
            return

        session_id = result["session_id"]
        session_key_b64 = result["session_key_b64"]

        self._facade._sessions[session_id] = ActiveSession(
            session_id=session_id,
            session_key_b64=session_key_b64,
        )

        log.info("E2EE session established: %s", session_id[:12])

        accept_envelope = build_session_accept(
            session_key_b64=session_key_b64,
            session_id=session_id,
            route_to=proto.ROUTE_TO_CLIENT,
        )

        await ws.send(json.dumps({
            "type": proto.WS_SESSION_ACCEPT,
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

        session = self._facade._sessions.get(session_id)
        if not session:
            log.warning(
                "Envelope for unknown session: %s",
                session_id[:12] if session_id else "?",
            )
            return

        try:
            frame = decrypt_envelope(session.session_key_b64, envelope)
        except Exception as exc:
            log.error("Failed to decrypt envelope: %s", exc)
            # Session may be compromised — close it.
            self._facade._sessions.pop(session_id, None)
            return

        frame_type = frame.get("frame_type")

        if frame_type == proto.FRAME_TYPE_CLOSE:
            sender = frame.get("sender")
            log.info("Session %s closed by %s", session_id[:12], sender)
            self._facade._sessions.pop(session_id, None)
            return

        if frame_type == proto.FRAME_TYPE_DATA:
            await self._handle_data_frame(session, frame, ws)
            return

        from build_bridge.relay_v1 import extract_v1_envelope
        if extract_v1_envelope(frame) is not None:
            await self._handle_data_frame(session, frame, ws)

    # ------------------------------------------------------------------
    # Data frame dispatch
    # ------------------------------------------------------------------

    async def _handle_data_frame(
        self,
        session: ActiveSession,
        frame: dict[str, Any],
        ws: Any,
    ) -> None:
        """Route a decrypted data frame to the registered handler.

        Validates payload shape first via `relay_protocol.VALIDATORS`.
        Validation failures emit a structured error frame (see
        `make_validation_error`) and short-circuit before the handler runs.
        """
        # v1 application envelopes ride inside the secure-transport payload
        # while v0 clients keep sending payload.action frames.
        from build_bridge.relay_v1 import extract_v1_envelope

        v1_app = extract_v1_envelope(frame)
        if v1_app is not None:
            await self._facade._handle_v1_frame(session, v1_app, ws)
            return

        payload = frame.get("payload") or {}
        action = payload.get("action", proto.DEFAULT_ACTION)

        # Light logging — match the legacy behaviour of being noisy only on
        # interesting actions (skip the polling-style reads).
        if action not in (
            proto.LIST_CHANNELS,
            proto.GET_MESSAGES,
            proto.GET_ACTIVITY,
            proto.GET_COMPLICATIONS,
            proto.MESSAGE,
        ):
            log.info(
                "E2EE action: %s (channel=%s)",
                action, str(payload.get("channel_id", "?"))[:8],
            )

        # Validate payload shape against the registered schema.
        validator = proto.VALIDATORS.get(action)
        if validator:
            valid, err = validator(payload)
            if not valid:
                log.warning(
                    "Validation failed for %s: %s (channel=%s)",
                    action, err, str(payload.get("channel_id", "?"))[:8],
                )
                # Route through the facade's `_send_frame` so tests that
                # monkeypatch `handler._send_frame = fake_send` intercept
                # validator-emitted error frames too.
                await self._facade._send_frame(
                    session, ws,
                    proto.make_validation_error(action, err, payload),
                )
                return

        # `message` and `retry_message` need the outer message_id, so they
        # are handled by methods on the facade rather than via the action
        # table. (The facade calls into relay_handlers.messages for the
        # actual work; this branch just routes the frame correctly.)
        if action == proto.MESSAGE:
            await self._facade._handle_chat_message(session, frame, ws)
            return
        if action == proto.RETRY_MESSAGE:
            await self._facade._retry_message(session, frame, ws)
            return

        handler = self._dispatch.get(action)
        if handler:
            await handler(session, payload, ws)
        else:
            log.warning("Unknown action: %s", action)

    # ------------------------------------------------------------------
    # Frame emission (called by handlers via HandlerContext)
    # ------------------------------------------------------------------

    async def send_frame(
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
                "route_to": proto.ROUTE_TO_CLIENT,
            },
            frame_fields={
                "frame_type": proto.FRAME_TYPE_DATA,
                "sender": proto.SENDER_DEVICE,
                "payload": payload,
            },
        )

        await ws.send(json.dumps({
            "type": proto.WS_E2EE_ENVELOPE,
            "session_id": session.session_id,
            "envelope": envelope,
        }))

    async def broadcast_to_sessions(
        self,
        channel_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Broadcast an encrypted payload to every active browser session."""
        relay_ws = self._facade._relay_ws
        if not relay_ws:
            log.warning(
                "No relay WS connection — skipping broadcast for %s",
                payload.get("event_type", payload.get("action", "?")),
            )
            return

        sessions = list(self._facade._sessions.values())
        if not sessions:
            log.warning(
                "No active sessions — skipping broadcast for %s",
                payload.get("event_type", payload.get("action", "?")),
            )
            return

        log.info(
            "Broadcasting %s to %d session(s) on channel %s",
            payload.get("event_type", payload.get("action", "?")),
            len(sessions), channel_id[:8],
        )

        dead_sessions: list[str] = []
        for session in sessions:
            try:
                out_payload = payload
                if session.protocol_version == 1:
                    from build_bridge.relay_v1 import translate_v0_broadcast
                    out_payload = translate_v0_broadcast(self._facade, payload)
                await self.send_frame(session, relay_ws, payload=out_payload)
            except Exception as exc:
                log.error(
                    "Failed to broadcast to session %s: %s",
                    session.session_id[:12], exc,
                )
                dead_sessions.append(session.session_id)

        for sid in dead_sessions:
            self._facade._sessions.pop(sid, None)
        if dead_sessions:
            log.info("Removed %d dead session(s)", len(dead_sessions))
