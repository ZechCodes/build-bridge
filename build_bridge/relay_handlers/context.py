"""HandlerContext — the dependency bundle passed to every relay handler.

Holds a back-reference to the E2EEHandler facade so that:

- handler-side `ctx.send_frame()` calls route through the facade's
  `_send_frame` method. Tests can monkeypatch
  `handler._send_frame = fake_send` and intercept every frame, just as
  they did when handlers were methods on E2EEHandler.

- handler-side reads of `agent_server` / `agent_spawner` go through
  property accessors that re-read the facade attributes on every call.
  Test code that does `handler._agent_server = mock` after construction
  still works.

- the shared path / channel helpers live in one place (`get_channel_cwd`,
  `resolve_safe_path`, `get_agent_name`) rather than duplicated across
  modules.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from build_bridge import relay_protocol as proto
from build_bridge.harness_registry import get_harness

if TYPE_CHECKING:
    from build_bridge.agent_server import AgentServer
    from build_bridge.agent_spawner import AgentSpawner
    from build_bridge.config import DeviceConfig
    from build_bridge.e2ee import E2EEHandler
    from build_bridge.relay_session import ActiveSession
    from build_bridge.storage import MessageStore

log = logging.getLogger(__name__)


@dataclass
class HandlerContext:
    """Dependency bundle passed to every relay handler.

    The single `facade` field is the E2EEHandler instance. All accessors
    below are read through the facade so test-time assignments stick.
    """

    facade: "E2EEHandler"

    # ------------------------------------------------------------------
    # Accessors (read through facade so test-time monkeypatching wins)
    # ------------------------------------------------------------------

    @property
    def config(self) -> "DeviceConfig":
        return self.facade.config

    @property
    def e2ee_store(self) -> "MessageStore":
        return self.facade.store

    @property
    def agent_server(self) -> "AgentServer | None":
        return self.facade._agent_server

    @property
    def agent_spawner(self) -> "AgentSpawner | None":
        return self.facade._agent_spawner

    @property
    def terminal_procs(self) -> dict[str, tuple[Any, str]]:
        return self.facade._terminal_procs

    @property
    def cookie_jars(self) -> dict[str, Any]:
        return self.facade._cookie_jars

    # ------------------------------------------------------------------
    # Frame-emitting helpers — every call hops through the facade so that
    # tests that reassign `facade._send_frame` intercept correctly.
    # ------------------------------------------------------------------

    async def send_frame(
        self,
        session: "ActiveSession",
        ws: Any,
        payload: dict[str, Any],
    ) -> None:
        await self.facade._send_frame(session, ws, payload)

    async def broadcast(self, channel_id: str, payload: dict[str, Any]) -> None:
        await self.facade.broadcast_to_sessions(channel_id, payload)

    async def send_system_message(
        self,
        session: "ActiveSession",
        ws: Any,
        channel_id: str,
        text: str,
    ) -> None:
        """Send an ephemeral system message to the browser.

        Not stored in the database — only shown in the current session.
        """
        await self.send_frame(
            session, ws,
            payload={
                "action": proto.SYSTEM_MESSAGE,
                "channel_id": channel_id,
                "text": text,
            },
        )

    async def send_error_message(
        self,
        session: "ActiveSession",
        ws: Any,
        channel_id: str,
        error_text: str,
    ) -> None:
        """Send an error as an agent message to the browser.

        Stored in the E2EE message store and broadcast as a regular
        `message` so the user sees it in chat history.
        """
        agent_name = self.get_agent_name(channel_id)
        error_id = str(uuid.uuid4())
        self.e2ee_store.store_message(
            message_id=error_id,
            channel_id=channel_id,
            session_id=session.session_id,
            sender=agent_name,
            content=f"⚠ {error_text}",
        )
        recent = self.e2ee_store.get_messages(channel_id, limit=1)
        created_at = recent[-1].created_at if recent else None
        await self.send_frame(
            session, ws,
            payload={
                "action": proto.MESSAGE,
                "message": {
                    "id": error_id,
                    "channel_id": channel_id,
                    "sender": agent_name,
                    "content": f"⚠ {error_text}",
                    "created_at": created_at,
                },
            },
        )

    # ------------------------------------------------------------------
    # Path / channel helpers
    # ------------------------------------------------------------------

    def resolve_safe_path(self, base_dir: str, relative_path: str) -> str | None:
        """Resolve relative_path under base_dir. Returns None if it escapes."""
        base = Path(base_dir).resolve()
        target = (base / relative_path).resolve()
        if target != base and not str(target).startswith(str(base) + os.sep):
            return None
        return str(target)

    def get_channel_cwd(self, channel_id: str) -> str:
        """Get working directory for a channel, falling back to process cwd."""
        if self.agent_server:
            ch = self.agent_server.store.get_channel(channel_id)
            if ch and ch.working_directory:
                return os.path.expanduser(ch.working_directory)
        return os.getcwd()

    def get_agent_name(self, channel_id: str) -> str:
        """Display name for the channel's agent ("Claude Code", "device", ...)."""
        if self.agent_server:
            channel = self.agent_server.store.get_channel(channel_id)
            if channel:
                harness_info = get_harness(channel.harness)
                if harness_info:
                    return harness_info.name
        return "device"
