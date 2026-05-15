"""Device-side E2EE handler — facade over relay_session + relay_handlers.

This module is the public entry point used by `cli.py` and tests. The real
work lives in:

- `relay_session.E2EESession` — session lifecycle, encryption, send/broadcast,
  data-frame dispatch.
- `relay_handlers/*.py` — one module per spec namespace from
  RELAY_PROTOCOL.md, each exposing module-level handler functions.
- `relay_protocol.py` — action name and error code constants.

The `E2EEHandler` class itself remains as a thin facade that wires the
pieces together. It also keeps a handful of `_send_*` / `_handle_*`
methods exposed for tests that monkeypatch them — see RELAY_PROTOCOL.md
and the refactor plan for context.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from build_bridge import relay_protocol as proto
from build_bridge.config import DeviceConfig
from build_bridge.harness_registry import get_harness
from build_bridge.relay_handlers import (
    activity,
    agents,
    channels,
    complications as complications_handlers,
    files,
    interactions,
    messages,
    sessions,
    terminal,
    uploads,
    url,
)
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession, E2EESession
from build_bridge.relay_v1 import V1Protocol
from build_bridge.storage import MessageStore

if TYPE_CHECKING:
    from build_bridge.agent_server import AgentServer
    from build_bridge.agent_spawner import AgentSpawner

log = logging.getLogger(__name__)


class E2EEHandler:
    """Facade for the relay protocol handler.

    Wires `E2EESession` (frame transport + dispatch) and the
    `relay_handlers` package (per-namespace action handlers) into the
    daemon. Keeps test-compatibility shims for the handful of private
    method names that existing tests poke directly.
    """

    # Class attributes preserved here because tests monkeypatch them
    # (`monkeypatch.setattr(E2EEHandler, "_UPLOADS_BASE", ...)` in
    # `test_upload_dest_dir.py`). The uploads handler reads these via
    # `ctx.facade._UPLOADS_BASE` so the monkeypatch remains effective.
    _UPLOADS_BASE = Path.home() / ".config" / "build" / "uploads"
    _MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

    def __init__(self, config: DeviceConfig, store: MessageStore) -> None:
        self.config = config
        self.store = store
        self._sessions: dict[str, ActiveSession] = {}
        self._relay_ws: Any = None  # Current relay WS connection
        self._agent_server: AgentServer | None = None
        self._agent_spawner: AgentSpawner | None = None
        # channel_id -> (proc, command_id)
        self._terminal_procs: dict[str, tuple[asyncio.subprocess.Process, str]] = {}
        # Per-tab HTTP cookie jars for url_fetch.
        self._cookie_jars: dict[str, Any] = {}

        # Frame transport + dispatch.
        self._session_mgr = E2EESession(self)
        self._ctx = HandlerContext(self)
        self._v1 = V1Protocol(self)
        self._session_mgr.set_dispatch_table(self._build_dispatch_table())

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

    def _resolve_safe_path(self, base_dir: str, relative_path: str) -> str | None:
        """Resolve relative_path under base_dir. Returns None if it escapes."""
        base = Path(base_dir).resolve()
        target = (base / relative_path).resolve()
        if target != base and not str(target).startswith(str(base) + os.sep):
            return None
        return str(target)

    def _get_channel_cwd(self, channel_id: str) -> str:
        """Get working directory for a channel, falling back to process cwd."""
        if self._agent_server:
            ch = self._agent_server.store.get_channel(channel_id)
            if ch and ch.working_directory:
                return os.path.expanduser(ch.working_directory)
        return os.getcwd()

    def set_agent_server(self, server: AgentServer) -> None:
        """Register the agent server for forwarding chat messages."""
        self._agent_server = server

    def set_agent_spawner(self, spawner: AgentSpawner) -> None:
        """Register the agent spawner for starting/stopping agents."""
        self._agent_spawner = spawner

    # ------------------------------------------------------------------
    # Test-compat shims — wrappers around relay_handlers that preserve the
    # old private method names so tests in tests/test_*.py can keep poking
    # them directly. Keep these as instance methods so test reassignments
    # like `handler._send_frame = fake_send` continue to work.
    # ------------------------------------------------------------------

    async def _send_channel_list(self, session: ActiveSession, ws: Any) -> None:
        """Test-compat: delegate to relay_handlers.channels.handle_list_channels."""
        await channels.handle_list_channels(self._ctx, session, ws)

    async def _handle_upload_chunk(
        self, session: ActiveSession, payload: dict[str, Any], ws: Any,
    ) -> None:
        """Test-compat: delegate to relay_handlers.uploads.handle_upload_chunk."""
        await uploads.handle_upload_chunk(self._ctx, session, payload, ws)

    async def _handle_upload_complete(
        self, session: ActiveSession, payload: dict[str, Any], ws: Any,
    ) -> None:
        """Test-compat: delegate to relay_handlers.uploads.handle_upload_complete."""
        await uploads.handle_upload_complete(self._ctx, session, payload, ws)

    async def _handle_files_list(
        self, session: ActiveSession, payload: dict[str, Any], ws: Any,
    ) -> None:
        """Test-compat: delegate to relay_handlers.files.handle_files_list."""
        await files.handle_files_list(self._ctx, session, payload, ws)

    async def _handle_files_changes(
        self, session: ActiveSession, payload: dict[str, Any], ws: Any,
    ) -> None:
        """Test-compat: delegate to relay_handlers.files.handle_files_changes."""
        await files.handle_files_changes(self._ctx, session, payload, ws)

    async def _handle_files_commits(
        self, session: ActiveSession, payload: dict[str, Any], ws: Any,
    ) -> None:
        """Test-compat: delegate to relay_handlers.files.handle_files_commits."""
        await files.handle_files_commits(self._ctx, session, payload, ws)

    async def _handle_file_read(
        self, session: ActiveSession, payload: dict[str, Any], ws: Any,
    ) -> None:
        """Test-compat: delegate to relay_handlers.files.handle_file_read."""
        await files.handle_file_read(self._ctx, session, payload, ws)

    async def _handle_file_diff(
        self, session: ActiveSession, payload: dict[str, Any], ws: Any,
    ) -> None:
        """Test-compat: delegate to relay_handlers.files.handle_file_diff."""
        await files.handle_file_diff(self._ctx, session, payload, ws)

    # ------------------------------------------------------------------
    # Frame transport — delegated to E2EESession
    # ------------------------------------------------------------------

    async def handle_message(self, msg: dict[str, Any], ws: Any) -> None:
        """Dispatch an incoming WS message from the relay.

        Wired in from `ws.py` via `cli.py` as the `e2e_handler` callback.
        Delegates to `E2EESession.handle_message`.
        """
        await self._session_mgr.handle_message(msg, ws)

    async def broadcast_to_sessions(
        self,
        channel_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Broadcast an E2EE-encrypted payload to ALL active browser sessions.

        Called by `AgentServer` and `ComplicationRegistry` via callback.
        """
        await self._session_mgr.broadcast_to_sessions(channel_id, payload)

    def _build_dispatch_table(self) -> dict[str, Any]:
        """Action name → handler coroutine for the data-frame dispatcher.

        During the refactor each entry points either at a `relay_handlers.<ns>`
        function (already migrated) or at an old facade method (not yet
        migrated). The dispatcher in `E2EESession._handle_data_frame` calls
        these entries with (session, payload, ws). `message` and
        `retry_message` are special-cased in the dispatcher because they need
        the outer frame's `message_id` — those go to facade methods directly.
        """
        ctx = self._ctx
        return {
            # Channels
            proto.LIST_CHANNELS:       lambda s, p, ws: channels.handle_list_channels(ctx, s, ws),
            proto.CREATE_CHANNEL:      lambda s, p, ws: channels.handle_create_channel(ctx, s, p, ws),
            proto.RENAME_CHANNEL:      lambda s, p, ws: channels.handle_rename_channel(ctx, s, p, ws),
            proto.UPDATE_CHANNEL:      lambda s, p, ws: channels.handle_update_channel(ctx, s, p, ws),
            proto.DELETE_CHANNEL:      lambda s, p, ws: channels.handle_delete_channel(ctx, s, p, ws),

            # Messages
            proto.GET_MESSAGES:        lambda s, p, ws: messages.handle_get_messages(ctx, s, p, ws),
            proto.MARK_READ:           lambda s, p, ws: messages.handle_mark_read(ctx, s, p, ws),
            proto.MARK_SEEN:           lambda s, p, ws: messages.handle_mark_seen(ctx, s, p, ws),

            # Activity
            proto.GET_ACTIVITY:        lambda s, p, ws: activity.handle_get_activity(ctx, s, p, ws),

            # Agents
            proto.LIST_HARNESSES:      lambda s, p, ws: agents.handle_list_harnesses(ctx, s, ws),
            proto.LIST_WORKERS:        lambda s, p, ws: agents.handle_list_workers(ctx, s, ws),
            proto.START_AGENT:         lambda s, p, ws: agents.handle_start_agent(ctx, s, p, ws),
            proto.STOP_AGENT:          lambda s, p, ws: agents.handle_stop_agent(ctx, s, p, ws),
            proto.CANCEL:              lambda s, p, ws: agents.handle_cancel(ctx, s, p, ws),
            proto.RESTART_AGENT:       lambda s, p, ws: agents.handle_restart_agent(ctx, s, p, ws),

            # Interactions
            proto.INTERACTION_RESPONSE: lambda s, p, ws: interactions.handle_interaction_response(ctx, s, p, ws),

            # Sessions
            proto.RESET_SESSION:       lambda s, p, ws: sessions.handle_reset_session(ctx, s, p, ws),
            proto.COMPACT_SESSION:     lambda s, p, ws: sessions.handle_compact_session(ctx, s, p, ws),

            # Complications
            proto.GET_COMPLICATIONS:   lambda s, p, ws: complications_handlers.handle_get_complications(ctx, s, p, ws),
            proto.COMPLICATION_ACTION: lambda s, p, ws: complications_handlers.handle_complication_action(ctx, s, p, ws),

            # Terminal — terminal_exec is wrapped in a task by the dispatcher so
            # other actions (notably terminal_kill) can interrupt it.
            proto.TERMINAL_EXEC:       lambda s, p, ws: asyncio.create_task(
                terminal.handle_terminal_exec(ctx, s, p, ws)
            ),
            proto.TERMINAL_KILL:       lambda s, p, ws: terminal.handle_terminal_kill(ctx, s, p, ws),
            proto.TERMINAL_COMPLETE:   lambda s, p, ws: terminal.handle_terminal_complete(ctx, s, p, ws),

            # Files
            proto.FILES_LIST:          lambda s, p, ws: files.handle_files_list(ctx, s, p, ws),
            proto.FILES_CHANGES:       lambda s, p, ws: files.handle_files_changes(ctx, s, p, ws),
            proto.FILES_COMMITS:       lambda s, p, ws: files.handle_files_commits(ctx, s, p, ws),
            proto.FILE_READ:           lambda s, p, ws: files.handle_file_read(ctx, s, p, ws),
            proto.FILE_DIFF:           lambda s, p, ws: files.handle_file_diff(ctx, s, p, ws),

            # URL
            proto.URL_FETCH:           lambda s, p, ws: url.handle_url_fetch(ctx, s, p, ws),

            # Uploads
            proto.UPLOAD_CHUNK:        lambda s, p, ws: uploads.handle_upload_chunk(ctx, s, p, ws),
            proto.UPLOAD_COMPLETE:     lambda s, p, ws: uploads.handle_upload_complete(ctx, s, p, ws),
        }

    # ------------------------------------------------------------------
    # Frame-correlated handlers — invoked directly by E2EESession because
    # they read the outer frame.message_id, not just payload. The dispatch
    # table can't route them through the standard (s, p, ws) signature.
    # ------------------------------------------------------------------

    async def _handle_chat_message(
        self, session: ActiveSession, frame: dict[str, Any], ws: Any,
    ) -> None:
        """Forward to `relay_handlers.messages.handle_chat_message`."""
        await messages.handle_chat_message(self._ctx, session, frame, ws)

    async def _retry_message(
        self, session: ActiveSession, frame: dict[str, Any], ws: Any,
    ) -> None:
        """Forward to `relay_handlers.messages.handle_retry_message`."""
        await messages.handle_retry_message(self._ctx, session, frame, ws)

    async def _handle_v1_frame(
        self, session: ActiveSession, app: dict[str, Any], ws: Any,
    ) -> None:
        """Route a v1 application envelope alongside the legacy v0 API."""
        await self._v1.handle_frame(session, app, ws)

    async def _send_frame(
        self,
        session: ActiveSession,
        ws: Any,
        payload: dict[str, Any],
    ) -> None:
        """Encrypt and send a frame to the browser via the relay.

        Kept as an instance method on the facade so tests that do
        `handler._send_frame = fake_send` still intercept every frame.
        Delegates to the session manager.
        """
        await self._session_mgr.send_frame(session, ws, payload)
