"""Build Relay Protocol v1 adapter.

v1 is the browser-facing application protocol described in
RELAY_PROTOCOL_V1.md. The transport still decrypts to the existing
secure-transport frame shape, so v1 application envelopes currently ride in
that frame's `payload` field:

    {"frame_type": "data", "payload": {"v": 1, "kind": "request", ...}}

This module keeps the v1 routing separate from the v0 `payload.action`
dispatcher. Most methods are initially implemented by adapting to the existing
v0 handlers, then normalising their outputs into v1 response/stream/event
envelopes. That gives v1 clients the new envelope and error contract without
forking all domain behavior on day one.
"""

from __future__ import annotations

import logging
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from build_bridge import relay_protocol as v0
from build_bridge.harness_registry import get_harness
from build_bridge.relay_handlers import (
    activity,
    agents,
    channels,
    complications,
    files,
    interactions,
    messages,
    sessions,
    terminal,
    uploads,
    url,
)
from build_bridge.relay_handlers.context import HandlerContext

if TYPE_CHECKING:
    from build_bridge.e2ee import E2EEHandler
    from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)

V1_VERSION = 1
V1_KINDS_IN = frozenset({"request", "close"})
REVIEW_DENIAL_BEGIN = "<build-review-denied>"
REVIEW_DENIAL_END = "</build-review-denied>"
REVIEW_DENIAL_REASON_LIMIT = 4000
REVIEW_DENIAL_DIFF_LIMIT = 20000


class V1Error(Exception):
    """Protocol-level error with a v1 error code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def as_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


def is_v1_envelope(value: Any) -> bool:
    """Return True when *value* looks like a v1 application envelope."""
    return isinstance(value, dict) and value.get("v") == V1_VERSION and "kind" in value


def extract_v1_envelope(frame: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a v1 envelope from a decrypted secure-transport frame."""
    if is_v1_envelope(frame):
        return frame
    payload = frame.get("payload")
    if is_v1_envelope(payload):
        return payload
    return None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _target(
    facade: "E2EEHandler",
    app: dict[str, Any] | None = None,
    channel_id: str | None = None,
) -> dict[str, Any]:
    target_in = app.get("target", {}) if app else {}
    out = {"device_id": str(target_in.get("device_id") or getattr(facade.config, "device_id", ""))}
    ch = channel_id or target_in.get("channel_id")
    if ch:
        out["channel_id"] = ch
    return out


def _error(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "details": details or {},
        "retryable": retryable,
    }


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise V1Error("invalid_request", f"{field} is required", details={"field": field})
    return value


def _channel_id(app: dict[str, Any]) -> str:
    target = app.get("target")
    if not isinstance(target, dict):
        raise V1Error("invalid_frame", "target must be an object", details={"field": "target"})
    return _require_str(target.get("channel_id"), "target.channel_id")


def _payload(app: dict[str, Any]) -> dict[str, Any]:
    payload = app.get("payload", {})
    if not isinstance(payload, dict):
        raise V1Error("invalid_frame", "payload must be an object", details={"field": "payload"})
    return payload


def _validate_app(app: dict[str, Any]) -> None:
    if app.get("v") != V1_VERSION:
        raise V1Error("invalid_frame", "unsupported protocol version", details={"field": "v"})
    if app.get("kind") not in V1_KINDS_IN:
        raise V1Error("invalid_frame", "unsupported frame kind", details={"field": "kind"})
    _require_str(app.get("id"), "id")
    if app.get("kind") == "close":
        return
    _require_str(app.get("type"), "type")
    target = app.get("target")
    if not isinstance(target, dict):
        raise V1Error("invalid_frame", "target must be an object", details={"field": "target"})
    _require_str(target.get("device_id"), "target.device_id")
    payload = app.get("payload", {})
    if not isinstance(payload, dict):
        raise V1Error("invalid_frame", "payload must be an object", details={"field": "payload"})
    meta = app.get("meta", {})
    if not isinstance(meta, dict):
        raise V1Error("invalid_frame", "meta must be an object", details={"field": "meta"})


class V1RequestContext:
    """Per-request context passed to v0 handlers during adaptation."""

    def __init__(self, protocol: "V1Protocol", app: dict[str, Any]) -> None:
        self.protocol = protocol
        self.app = app
        self.base = HandlerContext(protocol.facade)
        self.completed = False
        self.stream_seq = 0

    @property
    def facade(self) -> "E2EEHandler":
        return self.protocol.facade

    @property
    def config(self):
        return self.base.config

    @property
    def e2ee_store(self):
        return self.base.e2ee_store

    @property
    def agent_server(self):
        return self.base.agent_server

    @property
    def agent_spawner(self):
        return self.base.agent_spawner

    @property
    def terminal_procs(self):
        return self.base.terminal_procs

    @property
    def cookie_jars(self):
        return self.base.cookie_jars

    def resolve_safe_path(self, base_dir: str, relative_path: str) -> str | None:
        return self.base.resolve_safe_path(base_dir, relative_path)

    def get_channel_cwd(self, channel_id: str) -> str:
        return self.base.get_channel_cwd(channel_id)

    def get_agent_name(self, channel_id: str) -> str:
        return self.base.get_agent_name(channel_id)

    async def send_frame(self, session: "ActiveSession", ws: Any, payload: dict[str, Any]) -> None:
        await self.protocol._handle_v0_output(self, session, ws, payload)

    async def broadcast(self, channel_id: str, payload: dict[str, Any]) -> None:
        await self.protocol.facade.broadcast_to_sessions(channel_id, payload)

    async def send_system_message(
        self,
        session: "ActiveSession",
        ws: Any,
        channel_id: str,
        text: str,
    ) -> None:
        await self.protocol._send_event(
            session,
            ws,
            "message.created",
            {"message": {"sender": "system", "content": text}},
            target=_target(self.facade, self.app, channel_id),
        )

    async def send_error_message(
        self,
        session: "ActiveSession",
        ws: Any,
        channel_id: str,
        error_text: str,
    ) -> None:
        await self.protocol._send_event(
            session,
            ws,
            "message.created",
            {
                "message": {
                    "sender": self.get_agent_name(channel_id),
                    "content": f"⚠ {error_text}",
                }
            },
            target=_target(self.facade, self.app, channel_id),
        )


V0Call = tuple[str, dict[str, Any]]


class V1Protocol:
    """v1 request router and v0 compatibility adapter."""

    def __init__(self, facade: "E2EEHandler") -> None:
        self.facade = facade

    async def handle_frame(
        self,
        session: "ActiveSession",
        app: dict[str, Any],
        ws: Any,
    ) -> None:
        """Validate and route one v1 application envelope."""
        session.protocol_version = V1_VERSION
        try:
            _validate_app(app)
            if app.get("kind") == "close":
                self.facade._sessions.pop(session.session_id, None)
                return
            await self._dispatch(session, app, ws)
        except V1Error as exc:
            await self._send_response(
                session, ws, app,
                payload={},
                error=exc.as_payload(),
            )
        except Exception as exc:  # noqa: BLE001 - response boundary.
            log.exception("v1 handler failed")
            await self._send_response(
                session, ws, app,
                payload={},
                error=_error("internal", str(exc)),
            )

    async def _dispatch(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        method = app["type"]
        if method == "protocol.hello":
            await self._send_response(
                session, ws, app,
                payload={
                    "version": V1_VERSION,
                    "features": [
                        "streaming",
                        "uploads.v2",
                        "projects.v1",
                        "project.create",
                        "project.repo.list",
                        "plans.v1",
                        "worktree.create",
                        "worktree.snapshot",
                        "review.denied",
                        "dashboard.snapshot",
                    ],
                    "limits": {"max_encrypted_frame_bytes": 262144},
                },
            )
            return
        if method == "request.cancel":
            await self._send_response(session, ws, app, payload={"cancelled": False})
            return

        direct_handler = getattr(self, f"_handle_{method.replace('.', '_')}", None)
        if direct_handler:
            await direct_handler(session, app, ws)
            return

        calls = self._v1_to_v0_calls(app)
        if not calls:
            raise V1Error("unknown_method", f"unknown method: {method}")

        ctx = V1RequestContext(self, app)
        for action, payload in calls:
            await self._call_v0(ctx, session, ws, action, payload)

        # Some v0 operations are fire-and-forget. v1 still completes them.
        if not ctx.completed and method in {
            "message.mark_read",
            "message.mark_seen",
            "interaction.respond",
            "complication.invoke",
        }:
            await self._send_response(session, ws, app, payload={"accepted": True})
            ctx.completed = True

    def _v1_to_v0_calls(self, app: dict[str, Any]) -> list[V0Call]:
        method = app["type"]
        payload = _payload(app)
        calls: list[V0Call] = []

        if method == "channel.list":
            return [(v0.LIST_CHANNELS, {"action": v0.LIST_CHANNELS})]
        if method == "channel.create":
            agent = payload.get("agent") or {}
            return [(
                v0.CREATE_CHANNEL,
                {
                    "action": v0.CREATE_CHANNEL,
                    "name": payload.get("name", ""),
                    "harness": agent.get("harness", ""),
                    "model": agent.get("model", ""),
                    "effort": agent.get("effort", ""),
                    "system_prompt": agent.get("system_prompt", ""),
                    "working_directory": agent.get("working_directory", ""),
                    "auto_approve_tools": bool(agent.get("auto_approve_tools", False)),
                },
            )]
        if method == "channel.update":
            channel_id = _channel_id(app)
            if "name" in payload:
                calls.append((
                    v0.RENAME_CHANNEL,
                    {"action": v0.RENAME_CHANNEL, "channel_id": channel_id, "name": payload.get("name", "")},
                ))
            agent = payload.get("agent") or {}
            if agent:
                calls.append((
                    v0.UPDATE_CHANNEL,
                    {
                        "action": v0.UPDATE_CHANNEL,
                        "channel_id": channel_id,
                        "working_directory": agent.get("working_directory"),
                        "harness": agent.get("harness"),
                        "model": agent.get("model"),
                        "effort": agent.get("effort"),
                        "auto_approve_tools": agent.get("auto_approve_tools"),
                    },
                ))
            return calls
        if method == "channel.delete":
            channel_id = _channel_id(app)
            return [(v0.DELETE_CHANNEL, {"action": v0.DELETE_CHANNEL, "channel_id": channel_id})]

        if method == "message.list":
            channel_id = _channel_id(app)
            return [(
                v0.GET_MESSAGES,
                {"action": v0.GET_MESSAGES, "channel_id": channel_id, "limit": payload.get("limit", 50), "before": payload.get("before")},
            )]
        if method == "message.send":
            channel_id = _channel_id(app)
            opts = payload.get("agent_options") or {}
            return [(
                v0.MESSAGE,
                {
                    "action": v0.MESSAGE,
                    "channel_id": channel_id,
                    "content": payload.get("content", ""),
                    "attachments": [
                        {**att, "file_id": att.get("file_id") or att.get("upload_id")}
                        for att in payload.get("attachments", [])
                    ],
                    "model": opts.get("model"),
                    "effort": opts.get("effort"),
                    "plan_mode": opts.get("plan_mode"),
                },
            )]
        if method == "message.mark_read":
            return [(
                v0.MARK_READ,
                {"action": v0.MARK_READ, "channel_id": _channel_id(app), "message_ids": payload.get("message_ids", [])},
            )]
        if method == "message.mark_seen":
            return [(v0.MARK_SEEN, {"action": v0.MARK_SEEN, "channel_id": _channel_id(app)})]

        if method == "activity.list":
            return [(v0.GET_ACTIVITY, {"action": v0.GET_ACTIVITY, "channel_id": _channel_id(app)})]

        if method == "agent.harnesses":
            return [(v0.LIST_HARNESSES, {"action": v0.LIST_HARNESSES})]
        if method == "agent.workers":
            return [(v0.LIST_WORKERS, {"action": v0.LIST_WORKERS})]
        if method == "agent.start":
            channel_id = _channel_id(app)
            return [(
                v0.START_AGENT,
                {
                    "action": v0.START_AGENT,
                    "channel_id": channel_id,
                    "harness": payload.get("harness", ""),
                    "model": payload.get("model", ""),
                    "effort": payload.get("effort", ""),
                    "system_prompt": payload.get("system_prompt", ""),
                    "working_directory": payload.get("working_directory", ""),
                },
            )]
        if method == "agent.stop":
            return [(v0.STOP_AGENT, {"action": v0.STOP_AGENT, "channel_id": _channel_id(app)})]
        if method == "agent.cancel_turn":
            return [(v0.CANCEL, {"action": v0.CANCEL, "channel_id": _channel_id(app)})]
        if method == "agent.restart":
            return [(v0.RESTART_AGENT, {"action": v0.RESTART_AGENT, "channel_id": _channel_id(app)})]

        if method == "interaction.respond":
            channel_id = _channel_id(app)
            resp = payload.get("response") or {}
            return [(
                v0.INTERACTION_RESPONSE,
                {
                    "action": v0.INTERACTION_RESPONSE,
                    "channel_id": channel_id,
                    "interaction_id": payload.get("interaction_id", ""),
                    "selected_option": resp.get("selected_option"),
                    "selected_options": resp.get("selected_options"),
                    "freeform_response": resp.get("freeform"),
                    "step_answers": resp.get("step_answers"),
                },
            )]

        if method == "session.reset_agent_context":
            return [(v0.RESET_SESSION, {"action": v0.RESET_SESSION, "channel_id": _channel_id(app)})]
        if method == "session.compact_agent_context":
            return [(v0.COMPACT_SESSION, {"action": v0.COMPACT_SESSION, "channel_id": _channel_id(app)})]

        if method == "complication.list":
            return [(v0.GET_COMPLICATIONS, {"action": v0.GET_COMPLICATIONS, "channel_id": _channel_id(app)})]
        if method == "complication.invoke":
            return [(
                v0.COMPLICATION_ACTION,
                {
                    "action": v0.COMPLICATION_ACTION,
                    "channel_id": _channel_id(app),
                    "complication_id": payload.get("complication_id", ""),
                    "option_id": payload.get("option_id", ""),
                },
            )]

        if method == "terminal.exec":
            return [(
                v0.TERMINAL_EXEC,
                {
                    "action": v0.TERMINAL_EXEC,
                    "channel_id": _channel_id(app),
                    "command": payload.get("command", ""),
                    "cwd": payload.get("cwd", ""),
                    "command_id": app["id"],
                },
            )]
        if method == "terminal.kill":
            return [(
                v0.TERMINAL_KILL,
                {
                    "action": v0.TERMINAL_KILL,
                    "channel_id": _channel_id(app),
                    "command_id": payload.get("request_id", ""),
                },
            )]
        if method == "terminal.complete":
            line = payload.get("line", "")
            cursor = payload.get("cursor")
            partial = payload.get("partial")
            if partial is None:
                prefix = line[:cursor] if isinstance(cursor, int) else line
                partial = prefix.split()[-1] if prefix.split() else ""
            return [(
                v0.TERMINAL_COMPLETE,
                {
                    "action": v0.TERMINAL_COMPLETE,
                    "channel_id": _channel_id(app),
                    "line": line,
                    "partial": partial,
                    "cwd": payload.get("cwd", ""),
                },
            )]

        if method == "file.tree":
            return [(
                v0.FILES_LIST,
                {"action": v0.FILES_LIST, "channel_id": _channel_id(app), "path": payload.get("path", "")},
            )]
        if method == "file.changes":
            return [(
                v0.FILES_CHANGES,
                {
                    "action": v0.FILES_CHANGES,
                    "channel_id": _channel_id(app),
                    "repo_path": payload.get("repo_path", ""),
                    "older_ref": payload.get("base_ref"),
                    "newer_ref": payload.get("head_ref"),
                },
            )]
        if method == "file.commits":
            return [(
                v0.FILES_COMMITS,
                {
                    "action": v0.FILES_COMMITS,
                    "channel_id": _channel_id(app),
                    "repo_path": payload.get("repo_path", ""),
                    "limit": payload.get("limit"),
                },
            )]
        if method == "file.read":
            return [(
                v0.FILE_READ,
                {
                    "action": v0.FILE_READ,
                    "channel_id": _channel_id(app),
                    "path": payload.get("path", ""),
                    "offset": payload.get("offset", 0),
                    "limit": payload.get("limit"),
                },
            )]
        if method == "file.diff":
            return [(
                v0.FILE_DIFF,
                {
                    "action": v0.FILE_DIFF,
                    "channel_id": _channel_id(app),
                    "path": payload.get("path", ""),
                    "repo_path": payload.get("repo_path"),
                    "staged": payload.get("staged"),
                    "older_ref": payload.get("base_ref"),
                    "newer_ref": payload.get("head_ref"),
                },
            )]

        if method == "url.fetch":
            headers = payload.get("headers") or {}
            return [(
                v0.URL_FETCH,
                {
                    "action": v0.URL_FETCH,
                    "url": payload.get("url", ""),
                    "method": payload.get("method", "GET"),
                    "body": payload.get("body"),
                    "content_type": headers.get("content-type") or headers.get("Content-Type"),
                    "tab_id": payload.get("tab_id", ""),
                    "request_id": app["id"],
                },
            )]

        return []

    async def _call_v0(
        self,
        ctx: V1RequestContext,
        session: "ActiveSession",
        ws: Any,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        validator = v0.VALIDATORS.get(action)
        if validator:
            valid, err = validator(payload)
            if not valid:
                raise V1Error("invalid_request", err)

        if action == v0.MESSAGE:
            frame = {"message_id": ctx.app.get("payload", {}).get("message_id") or ctx.app["id"], "payload": payload}
            await messages.handle_chat_message(ctx, session, frame, ws)
            return
        if action == v0.RETRY_MESSAGE:
            frame = {"payload": payload}
            await messages.handle_retry_message(ctx, session, frame, ws)
            return

        table: dict[str, Callable[[V1RequestContext, "ActiveSession", dict[str, Any], Any], Any]] = {
            v0.LIST_CHANNELS: lambda c, s, p, w: channels.handle_list_channels(c, s, w),
            v0.CREATE_CHANNEL: channels.handle_create_channel,
            v0.RENAME_CHANNEL: channels.handle_rename_channel,
            v0.UPDATE_CHANNEL: channels.handle_update_channel,
            v0.DELETE_CHANNEL: channels.handle_delete_channel,
            v0.GET_MESSAGES: messages.handle_get_messages,
            v0.MARK_READ: messages.handle_mark_read,
            v0.MARK_SEEN: messages.handle_mark_seen,
            v0.GET_ACTIVITY: activity.handle_get_activity,
            v0.LIST_HARNESSES: lambda c, s, p, w: agents.handle_list_harnesses(c, s, w),
            v0.LIST_WORKERS: lambda c, s, p, w: agents.handle_list_workers(c, s, w),
            v0.START_AGENT: agents.handle_start_agent,
            v0.STOP_AGENT: agents.handle_stop_agent,
            v0.CANCEL: agents.handle_cancel,
            v0.RESTART_AGENT: agents.handle_restart_agent,
            v0.INTERACTION_RESPONSE: interactions.handle_interaction_response,
            v0.RESET_SESSION: sessions.handle_reset_session,
            v0.COMPACT_SESSION: sessions.handle_compact_session,
            v0.GET_COMPLICATIONS: complications.handle_get_complications,
            v0.COMPLICATION_ACTION: complications.handle_complication_action,
            v0.TERMINAL_EXEC: terminal.handle_terminal_exec,
            v0.TERMINAL_KILL: terminal.handle_terminal_kill,
            v0.TERMINAL_COMPLETE: terminal.handle_terminal_complete,
            v0.FILES_LIST: files.handle_files_list,
            v0.FILES_CHANGES: files.handle_files_changes,
            v0.FILES_COMMITS: files.handle_files_commits,
            v0.FILE_READ: files.handle_file_read,
            v0.FILE_DIFF: files.handle_file_diff,
            v0.URL_FETCH: url.handle_url_fetch,
            v0.UPLOAD_CHUNK: uploads.handle_upload_chunk,
            v0.UPLOAD_COMPLETE: uploads.handle_upload_complete,
        }
        if action == v0.TERMINAL_EXEC:
            asyncio.create_task(terminal.handle_terminal_exec(ctx, session, payload, ws))
            return

        handler = table.get(action)
        if not handler:
            raise V1Error("unknown_method", f"unknown adapted action: {action}")
        await handler(ctx, session, payload, ws)

    async def _handle_dashboard_snapshot(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        await self._send_response(
            session,
            ws,
            app,
            payload=_dashboard_snapshot(self.facade),
        )

    async def _handle_project_list(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        _ensure_project_graph(self.facade)
        await self._send_response(
            session,
            ws,
            app,
            payload={
                "projects": [
                    _project_primitive_payload(project, self.facade.store.list_worktrees(project.id))
                    for project in self.facade.store.list_projects()
                ],
            },
        )

    async def _handle_project_create(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        payload = _payload(app)
        name = _clean_text(payload.get("name"))
        raw_root = (
            _clean_text(payload.get("root_path"))
            or _clean_text(payload.get("directory"))
            or _clean_text(payload.get("path"))
        )
        if not name:
            raise V1Error("invalid_request", "payload.name is required", details={"field": "payload.name"})
        if not raw_root:
            raise V1Error("invalid_request", "payload.root_path is required", details={"field": "payload.root_path"})

        root = Path(os.path.expanduser(os.path.expandvars(raw_root))).resolve()
        if not root.is_dir():
            raise V1Error(
                "invalid_request",
                "payload.root_path must be an existing directory",
                details={"field": "payload.root_path", "root_path": str(root)},
            )

        repo_root, branch = _git_workspace(str(root))
        project = self.facade.store.upsert_project(
            _workspace_project_id(str(root), name),
            name,
            root_path=str(root),
            repo=(repo_root.name if repo_root and repo_root == root else root.name) or name,
            default_branch=(branch if repo_root and repo_root == root else "") or "main",
            color=_color_for(str(root)),
        )
        await self._send_response(
            session,
            ws,
            app,
            payload={"project": _project_primitive_payload(project, self.facade.store.list_worktrees(project.id))},
        )

    async def _handle_project_repo_list(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        _ensure_project_graph(self.facade)
        payload = _payload(app)
        project_id = _require_str(payload.get("project_id"), "payload.project_id")
        project = self.facade.store.get_project(project_id)
        if not project:
            raise V1Error("not_found", "project not found", details={"project_id": project_id})

        repos, error = await _project_repo_list(project)
        await self._send_response(
            session,
            ws,
            app,
            payload={
                "project_id": project.id,
                "root_path": project.root_path,
                "repos": repos,
                "error": error,
            },
        )

    async def _handle_worktree_list(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        _ensure_project_graph(self.facade)
        payload = _payload(app)
        project_id = payload.get("project_id")
        worktrees = self.facade.store.list_worktrees(project_id if isinstance(project_id, str) else None)
        await self._send_response(
            session,
            ws,
            app,
            payload={"worktrees": [_worktree_primitive_payload(worktree) for worktree in worktrees]},
        )

    async def _handle_worktree_create(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        _ensure_project_graph(self.facade)
        payload = _payload(app)
        project_id = _require_str(payload.get("project_id"), "payload.project_id")
        project = self.facade.store.get_project(project_id)
        if not project:
            raise V1Error("not_found", "project not found", details={"project_id": project_id})

        name = _clean_text(payload.get("name")) or "New worktree"
        worktree_id = _clean_text(payload.get("worktree_id")) or _new_id("wt")
        channel_id = _clean_text(payload.get("channel_id")) or str(uuid.uuid4())
        branch = _clean_text(payload.get("branch")) or f"agents/{_slug(name)}-{uuid.uuid4().hex[:6]}"
        repo_root = _resolve_project_repo(project, payload.get("repo_path") or payload.get("repository_path"))
        _, repo_branch = _git_workspace(str(repo_root)) if repo_root else (None, "")
        base_ref = _clean_text(payload.get("base_ref")) or repo_branch or project.default_branch or "HEAD"
        path = _clean_text(payload.get("path"))
        git_created = False
        git_error = None

        if payload.get("create_git_worktree") is True:
            created_path, git_error = await _create_git_worktree(project, repo_root, path, branch, base_ref)
            if git_error:
                raise V1Error(
                    "failed_precondition",
                    "git worktree creation failed",
                    details={
                        "repo_path": str(repo_root) if repo_root else project.root_path,
                        "branch": branch,
                        "base_ref": base_ref,
                        "error": git_error,
                    },
                )
            git_created = True
            path = created_path
        if not path:
            path = str(repo_root) if repo_root else project.root_path

        channel = self.facade.store.create_channel(channel_id, name)
        worktree = self.facade.store.upsert_worktree(
            worktree_id,
            project.id,
            name,
            path=path,
            branch=branch,
            status="idle",
            channel_id=channel.id,
            base_ref=base_ref,
            head_ref=branch,
        )
        agent_payload = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}
        agent, agent_error = await _attach_worktree_agent(self.facade, channel.id, agent_payload, path)
        plan = self.facade.store.upsert_plan(
            _new_id("plan"),
            project.id,
            name,
            worktree_id=worktree.id,
            channel_id=channel.id,
            status="draft",
            step_count=1,
            done_step_count=0,
            model=str((agent or {}).get("model") or agent_payload.get("model") or ""),
        )

        await self._send_response(
            session,
            ws,
            app,
            payload={
                "project": _project_primitive_payload(project, self.facade.store.list_worktrees(project.id)),
                "worktree": _worktree_primitive_payload(worktree),
                "plan": _plan_primitive_payload(plan),
                "channel": {"id": channel.id, "name": channel.name, "created_at": channel.created_at},
                "agent": agent,
                "agent_error": agent_error,
                "git": {"created": git_created, "error": git_error},
            },
            target=_target(self.facade, app, channel.id),
        )

    async def _handle_worktree_snapshot(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        _ensure_project_graph(self.facade)
        payload = _payload(app)
        worktree_id = _require_str(payload.get("worktree_id"), "payload.worktree_id")
        worktree = self.facade.store.get_worktree(worktree_id)
        if not worktree:
            raise V1Error("not_found", "worktree not found", details={"worktree_id": worktree_id})
        await self._send_response(
            session,
            ws,
            app,
            payload=await _worktree_snapshot(self.facade, worktree),
        )

    async def _handle_review_denied(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        channel_id = _channel_id(app)
        payload = _payload(app)
        file_path = _clean_text(payload.get("file") or payload.get("path"))
        reason = _clean_text(payload.get("reason"))
        diff = payload.get("diff") if isinstance(payload.get("diff"), str) else ""
        repo_path = _clean_text(payload.get("repo_path"))
        if not file_path:
            raise V1Error("invalid_request", "payload.file is required", details={"field": "payload.file"})
        if not reason:
            raise V1Error("invalid_request", "payload.reason is required", details={"field": "payload.reason"})

        ctx = V1RequestContext(self, app)
        await self._call_v0(
            ctx,
            session,
            ws,
            v0.MESSAGE,
            {
                "action": v0.MESSAGE,
                "channel_id": channel_id,
                "content": _review_denial_content(file_path, reason, diff, repo_path),
                "attachments": [],
                "model": None,
                "effort": None,
                "plan_mode": None,
            },
        )

    async def _handle_plan_list(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        _ensure_project_graph(self.facade)
        payload = _payload(app)
        project_id = payload.get("project_id")
        worktree_id = payload.get("worktree_id")
        plans = self.facade.store.list_plans(
            project_id if isinstance(project_id, str) else None,
            worktree_id if isinstance(worktree_id, str) else None,
        )
        await self._send_response(
            session,
            ws,
            app,
            payload={"plans": [_plan_primitive_payload(plan) for plan in plans]},
        )

    async def _handle_upload_create(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        payload = _payload(app)
        upload_id = payload.get("upload_id") or _new_id("upl")
        channel_id = _channel_id(app)
        filename = _require_str(payload.get("filename"), "payload.filename")
        size = payload.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise V1Error("invalid_request", "payload.size must be a non-negative integer", details={"field": "payload.size"})
        if size > self.facade._MAX_FILE_SIZE:
            raise V1Error(
                "quota_exceeded",
                f"file too large (max {self.facade._MAX_FILE_SIZE // (1024 * 1024)} MB)",
                details={"field": "payload.size"},
            )

        destination = payload.get("destination") or {"kind": "scratch"}
        if not isinstance(destination, dict):
            raise V1Error("invalid_request", "payload.destination must be an object", details={"field": "payload.destination"})
        kind = destination.get("kind", "scratch")
        if kind not in {"scratch", "workspace"}:
            raise V1Error("invalid_request", "payload.destination.kind is invalid", details={"field": "payload.destination.kind"})

        chunk_size = 64 * 1024
        total_chunks = max(1, (size + chunk_size - 1) // chunk_size)
        tmp_dir = uploads.upload_tmp_dir(self.facade, upload_id)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "filename": uploads.sanitize_filename(filename),
            "mime_type": payload.get("mime_type", "application/octet-stream"),
            "total_size": size,
            "total_chunks": total_chunks,
            "channel_id": channel_id,
            "dest_dir": destination.get("path", "") if kind == "workspace" else "",
            "sha256": payload.get("sha256", ""),
        }
        (tmp_dir / "meta.json").write_text(json.dumps(meta))

        await self._send_response(
            session, ws, app,
            payload={"upload_id": upload_id, "chunk_size": chunk_size, "accepted": True},
        )

    async def _handle_upload_write_chunk(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        payload = _payload(app)
        upload_id = _require_str(payload.get("upload_id"), "payload.upload_id")
        index = payload.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise V1Error("invalid_request", "payload.index must be an integer", details={"field": "payload.index"})
        data_b64 = _require_str(payload.get("data"), "payload.data")

        tmp_dir = uploads.upload_tmp_dir(self.facade, upload_id)
        meta_path = tmp_dir / "meta.json"
        if not meta_path.exists():
            raise V1Error("not_found", "upload resource not found", details={"upload_id": upload_id})

        try:
            chunk_data = base64.b64decode(data_b64 + "=" * (-len(data_b64) % 4))
        except Exception as exc:
            raise V1Error("invalid_request", f"payload.data is not valid base64: {exc}", details={"field": "payload.data"}) from exc

        (tmp_dir / f"chunk_{index}").write_bytes(chunk_data)
        await self._send_response(
            session, ws, app,
            payload={"upload_id": upload_id, "index": index, "received": True},
        )

    async def _handle_upload_complete(self, session: "ActiveSession", app: dict[str, Any], ws: Any) -> None:
        payload = _payload(app)
        upload_id = _require_str(payload.get("upload_id"), "payload.upload_id")
        tmp_dir = uploads.upload_tmp_dir(self.facade, upload_id)
        meta_path = tmp_dir / "meta.json"
        if not meta_path.exists():
            raise V1Error("not_found", "upload resource not found", details={"upload_id": upload_id})
        meta = json.loads(meta_path.read_text())

        ctx = V1RequestContext(self, app)
        await self._call_v0(
            ctx,
            session,
            ws,
            v0.UPLOAD_COMPLETE,
            {
                "action": v0.UPLOAD_COMPLETE,
                "file_id": upload_id,
                "channel_id": meta.get("channel_id") or _channel_id(app),
                "sha256": meta.get("sha256", ""),
            },
        )

    async def _handle_v0_output(
        self,
        ctx: V1RequestContext,
        session: "ActiveSession",
        ws: Any,
        payload: dict[str, Any],
    ) -> None:
        action = payload.get("action", "")
        target = _target(self.facade, ctx.app, payload.get("channel_id"))

        if action == v0.ERROR:
            await self._send_response(
                session, ws, ctx.app,
                payload={},
                error=_error("invalid_request", str(payload.get("error", "request failed"))),
                target=target,
            )
            ctx.completed = True
            return

        if action == v0.UPLOAD_ERROR:
            await self._send_response(
                session, ws, ctx.app,
                payload={},
                error=_error("invalid_request", str(payload.get("error", "upload failed"))),
                target=target,
            )
            ctx.completed = True
            return

        if payload.get("error"):
            await self._send_response(
                session, ws, ctx.app,
                payload={k: v for k, v in payload.items() if k not in ("action", "error")},
                error=_error("invalid_request", str(payload["error"])),
                target=target,
            )
            ctx.completed = True
            return

        if action == v0.TERMINAL_OUTPUT:
            if payload.get("done"):
                await self._send_response(
                    session, ws, ctx.app,
                    payload={
                        "exit_code": payload.get("exit_code"),
                        "cwd": payload.get("cwd", ""),
                        "data": payload.get("data", ""),
                    },
                    target=target,
                )
                ctx.completed = True
            else:
                await self._send_stream(
                    session, ws, ctx,
                    "terminal.exec.output",
                    {"stdout": payload.get("data", ""), "cwd": payload.get("cwd")},
                    target=target,
                )
            return

        if action == v0.FILE_READ_RESULT and payload.get("chunk_total", 0):
            await self._send_stream(
                session, ws, ctx,
                "file.read.chunk",
                {
                    "offset": payload.get("offset", 0),
                    "data": payload.get("content", ""),
                    "encoding": payload.get("encoding"),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_total": payload.get("chunk_total"),
                },
                target=target,
            )
            if payload.get("chunk_index") == payload.get("chunk_total", 1) - 1:
                await self._send_response(
                    session, ws, ctx.app,
                    payload={k: v for k, v in payload.items() if k not in ("action", "content")},
                    target=target,
                )
                ctx.completed = True
            return

        if action == v0.COMPACT_STARTED:
            await self._send_event(
                session, ws,
                "session.compaction_started",
                {k: v for k, v in payload.items() if k != "action"},
                target=target,
            )
            return

        if action in {v0.SYSTEM_MESSAGE, v0.MESSAGE, v0.AGENT_EVENT, v0.COMPLICATION_UPDATE, v0.COMPLICATION_REMOVE}:
            await self._send_v1_payload(session, ws, translate_v0_broadcast(self.facade, payload))
            return

        if ctx.completed:
            await self._send_event(
                session, ws,
                self._event_type_for_extra_payload(action),
                {k: v for k, v in payload.items() if k != "action"},
                target=target,
            )
            return

        await self._send_response(
            session, ws, ctx.app,
            payload=self._normalise_response_payload(action, payload),
            target=target,
        )
        ctx.completed = True

    def _normalise_response_payload(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in payload.items() if k != "action"}
        if action == v0.CHANNEL_LIST:
            return {
                "channels": [
                    _normalise_channel(ch)
                    for ch in payload.get("channels", [])
                ],
                "device": {"cwd": payload.get("agent_cwd", "")},
            }
        if action == v0.DELIVERED:
            return {
                "message": {"id": payload.get("message_id"), "channel_id": payload.get("channel_id")},
                "agent_delivery": "accepted",
            }
        if action == v0.READ:
            return {"read": payload.get("message_ids", [])}
        if action == v0.UPLOAD_ACCEPTED:
            sha256 = payload.get("sha256") or _sha256_file(payload.get("path"))
            return {
                "upload": {
                    "id": payload.get("file_id"),
                    "path": payload.get("path", ""),
                    "sha256": sha256,
                    "filename": payload.get("filename"),
                    "size": payload.get("size"),
                }
            }
        if action == v0.CHUNK_ACK:
            return {"upload_id": payload.get("file_id"), "index": payload.get("chunk_index"), "received": True}
        return out

    def _event_type_for_extra_payload(self, action: str) -> str:
        return {
            v0.FILES_LIST_RESULT: "file.tree_changed",
            v0.COMPLICATION_UPDATE: "complication.updated",
            v0.COMPLICATION_REMOVE: "complication.removed",
        }.get(action, "protocol.extra")

    async def _send_response(
        self,
        session: "ActiveSession",
        ws: Any,
        app: dict[str, Any],
        *,
        payload: dict[str, Any],
        error: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
    ) -> None:
        await self._send_v1_payload(
            session, ws,
            {
                "v": V1_VERSION,
                "kind": "response",
                "id": _new_id("res"),
                "ref": app.get("id"),
                "type": app.get("type", "protocol.error"),
                "target": target or _target(self.facade, app),
                "payload": payload if error is None else {},
                "error": error,
                "meta": _response_meta(app),
            },
        )

    async def _send_stream(
        self,
        session: "ActiveSession",
        ws: Any,
        ctx: V1RequestContext,
        stream_type: str,
        payload: dict[str, Any],
        *,
        target: dict[str, Any],
    ) -> None:
        await self._send_v1_payload(
            session, ws,
            {
                "v": V1_VERSION,
                "kind": "stream",
                "id": _new_id("str"),
                "ref": ctx.app.get("id"),
                "type": stream_type,
                "target": target,
                "payload": payload,
                "error": None,
                "meta": {**_response_meta(ctx.app), "seq": ctx.stream_seq},
            },
        )
        ctx.stream_seq += 1

    async def _send_event(
        self,
        session: "ActiveSession",
        ws: Any,
        event_type: str,
        payload: dict[str, Any],
        *,
        target: dict[str, Any],
    ) -> None:
        await self._send_v1_payload(
            session, ws,
            {
                "v": V1_VERSION,
                "kind": "event",
                "id": _new_id("evt"),
                "type": event_type,
                "target": target,
                "payload": payload,
                "error": None,
                "meta": {},
            },
        )

    async def _send_v1_payload(self, session: "ActiveSession", ws: Any, payload: dict[str, Any]) -> None:
        await self.facade._send_frame(session, ws, payload)


_DASHBOARD_COLORS = (
    "#6366f1",
    "#0891b2",
    "#9333ea",
    "#0f766e",
    "#e11d48",
    "#d97706",
)
_PROJECT_REPO_SCAN_DEPTH = 3
_PROJECT_REPO_SCAN_LIMIT = 64
_PROJECT_REPO_SCAN_SKIP = frozenset({
    ".cache",
    ".codex",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
})


def _dashboard_snapshot(facade: "E2EEHandler") -> dict[str, Any]:
    _ensure_project_graph(facade)
    agent_server = getattr(facade, "_agent_server", None)
    agent_spawner = getattr(facade, "_agent_spawner", None)
    device_id = str(getattr(facade.config, "device_id", ""))
    channel_by_id = {channel.id: channel for channel in facade.store.list_channels()}
    out_projects: list[dict[str, Any]] = []

    for project_record in facade.store.list_projects():
        project = {
            "id": project_record.id,
            "name": project_record.name,
            "description": "Repository workspace" if project_record.root_path else "Local workspace",
            "repo": project_record.repo or project_record.name,
            "branch": project_record.default_branch or "main",
            "color": project_record.color or _color_for(project_record.id),
            "needsYou": 0,
            "runningAgents": 0,
            "queued": 0,
            "lastActive": _relative_time(project_record.updated_at),
            "root_path": project_record.root_path,
            "worktrees": [],
            "plans": [],
            "activity": [],
            "_last_active_ts": project_record.updated_at,
        }
        worktrees = facade.store.list_worktrees(project_record.id)
        plans = facade.store.list_plans(project_record.id)
        plans_by_worktree: dict[str, list[Any]] = {}
        worktree_status: dict[str, str] = {}
        worktree_updated_at: dict[str, float] = {}
        model_by_channel: dict[str, str] = {}
        for plan in plans:
            if plan.worktree_id:
                plans_by_worktree.setdefault(plan.worktree_id, []).append(plan)

        for worktree in worktrees:
            channel = channel_by_id.get(worktree.channel_id or "")
            agent_channel = (
                agent_server.store.get_channel(worktree.channel_id)
                if agent_server and worktree.channel_id
                else None
            )
            is_running = bool(
                agent_channel
                and agent_spawner
                and worktree.channel_id
                and agent_spawner.is_running(worktree.channel_id)
                and getattr(agent_channel, "status", "") == "active"
            )
            status = _dashboard_worktree_status(agent_channel, is_running) if agent_channel else worktree.status
            agent_config = _agent_settings_payload(agent_channel, agent_spawner, worktree.channel_id)
            updated_at = (
                _timestamp(getattr(agent_channel, "updated_at", None))
                or worktree.updated_at
                or _timestamp(getattr(channel, "created_at", None))
                or project_record.updated_at
            )
            last_active = _relative_time(updated_at)
            model = str(agent_config.get("model") or agent_config.get("harness") or "device")
            display_agent = _display_agent(agent_channel)
            current_plan = plans_by_worktree.get(worktree.id, [None])[0]
            plan_id = current_plan.id if current_plan else ""

            if status == "working":
                project["runningAgents"] += 1
            worktree_status[worktree.id] = status
            worktree_updated_at[worktree.id] = updated_at
            if worktree.channel_id:
                model_by_channel[worktree.channel_id] = model
            project["_last_active_ts"] = max(float(project["_last_active_ts"]), updated_at)
            project["lastActive"] = _relative_time(float(project["_last_active_ts"]))
            project["worktrees"].append({
                "id": worktree.id,
                "project_id": worktree.project_id,
                "channel_id": worktree.channel_id,
                "branch": worktree.branch or project_record.default_branch or "main",
                "plan": plan_id,
                "model": model,
                "agent": display_agent,
                "agent_config": agent_config,
                "harness": agent_config.get("harness", ""),
                "effort": agent_config.get("effort", ""),
                "auto_approve_tools": agent_config.get("auto_approve_tools", False),
                "agent_running": agent_config.get("is_running", False),
                "agent_status": agent_config.get("status", ""),
                "status": status,
                "summary": worktree.name,
                "device": device_id or "local",
                "workspace": worktree.path or project_record.root_path,
                "pct": 40 if status == "working" else 0,
                "files": 0,
                "add": 0,
                "del": 0,
                "updated": last_active,
            })

        for plan in plans:
            wt_status = worktree_status.get(plan.worktree_id or "", "idle")
            updated_at = max(plan.updated_at, worktree_updated_at.get(plan.worktree_id or "", 0))
            project["_last_active_ts"] = max(float(project["_last_active_ts"]), updated_at)
            project["lastActive"] = _relative_time(float(project["_last_active_ts"]))
            project["plans"].append({
                "id": plan.id,
                "project_id": plan.project_id,
                "worktree_id": plan.worktree_id,
                "channel_id": plan.channel_id,
                "title": plan.title,
                "status": _dashboard_plan_status(plan.status, wt_status),
                "steps": plan.step_count,
                "doneSteps": plan.done_step_count,
                "model": plan.model or model_by_channel.get(plan.channel_id or "", "device"),
                "updated": _relative_time(updated_at),
            })

        for worktree in worktrees:
            status = worktree_status.get(worktree.id, worktree.status)
            display_agent = "device"
            if worktree.channel_id and agent_server:
                display_agent = _display_agent(agent_server.store.get_channel(worktree.channel_id))
            project["activity"].append(f"{display_agent} {status} in {worktree.name}")

        project["activity"] = project["activity"][:4] or ["No recent agent activity"]
        out_projects.append(project)

    out_projects.sort(key=lambda item: float(item.get("_last_active_ts", 0)), reverse=True)
    for project in out_projects:
        project.pop("_last_active_ts", None)
    return {
        "schema": "dashboard.snapshot.v1",
        "source": "projects",
        "generated_at": time.time(),
        "projects": out_projects,
    }


def _ensure_project_graph(facade: "E2EEHandler") -> None:
    channels_in = facade.store.list_channels()
    agent_server = getattr(facade, "_agent_server", None)

    for channel in channels_in:
        agent_channel = agent_server.store.get_channel(channel.id) if agent_server else None
        cwd = str(getattr(agent_channel, "working_directory", "") or "")
        workspace = _workspace_identity(channel.name, cwd)
        project = facade.store.upsert_project(
            workspace["id"],
            workspace["name"],
            root_path=workspace["root_path"],
            repo=workspace["repo"],
            default_branch=workspace["branch"] or "main",
            color=_color_for(workspace["id"]),
        )
        status = "idle"
        if agent_channel and getattr(agent_channel, "status", "") == "error":
            status = "blocked"
        worktree = facade.store.upsert_worktree(
            channel.id,
            project.id,
            channel.name,
            path=cwd,
            branch=workspace["branch"] or "workspace",
            status=status,
            channel_id=channel.id,
        )
        plan_id = f"plan-{worktree.id[:8]}"
        if not facade.store.get_plan(plan_id):
            facade.store.upsert_plan(
                plan_id,
                project.id,
                channel.name,
                worktree_id=worktree.id,
                channel_id=channel.id,
                status="draft",
                step_count=1,
                done_step_count=0,
                model=str(getattr(agent_channel, "model", "") or ""),
            )


def _project_primitive_payload(project: Any, worktrees: list[Any]) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "root_path": project.root_path,
        "repo": project.repo,
        "default_branch": project.default_branch,
        "color": project.color,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "worktree_count": len(worktrees),
    }


def _worktree_primitive_payload(worktree: Any) -> dict[str, Any]:
    return {
        "id": worktree.id,
        "project_id": worktree.project_id,
        "channel_id": worktree.channel_id,
        "name": worktree.name,
        "path": worktree.path,
        "branch": worktree.branch,
        "base_ref": worktree.base_ref,
        "head_ref": worktree.head_ref,
        "status": worktree.status,
        "created_at": worktree.created_at,
        "updated_at": worktree.updated_at,
    }


def _plan_primitive_payload(plan: Any) -> dict[str, Any]:
    return {
        "id": plan.id,
        "project_id": plan.project_id,
        "worktree_id": plan.worktree_id,
        "channel_id": plan.channel_id,
        "title": plan.title,
        "status": plan.status,
        "body": plan.body,
        "step_count": plan.step_count,
        "done_step_count": plan.done_step_count,
        "model": plan.model,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
    }


def _agent_settings_payload(agent_channel: Any, agent_spawner: Any, channel_id: str | None) -> dict[str, Any]:
    is_running = bool(channel_id and agent_spawner and agent_spawner.is_running(channel_id))
    if not agent_channel:
        return {
            "harness": "",
            "model": "",
            "effort": "",
            "auto_approve_tools": False,
            "working_directory": "",
            "status": "missing",
            "is_running": is_running,
        }
    return {
        "harness": str(getattr(agent_channel, "harness", "") or ""),
        "model": str(getattr(agent_channel, "model", "") or ""),
        "effort": str(getattr(agent_channel, "effort", "") or ""),
        "auto_approve_tools": bool(getattr(agent_channel, "auto_approve_tools", False)),
        "working_directory": str(getattr(agent_channel, "working_directory", "") or ""),
        "status": str(getattr(agent_channel, "status", "") or ""),
        "is_running": is_running,
    }


async def _project_repo_list(project: Any) -> tuple[list[dict[str, Any]], str | None]:
    root_text = _clean_text(getattr(project, "root_path", ""))
    if not root_text:
        return [], "project has no root_path"
    try:
        root = Path(os.path.expanduser(os.path.expandvars(root_text))).resolve()
    except OSError as exc:
        return [], str(exc)
    if not root.is_dir():
        return [], "project root_path is not an existing directory"

    repo_paths = _find_project_repos(root)
    repo_payloads = await asyncio.gather(*(_repo_payload(root, repo) for repo in repo_paths))
    return list(repo_payloads), None


def _find_project_repos(root: Path) -> list[Path]:
    repos: list[Path] = []

    def visit(path: Path, depth: int) -> None:
        if len(repos) >= _PROJECT_REPO_SCAN_LIMIT:
            return
        if _is_git_repo_path(path):
            repos.append(path)
            return
        if depth >= _PROJECT_REPO_SCAN_DEPTH:
            return
        try:
            children = sorted(path.iterdir(), key=lambda child: child.name.lower())
        except OSError:
            return
        for child in children:
            if len(repos) >= _PROJECT_REPO_SCAN_LIMIT:
                return
            if not child.is_dir():
                continue
            if child.name in _PROJECT_REPO_SCAN_SKIP:
                continue
            if child.name.startswith(".") and child.name != ".worktrees":
                continue
            visit(child, depth + 1)

    visit(root, 0)
    return repos


def _is_git_repo_path(path: Path) -> bool:
    return (path / ".git").exists()


async def _repo_payload(project_root: Path, repo: Path) -> dict[str, Any]:
    branch, _, upstream = await files.git_branch_info(str(repo))
    remote = await files.git_remote_name(str(repo))
    try:
        relative = str(repo.relative_to(project_root))
    except ValueError:
        relative = str(repo)
    if relative == ".":
        relative = "."
    name = remote or (repo.name or relative or "repository")
    return {
        "id": f"repo-{hashlib.sha256(str(repo).encode()).hexdigest()[:12]}",
        "name": name,
        "path": str(repo),
        "relative_path": relative,
        "branch": branch or "main",
        "upstream": upstream,
        "remote": remote,
        "is_root": repo == project_root,
    }


def _resolve_project_repo(project: Any, value: Any) -> Path | None:
    repo_path = _clean_text(value)
    if not repo_path:
        return None
    root_text = _clean_text(getattr(project, "root_path", ""))
    if not root_text:
        raise V1Error("invalid_request", "project has no root_path", details={"field": "payload.repo_path"})

    try:
        project_root = Path(os.path.expanduser(os.path.expandvars(root_text))).resolve()
        candidate = Path(os.path.expanduser(os.path.expandvars(repo_path)))
        if not candidate.is_absolute():
            candidate = project_root / candidate
        candidate = candidate.resolve()
        candidate.relative_to(project_root)
    except (OSError, ValueError):
        raise V1Error(
            "invalid_request",
            "payload.repo_path must be inside the project root",
            details={"field": "payload.repo_path", "repo_path": repo_path},
        ) from None

    repo_root, _ = _git_workspace(str(candidate))
    if not repo_root or repo_root.resolve() != candidate:
        raise V1Error(
            "invalid_request",
            "payload.repo_path must point to a git repository root",
            details={"field": "payload.repo_path", "repo_path": str(candidate)},
        )
    return repo_root


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


async def _attach_worktree_agent(
    facade: "E2EEHandler",
    channel_id: str,
    agent: dict[str, Any],
    working_directory: str,
) -> tuple[dict[str, Any] | None, str | None]:
    agent_server = getattr(facade, "_agent_server", None)
    agent_spawner = getattr(facade, "_agent_spawner", None)

    if agent_server:
        agent_server.store.ensure_channel_row(channel_id)
        if working_directory:
            agent_server.store.update_working_directory(channel_id, working_directory)

    harness = _clean_text(agent.get("harness"))
    if not harness:
        return None, None

    harness_info = get_harness(harness)
    if not harness_info:
        return None, f"unknown harness: {harness}"

    model = _clean_text(agent.get("model")) or harness_info.default_model
    effort = _clean_text(agent.get("effort"))
    system_prompt = _clean_text(agent.get("system_prompt"))
    auto_approve_tools = bool(agent.get("auto_approve_tools", False))

    if agent_server:
        agent_server.store.update_harness(channel_id, harness)
        agent_server.store.update_model(channel_id, model)
        agent_server.store.update_effort(channel_id, effort)
        agent_server.store.update_auto_approve_tools(channel_id, auto_approve_tools)

    if not agent_spawner:
        return None, "agent spawner not available"

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
    except Exception as exc:  # noqa: BLE001 - creation still succeeds without a running agent.
        log.error("Failed to spawn worktree agent for channel %s: %s", channel_id[:8], exc)
        return None, str(exc)

    return {
        "agent_id": worker.agent_id,
        "harness": harness,
        "model": model,
        "effort": effort,
        "working_directory": working_directory,
        "pid": worker.pid,
    }, None


async def _create_git_worktree(
    project: Any,
    repo_root: Path | None,
    path: str,
    branch: str,
    base_ref: str,
) -> tuple[str, str | None]:
    if repo_root is None:
        repo_root, _ = _git_workspace(project.root_path)
    if not repo_root:
        return path, "repository not found"

    target = Path(os.path.expanduser(path)).resolve() if path else _default_worktree_path(repo_root, branch)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return str(target), str(exc)

    args = ["worktree", "add", "-b", branch, str(target), base_ref or "HEAD"]
    out, ok = await files._run_git(str(repo_root), args, timeout=30.0)
    if not ok:
        return str(target), out or "git worktree add failed"
    return str(target), None


def _default_worktree_path(repo_root: Path, branch: str) -> Path:
    base = repo_root.parent / f"{repo_root.name}.worktrees"
    leaf = _slug(branch.split("/")[-1] or branch)
    candidate = base / leaf
    if not candidate.exists():
        return candidate
    suffix = uuid.uuid4().hex[:6]
    return base / f"{leaf}-{suffix}"


def _review_denial_content(file_path: str, reason: str, diff: str, repo_path: str = "") -> str:
    reason_text = _bounded_text(reason, REVIEW_DENIAL_REASON_LIMIT)
    diff_text = _bounded_text(diff, REVIEW_DENIAL_DIFF_LIMIT)
    data = {
        "kind": "review_denied",
        "file": file_path,
        "reason": reason_text,
        "diff": diff_text,
    }
    if repo_path:
        data["repo_path"] = repo_path

    marker = (
        f"{REVIEW_DENIAL_BEGIN}"
        f"{json.dumps(data, ensure_ascii=True, separators=(',', ':'))}"
        f"{REVIEW_DENIAL_END}"
    )
    parts = [
        marker,
        "",
        f"Review denied for `{file_path}`.",
        "",
        "User feedback:",
        reason_text,
    ]
    if repo_path:
        parts.extend(["", f"Repository: `{repo_path}`."])
    if diff_text:
        parts.extend(["", "Denied diff:", "```diff", diff_text, "```"])
    parts.extend(["", "Please revise the work to address the feedback. Do not treat this as approval."])
    return "\n".join(parts)


def _bounded_text(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


async def _worktree_snapshot(facade: "E2EEHandler", worktree: Any) -> dict[str, Any]:
    project = facade.store.get_project(worktree.project_id)
    agent_server = getattr(facade, "_agent_server", None)
    agent_spawner = getattr(facade, "_agent_spawner", None)
    agent_channel = (
        agent_server.store.get_channel(worktree.channel_id)
        if agent_server and worktree.channel_id
        else None
    )
    cwd = _worktree_cwd(worktree, project, agent_channel)
    workspace_repo, _ = _git_workspace(cwd)
    repo = str(workspace_repo) if workspace_repo else (files.find_git_repo(cwd) if cwd else None)
    git_payload: dict[str, Any] = {
        "repo_path": "",
        "branch": worktree.branch or getattr(project, "default_branch", "") or "main",
        "staged": 0,
        "unstaged": 0,
        "commits": [],
        "error": None,
    }
    file_entries: list[dict[str, Any]] = []
    diffs: list[dict[str, Any]] = []

    if repo:
        repo_changes = await files._collect_repo_changes(repo, cwd)
        branch, _, upstream = await files.git_branch_info(repo)
        commits = await _worktree_commits(repo)
        if repo_changes:
            raw_entries = repo_changes.get("entries", [])
            file_entries = [_snapshot_file_entry(entry) for entry in raw_entries]
            git_payload["repo_path"] = repo_changes.get("path", "")
            git_payload["staged"] = sum(1 for entry in raw_entries if entry.get("staged_status"))
            git_payload["unstaged"] = sum(1 for entry in raw_entries if entry.get("git_status"))
            diffs = await _worktree_diffs(repo, cwd, raw_entries)
        else:
            git_payload["repo_path"] = files._repo_display_path(repo, cwd)
        git_payload["branch"] = branch or git_payload["branch"]
        git_payload["upstream"] = upstream
        git_payload["commits"] = commits
    else:
        git_payload["error"] = "Repository not found"

    return {
        "schema": "worktree.snapshot.v1",
        "worktree": _worktree_primitive_payload(worktree),
        "project": _project_primitive_payload(project, facade.store.list_worktrees(project.id)) if project else None,
        "agent": _agent_settings_payload(agent_channel, agent_spawner, worktree.channel_id),
        "workspace": cwd,
        "git": git_payload,
        "files": file_entries,
        "diffs": diffs,
        "tests": [],
    }


def _worktree_cwd(worktree: Any, project: Any, agent_channel: Any) -> str:
    raw = (
        getattr(agent_channel, "working_directory", "")
        or getattr(worktree, "path", "")
        or getattr(project, "root_path", "")
        or ""
    )
    if not raw:
        return ""
    try:
        return str(Path(os.path.expanduser(str(raw))).resolve())
    except OSError:
        return str(raw)


def _snapshot_file_entry(entry: dict[str, Any]) -> dict[str, Any]:
    status = entry.get("git_status") or entry.get("staged_status") or "M"
    return {
        "path": entry.get("path", ""),
        "status": status,
        "add": int(entry.get("insertions") or 0),
        "del": int(entry.get("deletions") or 0),
        "staged_status": entry.get("staged_status"),
        "git_status": entry.get("git_status"),
    }


async def _worktree_commits(repo: str, limit: int = 8) -> list[dict[str, Any]]:
    fmt = "%H%x1f%h%x1f%ct%x1f%s"
    out, ok = await files._run_git(repo, ["log", f"--max-count={limit}", f"--pretty=format:{fmt}"])
    if not ok or not out:
        return []
    commits: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.split("\x1f", 3)
        if len(parts) != 4:
            continue
        sha, short_sha, timestamp, subject = parts
        commits.append({
            "sha": short_sha,
            "full_sha": sha,
            "message": subject,
            "time": _relative_time(_timestamp(timestamp)),
        })
    return commits


async def _worktree_diffs(repo: str, cwd: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for entry in entries[:3]:
        rel_path = _repo_relative_path(repo, cwd, entry.get("path", ""))
        if not rel_path:
            continue
        out, ok = await files._run_git(repo, ["diff", "--", rel_path], timeout=5.0)
        if not out:
            out, ok = await files._run_git(repo, ["diff", "--cached", "--", rel_path], timeout=5.0)
        if not ok or not out:
            continue
        parsed = _parse_unified_diff(out)
        if parsed:
            diffs.append({
                "file": entry.get("path", rel_path),
                "add": int(entry.get("insertions") or 0),
                "del": int(entry.get("deletions") or 0),
                "lines": parsed[:120],
            })
    return diffs


def _repo_relative_path(repo: str, cwd: str, path: str) -> str:
    try:
        abs_path = (Path(cwd) / path).resolve()
        return str(abs_path.relative_to(Path(repo).resolve()))
    except (OSError, ValueError):
        return path


def _parse_unified_diff(diff_text: str) -> list[dict[str, str]]:
    lines: list[dict[str, str]] = []
    for raw in diff_text.splitlines():
        if raw.startswith(("diff --git", "index ", "+++", "---")):
            continue
        if raw.startswith("@@"):
            continue
        if raw.startswith("+"):
            lines.append({"type": "add", "text": raw[1:]})
        elif raw.startswith("-"):
            lines.append({"type": "del", "text": raw[1:]})
        elif raw.startswith(" "):
            lines.append({"type": "ctx", "text": raw[1:]})
    return lines


def _dashboard_plan_status(plan_status: str, worktree_status: str) -> str:
    if plan_status not in {"draft", "queued"}:
        return plan_status
    if worktree_status in {"working", "blocked"}:
        return "in-progress"
    return plan_status


def _workspace_identity(channel_name: str, working_directory: str) -> dict[str, str]:
    cwd = os.path.expanduser(working_directory) if working_directory else ""
    root, branch = _git_workspace(cwd)
    if root:
        name = root.name or "workspace"
        key = str(root)
        return {
            "key": key,
            "id": _workspace_project_id(key, name),
            "name": name,
            "description": "Repository workspace",
            "repo": name,
            "root_path": key,
            "branch": branch or "main",
        }

    if cwd:
        path = Path(cwd).resolve()
        name = path.name or channel_name or "workspace"
        key = str(path)
        return {
            "key": key,
            "id": _workspace_project_id(key, name),
            "name": name,
            "description": "Local workspace",
            "repo": name,
            "root_path": key,
            "branch": "workspace",
        }

    return {
        "key": "channels",
        "id": "channels",
        "name": "Channels",
        "description": "Bridge channels",
        "repo": "local/channels",
        "root_path": "",
        "branch": "main",
    }


def _workspace_project_id(key: str, name: str) -> str:
    if key == "channels":
        return "channels"
    return f"{_slug(name)}-{hashlib.sha256(key.encode()).hexdigest()[:6]}"


def _git_workspace(cwd: str) -> tuple[Path | None, str]:
    if not cwd:
        return None, ""
    try:
        path = Path(cwd).expanduser().resolve()
    except OSError:
        return None, ""
    if path.is_file():
        path = path.parent
    candidates = (path, *path.parents)
    for candidate in candidates:
        git_marker = candidate / ".git"
        if git_marker.exists():
            return candidate, _git_branch(git_marker)
    return None, ""


def _git_branch(git_marker: Path) -> str:
    try:
        git_dir = git_marker
        if git_marker.is_file():
            text = git_marker.read_text(errors="ignore").strip()
            if text.startswith("gitdir:"):
                raw = text.split(":", 1)[1].strip()
                git_dir = Path(raw)
                if not git_dir.is_absolute():
                    git_dir = (git_marker.parent / git_dir).resolve()
        head = (git_dir / "HEAD").read_text(errors="ignore").strip()
    except OSError:
        return "main"

    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        prefix = "refs/heads/"
        return ref[len(prefix):] if ref.startswith(prefix) else ref
    return head[:7] if head else "main"


def _dashboard_worktree_status(agent_channel: Any, is_running: bool) -> str:
    status = str(getattr(agent_channel, "status", "") or "")
    if status == "error":
        return "blocked"
    if is_running:
        return "working"
    return "idle"


def _display_agent(agent_channel: Any) -> str:
    if not agent_channel:
        return "device"
    return str(
        getattr(agent_channel, "model", "")
        or getattr(agent_channel, "harness", "")
        or "device"
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "workspace"


def _color_for(value: str) -> str:
    digest = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)
    return _DASHBOARD_COLORS[digest % len(_DASHBOARD_COLORS)]


def _timestamp(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        if value.isdigit():
            return float(value)
        try:
            text = value[:-1] + "+00:00" if value.endswith("Z") else value
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _relative_time(timestamp: float) -> str:
    if not timestamp:
        return "now"
    seconds = max(0, int(datetime.now(timezone.utc).timestamp() - timestamp))
    if seconds < 60:
        return "now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def _response_meta(app: dict[str, Any]) -> dict[str, Any]:
    meta = app.get("meta") or {}
    out: dict[str, Any] = {}
    if isinstance(meta, dict) and meta.get("trace_id"):
        out["trace_id"] = meta["trace_id"]
    return out


def _normalise_channel(ch: dict[str, Any]) -> dict[str, Any]:
    status = "running" if ch.get("is_running") else "idle"
    return {
        "id": ch.get("id"),
        "name": ch.get("name"),
        "created_at": ch.get("created_at"),
        "agent": {
            "harness": ch.get("harness"),
            "display_name": ch.get("agent_name"),
            "model": ch.get("model"),
            "effort": ch.get("effort"),
            "working_directory": ch.get("working_directory"),
            "plan_mode": bool(ch.get("plan_mode", False)),
            "auto_approve_tools": bool(ch.get("auto_approve_tools", False)),
            "status": status,
            "last_seen_at": ch.get("last_seen_at"),
        },
    }


def _sha256_file(path: Any) -> str:
    if not path:
        return ""
    try:
        p = Path(str(path))
        if not p.is_file():
            return ""
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def translate_v0_broadcast(facade: "E2EEHandler", payload: dict[str, Any]) -> dict[str, Any]:
    """Translate a v0 broadcast payload into a v1 event envelope."""
    action = payload.get("action")
    channel_id = payload.get("channel_id") or payload.get("message", {}).get("channel_id")
    target = _target(facade, {"target": {}}, channel_id)

    event_type = "protocol.event"
    event_payload: dict[str, Any] = {k: v for k, v in payload.items() if k != "action"}

    if action == v0.MESSAGE:
        event_type = "message.created"
    elif action == v0.SYSTEM_MESSAGE:
        event_type = "message.created"
        event_payload = {
            "message": {
                "channel_id": payload.get("channel_id"),
                "sender": "system",
                "content": payload.get("text", ""),
            }
        }
    elif action == v0.COMPLICATION_UPDATE:
        event_type = "complication.updated"
    elif action == v0.COMPLICATION_REMOVE:
        event_type = "complication.removed"
    elif action == v0.AGENT_EVENT:
        event_type, event_payload = _translate_agent_event(payload)
        channel_id = payload.get("channel_id")
        target = _target(facade, {"target": {}}, channel_id)

    return {
        "v": V1_VERSION,
        "kind": "event",
        "id": _new_id("evt"),
        "type": event_type,
        "target": target,
        "payload": event_payload,
        "error": None,
        "meta": {},
    }


def _translate_agent_event(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    event_type = payload.get("event_type", "")
    event = payload.get("event", {})
    mapping = {
        "agent.connected": "agent.connected",
        "agent.disconnected": "agent.disconnected",
        "agent.error": "agent.error",
        "agent.state_update": "agent.state_changed",
        "agent.file_changes": "workspace.changed",
        "chat.response": "message.created",
        "activity.delta": "activity.delta",
        "activity.ping": "activity.ping",
        "activity.end": "activity.end",
        "tool.use": "tool.used",
        "tool.result": "tool.completed",
        "interaction.request": "interaction.requested",
    }
    out_type = mapping.get(event_type, "agent.event")
    if event_type == "chat.response":
        return out_type, {
            "message": {
                "id": event.get("id"),
                "sender": event.get("sender"),
                "content": event.get("content"),
                "suggested_actions": event.get("suggested_actions"),
            }
        }
    if event_type == "agent.file_changes":
        return out_type, {"paths": event.get("paths", [])}
    return out_type, event if isinstance(event, dict) else {"event": event}
