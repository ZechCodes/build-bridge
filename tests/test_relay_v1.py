"""Tests for the v1 relay adapter running alongside the v0 API."""

from __future__ import annotations

import base64
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from build_bridge.e2ee import E2EEHandler
from build_bridge.relay_session import ActiveSession
from build_bridge.storage import MessageStore


def _handler(tmp_path: Path) -> E2EEHandler:
    cfg = SimpleNamespace(device_id="dev-1")
    return E2EEHandler(cfg, MessageStore(tmp_path / "messages.db"))


def _session() -> ActiveSession:
    return ActiveSession(session_id="s1", session_key_b64="x" * 44)


def _v1_request(
    frame_id: str,
    method: str,
    *,
    channel_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = {"device_id": "dev-1"}
    if channel_id:
        target["channel_id"] = channel_id
    return {
        "v": 1,
        "kind": "request",
        "id": frame_id,
        "type": method,
        "target": target,
        "payload": payload or {},
        "meta": {"trace_id": "trace-1"},
    }


def _capture(handler: E2EEHandler) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def fake_send(session, ws, payload):
        sent.append(payload)

    handler._send_frame = fake_send  # type: ignore[assignment]
    return sent


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.mark.asyncio
async def test_v1_protocol_hello_routes_before_v0_dispatch(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    sent = _capture(handler)
    session = _session()

    await handler._session_mgr._handle_data_frame(
        session,
        {"frame_type": "data", "payload": _v1_request("req-1", "protocol.hello")},
        object(),
    )

    assert session.protocol_version == 1
    assert sent[0]["v"] == 1
    assert sent[0]["kind"] == "response"
    assert sent[0]["ref"] == "req-1"
    assert sent[0]["type"] == "protocol.hello"
    assert sent[0]["payload"]["version"] == 1
    assert sent[0]["meta"]["trace_id"] == "trace-1"


@pytest.mark.asyncio
async def test_v0_payload_still_uses_v0_dispatch(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    sent = _capture(handler)
    session = _session()

    await handler._session_mgr._handle_data_frame(
        session,
        {"frame_type": "data", "payload": {"action": "list_channels"}},
        object(),
    )

    assert session.protocol_version == 0
    assert sent[0]["action"] == "channel_list"
    assert "v" not in sent[0]


@pytest.mark.asyncio
async def test_v1_channel_list_wraps_v0_handler_output(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    handler.store.create_channel("ch-1", "General")
    sent = _capture(handler)

    await handler._session_mgr._handle_data_frame(
        _session(),
        {"frame_type": "data", "payload": _v1_request("req-2", "channel.list")},
        object(),
    )

    response = sent[0]
    assert response["kind"] == "response"
    assert response["type"] == "channel.list"
    assert response["payload"]["channels"][0]["id"] == "ch-1"
    assert response["payload"]["channels"][0]["name"] == "General"


@pytest.mark.asyncio
async def test_v1_dashboard_snapshot_derives_projects_from_channels(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    handler.store.create_channel("ch-1", "Tenant middleware")
    sent = _capture(handler)

    await handler._session_mgr._handle_data_frame(
        _session(),
        {"frame_type": "data", "payload": _v1_request("req-dash", "dashboard.snapshot")},
        object(),
    )

    response = sent[0]
    assert response["kind"] == "response"
    assert response["type"] == "dashboard.snapshot"
    assert response["payload"]["schema"] == "dashboard.snapshot.v1"
    assert response["payload"]["source"] == "projects"
    assert response["payload"]["projects"][0]["id"] == "channels"
    assert response["payload"]["projects"][0]["worktrees"][0]["project_id"] == "channels"
    assert response["payload"]["projects"][0]["plans"][0]["channel_id"] == "ch-1"
    assert response["payload"]["projects"][0]["worktrees"][0]["id"] == "ch-1"
    assert response["payload"]["projects"][0]["worktrees"][0]["summary"] == "Tenant middleware"
    assert handler.store.get_project("channels") is not None
    assert handler.store.get_worktree("ch-1") is not None
    assert handler.store.get_plan("plan-ch-1") is not None


@pytest.mark.asyncio
async def test_v1_project_worktree_and_plan_list_use_primitives(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    handler.store.upsert_project(
        "proj-1",
        "api-gateway",
        root_path="/work/api-gateway",
        repo="api-gateway",
        default_branch="main",
        color="#6366f1",
    )
    handler.store.upsert_worktree(
        "wt-1",
        "proj-1",
        "Tenant middleware",
        path="/work/api-gateway",
        branch="agents/tenant",
        status="idle",
        channel_id="ch-1",
    )
    handler.store.upsert_plan(
        "plan-1",
        "proj-1",
        "Migrate tenant middleware",
        worktree_id="wt-1",
        channel_id="ch-1",
        status="queued",
        step_count=3,
        done_step_count=1,
        model="claude-sonnet",
    )
    sent = _capture(handler)

    await handler._session_mgr._handle_data_frame(
        _session(),
        {"frame_type": "data", "payload": _v1_request("req-projects", "project.list")},
        object(),
    )
    await handler._session_mgr._handle_data_frame(
        _session(),
        {
            "frame_type": "data",
            "payload": _v1_request(
                "req-worktrees",
                "worktree.list",
                payload={"project_id": "proj-1"},
            ),
        },
        object(),
    )
    await handler._session_mgr._handle_data_frame(
        _session(),
        {
            "frame_type": "data",
            "payload": _v1_request(
                "req-plans",
                "plan.list",
                payload={"worktree_id": "wt-1"},
            ),
        },
        object(),
    )

    assert sent[0]["type"] == "project.list"
    assert sent[0]["payload"]["projects"][0]["id"] == "proj-1"
    assert sent[0]["payload"]["projects"][0]["worktree_count"] == 1
    assert sent[1]["type"] == "worktree.list"
    assert sent[1]["payload"]["worktrees"][0]["id"] == "wt-1"
    assert sent[1]["payload"]["worktrees"][0]["project_id"] == "proj-1"
    assert sent[2]["type"] == "plan.list"
    assert sent[2]["payload"]["plans"][0]["id"] == "plan-1"
    assert sent[2]["payload"]["plans"][0]["worktree_id"] == "wt-1"


@pytest.mark.asyncio
async def test_v1_worktree_create_attaches_channel_and_plan(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    handler.store.upsert_project(
        "proj-1",
        "api-gateway",
        root_path="/work/api-gateway",
        repo="api-gateway",
        default_branch="main",
        color="#6366f1",
    )
    sent = _capture(handler)

    await handler._session_mgr._handle_data_frame(
        _session(),
        {
            "frame_type": "data",
            "payload": _v1_request(
                "req-worktree-create",
                "worktree.create",
                payload={"project_id": "proj-1", "name": "Tenant middleware"},
            ),
        },
        object(),
    )

    payload = sent[0]["payload"]
    channel_id = payload["channel"]["id"]
    worktree_id = payload["worktree"]["id"]
    assert sent[0]["type"] == "worktree.create"
    assert sent[0]["target"]["channel_id"] == channel_id
    assert handler.store.get_channel(channel_id).name == "Tenant middleware"
    assert handler.store.get_worktree(worktree_id).channel_id == channel_id
    assert payload["worktree"]["project_id"] == "proj-1"
    assert payload["worktree"]["path"] == "/work/api-gateway"
    assert payload["plan"]["worktree_id"] == worktree_id
    assert payload["plan"]["channel_id"] == channel_id


@pytest.mark.asyncio
async def test_v1_worktree_snapshot_returns_live_git_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    app_file = repo / "app.txt"
    app_file.write_text("one\n")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "initial")
    app_file.write_text("one\ntwo\n")

    handler = _handler(tmp_path)
    handler.store.upsert_project("proj-1", "repo", root_path=str(repo), repo="repo", default_branch="main")
    handler.store.upsert_worktree("wt-1", "proj-1", "Edit app", path=str(repo), branch="main", status="idle")
    sent = _capture(handler)

    await handler._session_mgr._handle_data_frame(
        _session(),
        {"frame_type": "data", "payload": _v1_request("req-wt-snap", "worktree.snapshot", payload={"worktree_id": "wt-1"})},
        object(),
    )

    payload = sent[0]["payload"]
    assert sent[0]["type"] == "worktree.snapshot"
    assert payload["schema"] == "worktree.snapshot.v1"
    assert payload["worktree"]["id"] == "wt-1"
    assert payload["files"][0]["path"] == "app.txt"
    assert payload["files"][0]["status"] == "M"
    assert payload["git"]["unstaged"] == 1
    assert payload["git"]["commits"][0]["message"] == "initial"
    assert payload["diffs"][0]["file"] == "app.txt"
    assert {"type": "add", "text": "two"} in payload["diffs"][0]["lines"]


@pytest.mark.asyncio
async def test_v1_unknown_method_returns_normalized_error(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    sent = _capture(handler)

    await handler._session_mgr._handle_data_frame(
        _session(),
        {"frame_type": "data", "payload": _v1_request("req-3", "nope.missing")},
        object(),
    )

    assert sent[0]["kind"] == "response"
    assert sent[0]["ref"] == "req-3"
    assert sent[0]["error"]["code"] == "unknown_method"
    assert sent[0]["payload"] == {}


@pytest.mark.asyncio
async def test_v1_broadcasts_are_translated_per_session(tmp_path: Path) -> None:
    handler = _handler(tmp_path)
    sent: list[dict[str, Any]] = []

    async def fake_send(session, ws, payload):
        sent.append(payload)

    handler._session_mgr.send_frame = fake_send  # type: ignore[assignment]
    handler._relay_ws = object()
    handler._sessions = {
        "v0": ActiveSession("v0", "x" * 44),
        "v1": ActiveSession("v1", "x" * 44, protocol_version=1),
    }

    await handler.broadcast_to_sessions(
        "ch-1",
        {"action": "message", "message": {"id": "m1", "channel_id": "ch-1", "content": "hi"}},
    )

    assert sent[0]["action"] == "message"
    assert sent[1]["v"] == 1
    assert sent[1]["kind"] == "event"
    assert sent[1]["type"] == "message.created"
    assert sent[1]["payload"]["message"]["id"] == "m1"


@pytest.mark.asyncio
async def test_v1_upload_create_write_complete_uses_upload_assembler(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(E2EEHandler, "_UPLOADS_BASE", tmp_path / "uploads")
    handler = _handler(tmp_path)
    sent = _capture(handler)
    session = _session()
    data = b"hello"
    sha = hashlib.sha256(data).hexdigest()

    await handler._session_mgr._handle_data_frame(
        session,
        {
            "frame_type": "data",
            "payload": _v1_request(
                "upl-create",
                "upload.create",
                channel_id="ch-1",
                payload={
                    "upload_id": "upload-1",
                    "filename": "hello.txt",
                    "mime_type": "text/plain",
                    "size": len(data),
                    "sha256": sha,
                    "destination": {"kind": "scratch"},
                },
            ),
        },
        object(),
    )
    await handler._session_mgr._handle_data_frame(
        session,
        {
            "frame_type": "data",
            "payload": _v1_request(
                "upl-write",
                "upload.write_chunk",
                channel_id="ch-1",
                payload={
                    "upload_id": "upload-1",
                    "index": 0,
                    "data": base64.b64encode(data).decode(),
                },
            ),
        },
        object(),
    )
    await handler._session_mgr._handle_data_frame(
        session,
        {
            "frame_type": "data",
            "payload": _v1_request(
                "upl-complete",
                "upload.complete",
                channel_id="ch-1",
                payload={"upload_id": "upload-1"},
            ),
        },
        object(),
    )

    completed = sent[-1]
    assert completed["kind"] == "response"
    assert completed["ref"] == "upl-complete"
    assert completed["payload"]["upload"]["id"] == "upload-1"
    assert completed["payload"]["upload"]["sha256"] == sha
    assert Path(completed["payload"]["upload"]["path"]).read_bytes() == data
