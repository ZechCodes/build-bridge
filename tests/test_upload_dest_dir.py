"""Tests for the workspace-upload path: uploads with a `dest_dir`
land inside the channel's working directory, not the scratch
uploads registry."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from build_bridge.e2ee import E2EEHandler


class _FakeSession:
    session_id = "fake"
    session_key_b64 = "fake"


@pytest.fixture
def handler(tmp_path: Path, monkeypatch) -> E2EEHandler:
    """Build an E2EEHandler with just enough wiring to exercise the
    upload handlers. _send_frame is stubbed to record payloads."""
    # Redirect the uploads base to tmp so we don't touch the user's
    # real ~/.config tree.
    monkeypatch.setattr(E2EEHandler, "_UPLOADS_BASE", tmp_path / "_uploads_base")

    # Minimal config + store — neither is used by the upload handlers
    # directly, but the constructor requires them.
    cfg = SimpleNamespace()
    store = MagicMock()
    h = E2EEHandler(cfg, store)

    # Record every `_send_frame` payload.
    h._sent_frames = []

    async def fake_send(session, ws, payload):
        h._sent_frames.append(payload)

    h._send_frame = fake_send   # type: ignore[assignment]

    # Minimal agent_server stub so `_get_channel_cwd` resolves.
    agent_server = MagicMock()
    h._agent_server = agent_server
    return h


def _set_channel_cwd(h: E2EEHandler, channel_id: str, cwd: Path) -> None:
    ch = SimpleNamespace(
        id=channel_id,
        working_directory=str(cwd),
        harness="",
    )
    h._agent_server.store.get_channel = lambda cid: ch if cid == channel_id else None


async def _feed_one_chunk(
    h: E2EEHandler,
    *,
    channel_id: str,
    file_id: str,
    data: bytes,
    filename: str,
    dest_dir: str | None = None,
) -> str:
    """Feed a single-chunk upload through handle_upload_chunk +
    handle_upload_complete. Returns the SHA256 used."""
    sha = hashlib.sha256(data).hexdigest()
    payload = {
        "file_id": file_id,
        "channel_id": channel_id,
        "chunk_index": 0,
        "total_chunks": 1,
        "total_size": len(data),
        "filename": filename,
        "mime_type": "text/plain",
        "data": base64.b64encode(data).decode(),
    }
    if dest_dir is not None:
        payload["dest_dir"] = dest_dir
    await h._handle_upload_chunk(_FakeSession(), payload, None)
    await h._handle_upload_complete(_FakeSession(), {
        "file_id": file_id,
        "channel_id": channel_id,
        "sha256": sha,
    }, None)
    return sha


@pytest.mark.asyncio
async def test_upload_with_dest_dir_lands_in_workspace(handler, tmp_path: Path):
    cwd = tmp_path / "work"; cwd.mkdir()
    (cwd / "api").mkdir()
    _set_channel_cwd(handler, "ch1", cwd)

    await _feed_one_chunk(
        handler, channel_id="ch1", file_id="f1",
        data=b"hello\n", filename="hello.txt", dest_dir="api",
    )

    expected = cwd / "api" / "hello.txt"
    assert expected.is_file()
    assert expected.read_bytes() == b"hello\n"

    accepted = next(p for p in handler._sent_frames if p["action"] == "upload_accepted")
    assert accepted["path"] == str(expected)


@pytest.mark.asyncio
async def test_upload_creates_missing_parents(handler, tmp_path: Path):
    cwd = tmp_path / "work"; cwd.mkdir()
    _set_channel_cwd(handler, "ch1", cwd)

    await _feed_one_chunk(
        handler, channel_id="ch1", file_id="f2",
        data=b"x", filename="deep.txt",
        dest_dir="nested/subdir",
    )
    assert (cwd / "nested" / "subdir" / "deep.txt").is_file()


@pytest.mark.asyncio
async def test_upload_rejects_escape(handler, tmp_path: Path):
    cwd = tmp_path / "work"; cwd.mkdir()
    _set_channel_cwd(handler, "ch1", cwd)
    (tmp_path / "sibling").mkdir()

    await _feed_one_chunk(
        handler, channel_id="ch1", file_id="f3",
        data=b"x", filename="pwned.txt",
        dest_dir="../sibling",
    )
    assert not (tmp_path / "sibling" / "pwned.txt").exists()
    err = next(p for p in handler._sent_frames if p["action"] == "upload_error")
    assert "outside workspace" in err["error"]


@pytest.mark.asyncio
async def test_upload_rejects_existing_file(handler, tmp_path: Path):
    cwd = tmp_path / "work"; cwd.mkdir()
    (cwd / "api").mkdir()
    (cwd / "api" / "already.txt").write_text("old")
    _set_channel_cwd(handler, "ch1", cwd)

    await _feed_one_chunk(
        handler, channel_id="ch1", file_id="f4",
        data=b"new", filename="already.txt", dest_dir="api",
    )
    assert (cwd / "api" / "already.txt").read_text() == "old"
    err = next(p for p in handler._sent_frames if p["action"] == "upload_error")
    assert err["error"] == "file exists"


@pytest.mark.asyncio
async def test_upload_without_dest_dir_uses_scratch(handler, tmp_path: Path):
    cwd = tmp_path / "work"; cwd.mkdir()
    _set_channel_cwd(handler, "ch1", cwd)

    await _feed_one_chunk(
        handler, channel_id="ch1", file_id="f5",
        data=b"scratch", filename="s.txt",
    )
    # No dest_dir → scratch uploads base, NOT cwd.
    assert not (cwd / "s.txt").exists()
    scratch = handler._UPLOADS_BASE / "ch1" / "f5" / "s.txt"
    assert scratch.is_file()
    assert scratch.read_bytes() == b"scratch"
