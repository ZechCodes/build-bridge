from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from build_bridge.e2ee import E2EEHandler


class _FakeSession:
    session_id = "fake"
    session_key_b64 = "fake"


@pytest.fixture
def handler() -> E2EEHandler:
    h = E2EEHandler(SimpleNamespace(), MagicMock())
    h._sent_frames = []

    async def fake_send(session, ws, payload):
        h._sent_frames.append(payload)

    h._send_frame = fake_send  # type: ignore[assignment]
    agent_server = MagicMock()
    h._agent_server = agent_server
    return h


def _set_channel_cwd(h: E2EEHandler, channel_id: str, cwd: Path) -> None:
    ch = SimpleNamespace(id=channel_id, working_directory=str(cwd), harness="")
    h._agent_server.store.get_channel = lambda cid: ch if cid == channel_id else None


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "app.txt").write_text("one\n")
    _commit(repo, "initial")
    (repo / "app.txt").write_text("one\ntwo\n")
    _commit(repo, "second")
    return repo


@pytest.mark.asyncio
async def test_files_commits_lists_recent_commits(handler: E2EEHandler, repo: Path):
    _set_channel_cwd(handler, "ch1", repo.parent)

    await handler._handle_files_commits(_FakeSession(), {
        "channel_id": "ch1",
        "repo_path": "repo",
    }, None)

    frame = next(f for f in handler._sent_frames if f["action"] == "files_commits_result")
    assert frame["repo_path"] == "repo"
    assert [c["subject"] for c in frame["commits"][:2]] == ["second", "initial"]
    assert len(frame["commits"][0]["sha"]) == 40


@pytest.mark.asyncio
async def test_revision_changes_between_commits(handler: E2EEHandler, repo: Path):
    _set_channel_cwd(handler, "ch1", repo.parent)
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", "HEAD~1")

    await handler._handle_files_changes(_FakeSession(), {
        "channel_id": "ch1",
        "repo_path": "repo",
        "newer_ref": head,
        "older_ref": parent,
    }, None)

    frame = next(f for f in handler._sent_frames if f["action"] == "files_changes_result")
    repo_frame = frame["repos"][0]
    assert repo_frame["path"] == "repo"
    assert repo_frame["newer_ref"] == head
    assert repo_frame["older_ref"] == parent
    assert [(e["path"], e["insertions"], e["deletions"]) for e in repo_frame["entries"]] == [
        ("repo/app.txt", 1, 0),
    ]


@pytest.mark.asyncio
async def test_worktree_changes_include_untracked(handler: E2EEHandler, repo: Path):
    _set_channel_cwd(handler, "ch1", repo.parent)
    (repo / "app.txt").write_text("one\ntwo\nthree\n")
    (repo / "new.txt").write_text("new\n")

    await handler._handle_files_changes(_FakeSession(), {
        "channel_id": "ch1",
        "repo_path": "repo",
        "newer_ref": "worktree",
        "older_ref": "HEAD",
    }, None)

    frame = next(f for f in handler._sent_frames if f["action"] == "files_changes_result")
    entries = {e["path"]: e for e in frame["repos"][0]["entries"]}
    assert entries["repo/app.txt"]["insertions"] == 1
    assert entries["repo/new.txt"]["git_status"] == "?"


@pytest.mark.asyncio
async def test_revision_file_diff(handler: E2EEHandler, repo: Path):
    _set_channel_cwd(handler, "ch1", repo.parent)
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", "HEAD~1")

    await handler._handle_file_diff(_FakeSession(), {
        "channel_id": "ch1",
        "path": "repo/app.txt",
        "repo_path": "repo",
        "newer_ref": head,
        "older_ref": parent,
    }, None)

    frame = next(f for f in handler._sent_frames if f["action"] == "file_diff_result")
    assert frame["repo_path"] == "repo"
    assert "+two" in frame["diff"]
