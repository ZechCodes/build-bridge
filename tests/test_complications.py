"""Tests for ComplicationRegistry — the bits we actually care about
bug-for-bug: `on_filesystem_change` must schedule a debounced evaluation
for the channel's git repo without requiring a tool event first."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from build_bridge.complications import (
    ComplicationRegistry,
    GitStatusData,
    FileDiffSummary,
    invalidate_git_repo_cache,
)


def _init_git_repo(path: Path) -> None:
    """Create a minimal git repo with one committed file."""
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init", "--quiet"], cwd=path, check=True)


@pytest.mark.asyncio
async def test_on_filesystem_change_schedules_eval(tmp_path: Path) -> None:
    """on_filesystem_change schedules a debounced eval when the working
    directory resolves to a git repo."""
    _init_git_repo(tmp_path)
    invalidate_git_repo_cache()

    broadcast = AsyncMock()
    reg = ComplicationRegistry(broadcast=broadcast, debounce_ms=10)

    fake_status = GitStatusData(repo=str(tmp_path.resolve()), branch="main")
    with patch("build_bridge.complications.evaluate_git_status",
               AsyncMock(return_value=fake_status)):
        await reg.on_filesystem_change("ch-1", str(tmp_path))
        await asyncio.sleep(0.05)

    assert broadcast.await_count >= 1
    msg = broadcast.await_args_list[0].args[1]
    assert msg["action"] == "complication:update"
    assert msg["kind"] == "git-status"
    assert msg["channel_id"] == "ch-1"
    assert msg["data"]["branch"] == "main"

    reg.stop_polling()


@pytest.mark.asyncio
async def test_on_filesystem_change_noop_outside_repo(tmp_path: Path) -> None:
    """If the working directory isn't in a git repo and nothing is
    tracked yet, on_filesystem_change is a no-op."""
    invalidate_git_repo_cache()
    broadcast = AsyncMock()
    reg = ComplicationRegistry(broadcast=broadcast, debounce_ms=10)

    with patch("build_bridge.complications.evaluate_git_status", AsyncMock()):
        await reg.on_filesystem_change("ch-none", str(tmp_path))
        await asyncio.sleep(0.05)

    broadcast.assert_not_awaited()
    reg.stop_polling()


@pytest.mark.asyncio
async def test_on_filesystem_change_reuses_tracked_repos(tmp_path: Path) -> None:
    """Even if working_directory doesn't resolve to a repo, previously-
    tracked repos for the channel get re-evaluated."""
    _init_git_repo(tmp_path)
    invalidate_git_repo_cache()

    broadcast = AsyncMock()
    reg = ComplicationRegistry(broadcast=broadcast, debounce_ms=10)
    repo_str = str(tmp_path.resolve())
    # Pretend a prior tool_event tracked this repo.
    reg._active_repos["ch-2"] = {repo_str}

    fake_status = GitStatusData(repo=repo_str, branch="main")
    with patch("build_bridge.complications.evaluate_git_status",
               AsyncMock(return_value=fake_status)):
        # Bogus working directory — but tracking should kick in anyway.
        await reg.on_filesystem_change("ch-2", "/nonexistent/dir")
        await asyncio.sleep(0.05)

    assert broadcast.await_count >= 1
    reg.stop_polling()


@pytest.mark.asyncio
async def test_rapid_filesystem_changes_debounce_to_one_eval(tmp_path: Path) -> None:
    """Multiple filesystem-change signals within the debounce window
    collapse into one git evaluation (not one per event)."""
    _init_git_repo(tmp_path)
    invalidate_git_repo_cache()

    broadcast = AsyncMock()
    reg = ComplicationRegistry(broadcast=broadcast, debounce_ms=50)

    eval_mock = AsyncMock(return_value=GitStatusData(repo=str(tmp_path.resolve()), branch="main"))
    with patch("build_bridge.complications.evaluate_git_status", eval_mock):
        for _ in range(5):
            await reg.on_filesystem_change("ch-burst", str(tmp_path))
        await asyncio.sleep(0.2)

    # One evaluation covers all five pokes.
    assert eval_mock.await_count == 1
    reg.stop_polling()
