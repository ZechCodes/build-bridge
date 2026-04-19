"""Tests for build_bridge.agent_spawner — agent process management."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from build_bridge.agent_spawner import AgentSpawner, WorkerInfo, runtime_module_for_harness
from build_bridge.agent_store import AgentStore


@pytest.fixture
def store(tmp_path: Path) -> AgentStore:
    s = AgentStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def spawner(store: AgentStore) -> AgentSpawner:
    return AgentSpawner(store=store, agent_port=19999)


class TestSpawn:
    async def test_spawn_creates_channel_in_db(self, spawner, store):
        worker = await spawner.spawn(
            channel_id="ch_test1",
            harness="claude-code",
            model="claude-sonnet-4-20250514",
        )

        assert worker.channel_id == "ch_test1"
        assert worker.harness == "claude-code"
        assert worker.model == "claude-sonnet-4-20250514"
        assert worker.agent_id.startswith("agt_")

        # Channel should exist in DB.
        channel = store.get_channel("ch_test1")
        assert channel is not None
        assert channel.harness == "claude-code"

        await spawner.stop("ch_test1")

    async def test_spawn_with_system_prompt(self, spawner, store):
        worker = await spawner.spawn(
            channel_id="ch_prompt",
            harness="claude-code",
            model="claude-sonnet-4-20250514",
            system_prompt="You are a code reviewer.",
        )

        assert worker.system_prompt == "You are a code reviewer."

        channel = store.get_channel("ch_prompt")
        assert channel is not None
        assert channel.system_prompt == "You are a code reviewer."

        await spawner.stop("ch_prompt")

    async def test_spawn_with_working_directory(self, spawner, store, tmp_path):
        wd = str(tmp_path / "project")
        (tmp_path / "project").mkdir()
        worker = await spawner.spawn(
            channel_id="ch_wd",
            harness="claude-code",
            model="claude-sonnet-4-20250514",
            working_directory=wd,
        )

        assert worker.working_directory == wd
        await spawner.stop("ch_wd")

    async def test_spawn_replaces_existing(self, spawner):
        worker1 = await spawner.spawn(
            channel_id="ch_replace",
            harness="claude-code",
            model="claude-sonnet-4-20250514",
        )
        pid1 = worker1.pid

        worker2 = await spawner.spawn(
            channel_id="ch_replace",
            harness="claude-code",
            model="claude-sonnet-4-20250514",
        )
        pid2 = worker2.pid

        # Different processes.
        assert pid1 != pid2
        assert len(spawner.workers) == 1

        await spawner.stop("ch_replace")

    async def test_spawn_generates_unique_agent_ids(self, spawner):
        w1 = await spawner.spawn("ch_a", "claude-code", "claude-sonnet-4-20250514")
        w2 = await spawner.spawn("ch_b", "claude-code", "claude-sonnet-4-20250514")

        assert w1.agent_id != w2.agent_id

        await spawner.stop_all()

    async def test_spawn_codex_uses_codex_runtime_module(self, spawner, monkeypatch):
        captured: dict[str, Any] = {}

        class DummyProcess:
            pid = 12345
            returncode = None
            stdout = None
            stderr = None

            def terminate(self):
                self.returncode = 0

            def kill(self):
                self.returncode = -9

            async def wait(self):
                self.returncode = 0
                return 0

        async def fake_exec(*cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return DummyProcess()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        worker = await spawner.spawn("ch_codex", "codex", "gpt-5.4")
        assert worker.harness == "codex"
        assert "build_bridge.codex_agent" in captured["cmd"]
        await spawner.stop("ch_codex")


class TestStop:
    async def test_stop_running_worker(self, spawner):
        await spawner.spawn("ch_stop", "claude-code", "claude-sonnet-4-20250514")
        assert spawner.is_running("ch_stop")

        stopped = await spawner.stop("ch_stop")
        assert stopped is True
        assert not spawner.is_running("ch_stop")

    async def test_stop_nonexistent_returns_false(self, spawner):
        stopped = await spawner.stop("ch_nonexistent")
        assert stopped is False

    async def test_stop_all(self, spawner):
        await spawner.spawn("ch_s1", "claude-code", "claude-sonnet-4-20250514")
        await spawner.spawn("ch_s2", "claude-code", "claude-sonnet-4-20250514")
        assert len(spawner.workers) == 2

        await spawner.stop_all()
        assert len(spawner.workers) == 0

    async def test_stop_updates_channel_status(self, spawner, store):
        await spawner.spawn("ch_status", "claude-code", "claude-sonnet-4-20250514")
        await spawner.stop("ch_status")

        channel = store.get_channel("ch_status")
        assert channel is not None
        assert channel.status == "closed"


class TestRestart:
    async def test_restart_existing(self, spawner):
        w1 = await spawner.spawn("ch_restart", "claude-code", "claude-sonnet-4-20250514")
        pid1 = w1.pid

        w2 = await spawner.restart("ch_restart")
        assert w2 is not None
        assert w2.pid != pid1
        assert w2.harness == "claude-code"

        await spawner.stop("ch_restart")

    async def test_restart_from_db(self, spawner, store):
        """Restart a channel that has DB record but no running worker."""
        store.create_channel(
            channel_id="ch_db_restart",
            agent_id="agt_old",
            harness="claude-code",
            model="claude-sonnet-4-20250514",
            system_prompt="Test prompt",
        )

        w = await spawner.restart("ch_db_restart")
        assert w is not None
        assert w.harness == "claude-code"

        await spawner.stop("ch_db_restart")

    async def test_restart_nonexistent_returns_none(self, spawner):
        result = await spawner.restart("ch_nonexistent")
        assert result is None


class TestListWorkers:
    async def test_list_empty(self, spawner):
        workers = spawner.list_workers()
        assert workers == []

    async def test_list_with_workers(self, spawner):
        await spawner.spawn("ch_list1", "claude-code", "claude-sonnet-4-20250514")
        await spawner.spawn("ch_list2", "codex", "o4-mini")

        workers = spawner.list_workers()
        assert len(workers) == 2

        channel_ids = {w["channel_id"] for w in workers}
        assert channel_ids == {"ch_list1", "ch_list2"}

        for w in workers:
            assert "agent_id" in w
            assert "harness" in w
            assert "model" in w
            assert "pid" in w
            assert "running" in w

        await spawner.stop_all()


class TestIsRunning:
    async def test_not_running_for_unknown(self, spawner):
        assert not spawner.is_running("ch_unknown")

    async def test_running_after_spawn(self, spawner):
        await spawner.spawn("ch_run", "claude-code", "claude-sonnet-4-20250514")
        assert spawner.is_running("ch_run")
        await spawner.stop("ch_run")

    async def test_not_running_after_stop(self, spawner):
        await spawner.spawn("ch_stopped", "claude-code", "claude-sonnet-4-20250514")
        await spawner.stop("ch_stopped")
        assert not spawner.is_running("ch_stopped")


class TestHarnessRuntimeDispatch:
    def test_known_harness_modules(self):
        assert runtime_module_for_harness("claude-code") == "build_bridge.build_agent"
        assert runtime_module_for_harness("codex") == "build_bridge.codex_agent"


class TestSnapshotAndAdopt:
    """Covers --keep-agents: persist workers across daemon re-exec and
    re-attach them via PID polling (process handle is lost across execv)."""

    async def test_snapshot_round_trip_adopts_live_pid(self, spawner, store, tmp_path):
        # Spawn a long-running dummy process we can legitimately own the PID of.
        # (The real agent runtime requires a running WS server — out of scope here.)
        proc = await asyncio.create_subprocess_exec(
            "sleep", "30",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            store.create_channel(
                channel_id="ch_adopt", agent_id="agt_adopt",
                harness="claude-code", model="claude-sonnet-4-20250514",
            )
            spawner._workers["ch_adopt"] = WorkerInfo(
                channel_id="ch_adopt", agent_id="agt_adopt",
                harness="claude-code", model="claude-sonnet-4-20250514",
                system_prompt="", working_directory="",
                pid=proc.pid, process=proc,
            )

            snapshot_path = tmp_path / "snap.json"
            count = spawner.snapshot_to_disk(snapshot_path)
            assert count == 1
            assert snapshot_path.exists()

            # Simulate a fresh daemon process: new spawner reads the snapshot.
            fresh = AgentSpawner(store=store, agent_port=19999)
            adopted = fresh.rehydrate_from_snapshot(snapshot_path)
            assert adopted == 1
            assert not snapshot_path.exists()  # consumed
            assert fresh.is_running("ch_adopt")  # alive via kill(pid, 0)

            worker = fresh._workers["ch_adopt"]
            assert worker.process is None
            assert worker.pid == proc.pid
            assert worker.agent_id == "agt_adopt"
        finally:
            proc.terminate()
            await proc.wait()
            # Cancel any monitor tasks the fresh spawner started.
            for task in list(fresh._monitor_tasks.values()):
                task.cancel()

    async def test_rehydrate_skips_dead_pids(self, spawner, store, tmp_path):
        # A guaranteed-dead PID: spawn a short-lived process, wait for exit.
        proc = await asyncio.create_subprocess_exec(
            "true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        dead_pid = proc.pid

        snapshot_path = tmp_path / "snap.json"
        import json as _json
        snapshot_path.write_text(_json.dumps([{
            "channel_id": "ch_dead", "agent_id": "agt_dead",
            "harness": "claude-code", "model": "m",
            "system_prompt": "", "working_directory": "",
            "effort": "", "auto_approve_tools": False,
            "pid": dead_pid,
        }]))

        adopted = spawner.rehydrate_from_snapshot(snapshot_path)
        assert adopted == 0
        assert "ch_dead" not in spawner._workers
        assert not snapshot_path.exists()

    async def test_rehydrate_missing_snapshot_is_noop(self, spawner, tmp_path):
        adopted = spawner.rehydrate_from_snapshot(tmp_path / "nope.json")
        assert adopted == 0
