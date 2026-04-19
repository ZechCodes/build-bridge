"""Agent process spawner — manages build-agent worker processes per channel.

The device client spawns one agent process per channel. Each process runs
build_agent.py which connects back to the device client's WS server via
the AgentWrapper.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

from build_bridge.agent_store import AgentStore

log = logging.getLogger(__name__)

RESTART_BACKOFF_BASE_S = 1.0
RESTART_BACKOFF_MAX_S = 30.0
MAX_RESTART_ATTEMPTS = 5
RESTART_ATTEMPTS_RESET_AFTER_S = 300.0  # reset counter after 5 min of stability


_HARNESS_MODULES: dict[str, str] = {
    "claude-code": "build_bridge.build_agent",
    "codex": "build_bridge.codex_agent",
}


def runtime_module_for_harness(harness: str) -> str:
    """Return the runtime entrypoint module for a harness."""
    try:
        return _HARNESS_MODULES[harness]
    except KeyError as exc:
        raise ValueError(f"Unsupported harness: {harness}") from exc


@dataclass
class WorkerInfo:
    """Tracks a running agent worker process."""

    channel_id: str
    agent_id: str
    harness: str
    model: str
    system_prompt: str
    working_directory: str
    effort: str = ""
    auto_approve_tools: bool = False
    pid: int | None = None
    process: asyncio.subprocess.Process | None = None


class AgentSpawner:
    """Manages agent worker processes per channel.

    Spawns build-agent subprocesses, tracks their PIDs, handles restarts,
    and cleans up on shutdown.
    """

    def __init__(
        self,
        store: AgentStore,
        agent_port: int = 9783,
        agent_host: str = "127.0.0.1",
    ) -> None:
        self._store = store
        self._agent_port = agent_port
        self._agent_host = agent_host
        self._workers: dict[str, WorkerInfo] = {}  # channel_id -> WorkerInfo
        self._monitor_tasks: dict[str, asyncio.Task] = {}
        # Tracks auto-restart attempts per channel: (attempts, last_restart_ts).
        # Survives WorkerInfo replacement so counter persists across respawns.
        self._restart_state: dict[str, tuple[int, float]] = {}

    @property
    def workers(self) -> dict[str, WorkerInfo]:
        return dict(self._workers)

    def is_running(self, channel_id: str) -> bool:
        """Check if an agent is running on a channel."""
        worker = self._workers.get(channel_id)
        if not worker or not worker.process:
            return False
        return worker.process.returncode is None

    # -----------------------------------------------------------------
    # Spawn
    # -----------------------------------------------------------------

    async def spawn(
        self,
        channel_id: str,
        harness: str,
        model: str,
        system_prompt: str = "",
        working_directory: str = "",
        effort: str = "",
        auto_approve_tools: bool = False,
    ) -> WorkerInfo:
        """Spawn a build-agent process for a channel.

        Pre-creates the channel in the DB with a generated agent_id,
        then launches the agent subprocess which connects back to the
        device client's WS server.
        """
        # Stop existing worker if any.
        if channel_id in self._workers:
            await self.stop(channel_id)

        # Generate agent_id for this channel.
        agent_id = f"agt_{uuid.uuid4().hex[:8]}"

        # Pre-create the channel in the DB so the agent server can match it.
        existing = self._store.get_channel(channel_id)
        if not existing:
            self._store.create_channel(
                channel_id=channel_id,
                agent_id=agent_id,
                harness=harness,
                model=model,
                system_prompt=system_prompt,
                working_directory=working_directory,
                auto_approve_tools=auto_approve_tools,
            )
        else:
            # Update existing channel with new agent_id.
            self._store.update_channel_agent(
                channel_id, agent_id, harness, model, system_prompt,
                working_directory=working_directory,
            )

        # Build the command.
        runtime_module = runtime_module_for_harness(harness)
        cmd = [
            sys.executable, "-m", runtime_module,
            "--port", str(self._agent_port),
            "--host", self._agent_host,
            "--model", model,
            "--agent-id", agent_id,
        ]
        if working_directory:
            # Expand ~ and env vars so the agent subprocess gets a real path.
            resolved_wd = os.path.expanduser(os.path.expandvars(working_directory))
            cmd.extend(["--working-directory", resolved_wd])

        # Read effort from DB if not passed directly (user may have changed it via edit modal).
        channel = existing or self._store.get_channel(channel_id)
        effective_effort = effort or (channel.effort if channel else "")
        if effective_effort:
            cmd.extend(["--effort", effective_effort])

        # Resolve auto_approve_tools: spawn-time arg wins on fresh create;
        # DB value wins on re-spawn of existing channels (respects user edits).
        effective_auto_approve = (
            auto_approve_tools if not existing else (channel.auto_approve_tools if channel else False)
        )
        if effective_auto_approve:
            cmd.append("--auto-approve-tools")

        # Pass resume cursor if available (not cleared by reset).
        resume_cursor = channel.resume_cursor if channel else ""
        if resume_cursor:
            cmd.extend(["--resume-session", resume_cursor])

        # Set up environment.
        env = os.environ.copy()
        env["BUILD_AGENT_PORT"] = str(self._agent_port)
        env["BUILD_AGENT_HOST"] = self._agent_host

        # Remove CLAUDECODE env var so the agent subprocess isn't rejected
        # as a nested Claude Code session.
        env.pop("CLAUDECODE", None)

        # Ensure build_bridge is importable from any cwd by adding the
        # project root to PYTHONPATH.  sys.executable lives in the venv
        # but the package may not be pip-installed (uv run sets it up).
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{project_root}:{existing_pp}" if existing_pp else project_root

        log.info(
            "Spawning agent for channel %s (harness=%s, model=%s, agent_id=%s)",
            channel_id[:8], harness, model, agent_id[:8],
        )

        # Resolve working directory (expand ~ and env vars).
        cwd = None
        if working_directory:
            cwd = os.path.expanduser(os.path.expandvars(working_directory))

        # Spawn the process.
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        worker = WorkerInfo(
            channel_id=channel_id,
            agent_id=agent_id,
            harness=harness,
            model=model,
            system_prompt=system_prompt,
            working_directory=working_directory,
            effort=effective_effort,
            auto_approve_tools=effective_auto_approve,
            pid=process.pid,
            process=process,
        )
        self._workers[channel_id] = worker

        # Start a monitor task for this worker.
        self._monitor_tasks[channel_id] = asyncio.create_task(
            self._monitor_worker(worker)
        )

        log.info("Agent spawned: pid=%s, channel=%s", process.pid, channel_id[:8])
        return worker

    # -----------------------------------------------------------------
    # Stop
    # -----------------------------------------------------------------

    async def stop(self, channel_id: str, *, resumable: bool = False) -> bool:
        """Stop the agent process on a channel.

        If *resumable* is True, the channel is set to 'idle' so it will be
        re-spawned on the next startup. Otherwise it's set to 'closed'.
        """
        worker = self._workers.pop(channel_id, None)
        if not worker:
            return False

        # Cancel monitor task.
        task = self._monitor_tasks.pop(channel_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if worker.process and worker.process.returncode is None:
            log.info("Stopping agent pid=%s on channel %s", worker.pid, channel_id[:8])
            try:
                worker.process.terminate()
                try:
                    await asyncio.wait_for(worker.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    log.warning("Agent pid=%s didn't stop, killing", worker.pid)
                    worker.process.kill()
                    await worker.process.wait()
            except ProcessLookupError:
                pass

        # Update channel status.
        self._store.update_channel_status(channel_id, "idle" if resumable else "closed")

        # Clear auto-restart state — user-initiated stop should start fresh on next start.
        self._restart_state.pop(channel_id, None)

        return True

    async def stop_all(self, *, resumable: bool = False) -> None:
        """Stop all running agent processes.

        If *resumable* is True, channels are set to 'idle' so agents will be
        re-spawned on the next startup (used during daemon restart).
        """
        channel_ids = list(self._workers.keys())
        for channel_id in channel_ids:
            await self.stop(channel_id, resumable=resumable)

    # -----------------------------------------------------------------
    # Restart
    # -----------------------------------------------------------------

    async def restart(self, channel_id: str) -> WorkerInfo | None:
        """Restart the agent on a channel using its stored config."""
        worker = self._workers.get(channel_id)

        # Always read working_directory from DB — user may have updated it
        # via the edit modal after the worker was spawned.
        channel = self._store.get_channel(channel_id)

        if not worker and not channel:
            return None

        harness = worker.harness if worker else channel.harness
        model = worker.model if worker else channel.model
        system_prompt = worker.system_prompt if worker else channel.system_prompt
        working_directory = channel.working_directory if channel else (worker.working_directory if worker else "")

        return await self.spawn(
            channel_id=channel_id,
            harness=harness,
            model=model,
            system_prompt=system_prompt,
            working_directory=working_directory,
        )

    # -----------------------------------------------------------------
    # Monitor
    # -----------------------------------------------------------------

    async def _drain_stream(self, stream, label: str, channel_id: str) -> bytes:
        """Continuously drain a subprocess stream to prevent pipe buffer deadlock."""
        chunks = []
        try:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.debug("Error draining %s for channel %s", label, channel_id[:8])
        return b"".join(chunks)

    async def _monitor_worker(self, worker: WorkerInfo) -> None:
        """Monitor a worker process and log when it exits."""
        if not worker.process:
            return

        try:
            # Start draining stdout/stderr immediately to prevent pipe buffer
            # deadlock. If the agent writes more than the OS pipe buffer size
            # (~64KB) without these being read, the process blocks on write.
            drain_tasks = []
            if worker.process.stdout:
                drain_tasks.append(asyncio.create_task(
                    self._drain_stream(worker.process.stdout, "stdout", worker.channel_id)
                ))
            if worker.process.stderr:
                drain_tasks.append(asyncio.create_task(
                    self._drain_stream(worker.process.stderr, "stderr", worker.channel_id)
                ))

            returncode = await worker.process.wait()

            # Collect drained output.
            stdout = b""
            stderr = b""
            results = await asyncio.gather(*drain_tasks, return_exceptions=True)
            if len(results) >= 1 and isinstance(results[0], bytes):
                stdout = results[0]
            if len(results) >= 2 and isinstance(results[1], bytes):
                stderr = results[1]

            combined = (stdout + b"\n" + stderr).decode(errors="replace").strip()

            if returncode == 0:
                log.info(
                    "Agent pid=%s on channel %s exited normally. Output: %s",
                    worker.pid, worker.channel_id[:8],
                    combined[-1000:] if combined else "(none)",
                )
            else:
                log.warning(
                    "Agent pid=%s on channel %s exited with code %s: %s",
                    worker.pid, worker.channel_id[:8], returncode,
                    combined[-1000:] if combined else "(none)",
                )

            # Update channel status.
            self._store.update_channel_status(
                worker.channel_id,
                "closed" if returncode == 0 else "error",
            )

            # Auto-restart on unexpected non-zero exit.
            # User-initiated stops cancel this monitor task before process exit,
            # so they never reach this branch.
            if returncode != 0:
                self._maybe_schedule_auto_restart(worker.channel_id)

        except asyncio.CancelledError:
            pass

    def _maybe_schedule_auto_restart(self, channel_id: str) -> None:
        """Schedule an auto-restart with exponential backoff, up to MAX_RESTART_ATTEMPTS."""
        attempts, last_ts = self._restart_state.get(channel_id, (0, 0.0))
        now = time.time()
        # Reset counter if the agent ran stably for a while.
        if last_ts and now - last_ts > RESTART_ATTEMPTS_RESET_AFTER_S:
            attempts = 0
        if attempts >= MAX_RESTART_ATTEMPTS:
            log.error(
                "Agent on channel %s exited %d times in a row; giving up auto-restart.",
                channel_id[:8], attempts,
            )
            self._restart_state.pop(channel_id, None)
            return
        backoff = min(
            RESTART_BACKOFF_BASE_S * (2 ** attempts),
            RESTART_BACKOFF_MAX_S,
        )
        self._restart_state[channel_id] = (attempts + 1, now)
        log.warning(
            "Auto-restarting agent on channel %s (attempt %d) after %.1fs backoff",
            channel_id[:8], attempts + 1, backoff,
        )
        asyncio.create_task(self._auto_restart(channel_id, backoff))

    async def _auto_restart(self, channel_id: str, backoff: float) -> None:
        await asyncio.sleep(backoff)
        # Channel may have been deleted during backoff.
        channel = self._store.get_channel(channel_id)
        if not channel:
            self._restart_state.pop(channel_id, None)
            return
        try:
            await self.restart(channel_id)
        except Exception:
            log.exception("Auto-restart failed for channel %s", channel_id[:8])

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    def list_workers(self) -> list[dict[str, Any]]:
        """List all workers with their status."""
        result = []
        for channel_id, worker in self._workers.items():
            running = worker.process is not None and worker.process.returncode is None
            result.append({
                "channel_id": channel_id,
                "agent_id": worker.agent_id,
                "harness": worker.harness,
                "model": worker.model,
                "pid": worker.pid,
                "running": running,
                "working_directory": worker.working_directory,
            })
        return result
