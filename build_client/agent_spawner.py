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
import uuid
from dataclasses import dataclass
from typing import Any

from build_client.agent_store import AgentStore

log = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Tracks a running agent worker process."""

    channel_id: str
    agent_id: str
    harness: str
    model: str
    system_prompt: str
    working_directory: str
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
            )
        else:
            # Update existing channel with new agent_id.
            self._store.db.execute(
                "UPDATE agent_channels SET agent_id = ?, harness = ?, model = ?, "
                "system_prompt = ?, status = 'active' WHERE id = ?",
                (agent_id, harness, model, system_prompt, channel_id),
            )
            self._store.db.commit()

        # Build the command.
        cmd = [
            sys.executable, "-m", "build_client.build_agent",
            "--port", str(self._agent_port),
            "--host", self._agent_host,
            "--model", model,
            "--agent-id", agent_id,
        ]
        if working_directory:
            cmd.extend(["--working-directory", working_directory])

        # Set up environment.
        env = os.environ.copy()
        env["BUILD_AGENT_PORT"] = str(self._agent_port)
        env["BUILD_AGENT_HOST"] = self._agent_host

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

    async def stop(self, channel_id: str) -> bool:
        """Stop the agent process on a channel. Returns True if stopped."""
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
        self._store.update_channel_status(channel_id, "closed")

        return True

    async def stop_all(self) -> None:
        """Stop all running agent processes."""
        channel_ids = list(self._workers.keys())
        for channel_id in channel_ids:
            await self.stop(channel_id)

    # -----------------------------------------------------------------
    # Restart
    # -----------------------------------------------------------------

    async def restart(self, channel_id: str) -> WorkerInfo | None:
        """Restart the agent on a channel using its stored config."""
        worker = self._workers.get(channel_id)
        if not worker:
            # Try to restore from DB.
            channel = self._store.get_channel(channel_id)
            if not channel:
                return None
            return await self.spawn(
                channel_id=channel_id,
                harness=channel.harness,
                model=channel.model,
                system_prompt=channel.system_prompt,
            )

        # Restart with same config.
        return await self.spawn(
            channel_id=channel_id,
            harness=worker.harness,
            model=worker.model,
            system_prompt=worker.system_prompt,
            working_directory=worker.working_directory,
        )

    # -----------------------------------------------------------------
    # Monitor
    # -----------------------------------------------------------------

    async def _monitor_worker(self, worker: WorkerInfo) -> None:
        """Monitor a worker process and log when it exits."""
        if not worker.process:
            return

        try:
            returncode = await worker.process.wait()

            # Read any stderr output.
            stderr = b""
            if worker.process.stderr:
                stderr = await worker.process.stderr.read()

            if returncode == 0:
                log.info(
                    "Agent pid=%s on channel %s exited normally",
                    worker.pid, worker.channel_id[:8],
                )
            else:
                log.warning(
                    "Agent pid=%s on channel %s exited with code %s: %s",
                    worker.pid, worker.channel_id[:8], returncode,
                    stderr.decode(errors="replace")[:500],
                )

            # Update channel status.
            self._store.update_channel_status(
                worker.channel_id,
                "closed" if returncode == 0 else "error",
            )

        except asyncio.CancelledError:
            pass

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
