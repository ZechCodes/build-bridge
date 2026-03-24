"""Singleton process manager for the Build device client.

Provides:
- SingletonLock: flock-based singleton enforcement with pidfile
- DaemonContext: shared state between control plane and async_main
- ControlServer: Unix domain socket server for CLI commands
- run_daemon / main_with_watchdog: daemon lifecycle with crash recovery
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import signal
import socket
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from build_client.agent_server import AgentServer
    from build_client.agent_spawner import AgentSpawner
    from build_client.agent_store import AgentStore

log = logging.getLogger(__name__)

RUNTIME_DIR = Path.home() / ".config" / "build"
PIDFILE = RUNTIME_DIR / "build-device.pid"
LOCKFILE = RUNTIME_DIR / "build-device.lock"
SOCKFILE = RUNTIME_DIR / "build-device.sock"

# Watchdog parameters.
INITIAL_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0
HEALTHY_THRESHOLD_S = 60.0  # Reset backoff after this many seconds of healthy running.


# ---------------------------------------------------------------------------
# Singleton lock
# ---------------------------------------------------------------------------


class SingletonLock:
    """Flock-based singleton enforcement with pidfile."""

    def __init__(
        self,
        lockfile: Path = LOCKFILE,
        pidfile: Path = PIDFILE,
    ) -> None:
        self._lockfile = lockfile
        self._pidfile = pidfile
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Try to acquire the singleton lock.

        Returns True on success, False if another instance holds the lock.
        """
        self._lockfile.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._lockfile), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._fd)
            self._fd = None
            return False
        # Write PID.
        self._pidfile.write_text(str(os.getpid()))
        return True

    def release(self) -> None:
        """Release the lock and clean up files."""
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        self._pidfile.unlink(missing_ok=True)
        self._lockfile.unlink(missing_ok=True)

    @staticmethod
    def read_pid() -> int | None:
        """Read the PID from the pidfile. Returns None if missing or invalid."""
        try:
            return int(PIDFILE.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Daemon context — shared state
# ---------------------------------------------------------------------------


@dataclass
class DaemonContext:
    """Shared state between the daemon control plane and async_main."""

    agent_spawner: AgentSpawner | None = None
    agent_server: AgentServer | None = None
    agent_store: AgentStore | None = None
    relay_connected: bool = False
    started_at: float = field(default_factory=time.time)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    keep_agents_on_stop: bool = False
    restart_requested: bool = False


class _RestartRequested(Exception):
    """Raised to signal the watchdog to restart the daemon."""


# ---------------------------------------------------------------------------
# Wire protocol — length-prefixed JSON
# ---------------------------------------------------------------------------


async def _send_msg(writer: asyncio.StreamWriter, obj: dict[str, Any]) -> None:
    data = json.dumps(obj).encode()
    writer.write(struct.pack("!I", len(data)) + data)
    await writer.drain()


async def _recv_msg(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    header = await reader.readexactly(4)
    length = struct.unpack("!I", header)[0]
    if length > 1_000_000:  # 1 MB safety limit.
        return None
    data = await reader.readexactly(length)
    return json.loads(data)


# ---------------------------------------------------------------------------
# Control server
# ---------------------------------------------------------------------------


class ControlServer:
    """Unix domain socket server for CLI control commands."""

    def __init__(self, ctx: DaemonContext) -> None:
        self._ctx = ctx
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        SOCKFILE.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(SOCKFILE),
        )
        os.chmod(str(SOCKFILE), 0o600)
        log.info("Control server listening on %s", SOCKFILE)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        SOCKFILE.unlink(missing_ok=True)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            msg = await _recv_msg(reader)
            if not msg:
                return
            response = await self._dispatch(msg)
            await _send_msg(writer, response)
        except Exception as exc:
            try:
                await _send_msg(writer, {"ok": False, "error": str(exc)})
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        cmd = msg.get("cmd")
        handlers = {
            "ping": self._cmd_ping,
            "status": self._cmd_status,
            "stop": self._cmd_stop,
            "agents": self._cmd_agents,
            "agent_stop": self._cmd_agent_stop,
            "agent_restart": self._cmd_agent_restart,
        }
        handler = handlers.get(cmd)
        if not handler:
            return {"ok": False, "error": f"unknown command: {cmd}"}
        return await handler(msg)

    async def _cmd_ping(self, _msg: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "pid": os.getpid(),
            "uptime_s": time.time() - self._ctx.started_at,
        }

    async def _cmd_status(self, _msg: dict[str, Any]) -> dict[str, Any]:
        agent_count = 0
        if self._ctx.agent_spawner:
            agent_count = len(self._ctx.agent_spawner.workers)
        return {
            "ok": True,
            "pid": os.getpid(),
            "uptime_s": time.time() - self._ctx.started_at,
            "relay_connected": self._ctx.relay_connected,
            "agent_count": agent_count,
            "started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.localtime(self._ctx.started_at),
            ),
        }

    async def _cmd_stop(self, msg: dict[str, Any]) -> dict[str, Any]:
        kill_agents = msg.get("kill_agents", True)
        restart = msg.get("restart", False)
        self._ctx.keep_agents_on_stop = not kill_agents
        self._ctx.restart_requested = restart
        self._ctx.shutdown_event.set()
        return {"ok": True}

    async def _cmd_agents(self, _msg: dict[str, Any]) -> dict[str, Any]:
        if not self._ctx.agent_spawner:
            return {"ok": True, "agents": []}
        workers = self._ctx.agent_spawner.list_workers()
        return {"ok": True, "agents": workers}

    async def _cmd_agent_stop(self, msg: dict[str, Any]) -> dict[str, Any]:
        channel = msg.get("channel", "")
        if not channel or not self._ctx.agent_spawner:
            return {"ok": False, "error": "channel required"}
        stopped = await self._ctx.agent_spawner.stop(channel)
        if stopped:
            return {"ok": True}
        return {"ok": False, "error": f"no agent on channel {channel[:8]}"}

    async def _cmd_agent_restart(self, msg: dict[str, Any]) -> dict[str, Any]:
        channel = msg.get("channel", "")
        if not channel or not self._ctx.agent_spawner:
            return {"ok": False, "error": "channel required"}
        worker = await self._ctx.agent_spawner.restart(channel)
        if worker:
            return {
                "ok": True,
                "agent": {
                    "channel_id": worker.channel_id,
                    "agent_id": worker.agent_id,
                    "harness": worker.harness,
                    "model": worker.model,
                    "pid": worker.pid,
                },
            }
        return {"ok": False, "error": f"no agent config for channel {channel[:8]}"}


# ---------------------------------------------------------------------------
# Daemon lifecycle
# ---------------------------------------------------------------------------


def _is_socket_alive() -> bool:
    """Check if the control socket is active (another daemon is running)."""
    if not SOCKFILE.exists():
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(str(SOCKFILE))
        s.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False


async def run_daemon(
    base_url: str,
    reset: bool = False,
    agent_port: int = 9783,
) -> None:
    """Run the device client as a managed daemon.

    Acquires singleton lock, starts control server, runs async_main.
    Raises _RestartRequested if the user requested a restart.
    """
    lock = SingletonLock()

    if not lock.acquire():
        pid = SingletonLock.read_pid()
        raise SystemExit(
            f"Another instance is already running (PID {pid}). "
            "Use 'build-device stop' to stop it first."
        )

    # Clean up stale socket.
    if SOCKFILE.exists() and not _is_socket_alive():
        SOCKFILE.unlink(missing_ok=True)

    ctx = DaemonContext()
    control = ControlServer(ctx)

    # Register signal handlers for graceful shutdown.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, ctx.shutdown_event.set)

    try:
        await control.start()

        # Import here to avoid circular imports.
        from build_client.cli import async_main
        await async_main(base_url, reset=reset, agent_port=agent_port, daemon_ctx=ctx)

        if ctx.restart_requested:
            raise _RestartRequested()
    finally:
        await control.stop()
        lock.release()
        # Remove signal handlers.
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


async def main_with_watchdog(
    base_url: str,
    reset: bool = False,
    agent_port: int = 9783,
) -> None:
    """Run the daemon with automatic restart on crash."""
    backoff = INITIAL_BACKOFF_S

    while True:
        start = time.monotonic()
        try:
            await run_daemon(base_url, reset=reset, agent_port=agent_port)
            break  # Clean exit — don't restart.
        except _RestartRequested:
            log.info("Restart requested, restarting...")
            backoff = INITIAL_BACKOFF_S
            # Don't reset on subsequent runs.
            reset = False
            continue
        except SystemExit:
            raise
        except KeyboardInterrupt:
            break
        except Exception as exc:
            elapsed = time.monotonic() - start
            if elapsed > HEALTHY_THRESHOLD_S:
                backoff = INITIAL_BACKOFF_S
            log.error(
                "Daemon crashed after %.1fs: %s. Restarting in %.0fs...",
                elapsed, exc, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
            # Don't reset on subsequent runs.
            reset = False
