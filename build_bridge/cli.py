"""CLI entry point for the Build device client.

Usage:
    build-device                     # Start the daemon (in background)
    build-device start [-f]          # Start the daemon (--foreground to run inline)
    build-device stop [--keep-agents]  # Stop the daemon
    build-device restart             # Restart the daemon
    build-device status              # Show daemon status
    build-device logs [-f] [-n N]    # Show daemon logs (--follow to tail)
    build-device agents              # List running agents
    build-device agent-stop <ch>     # Stop an agent
    build-device agent-restart <ch>  # Restart an agent
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from build_bridge.agent_server import AgentServer, DEFAULT_AGENT_PORT
from build_bridge.agent_spawner import (
    AgentSpawner,
    KEEP_AGENTS_ENV,
    WORKER_SNAPSHOT_PATH,
)
from build_bridge.agent_store import AgentStore
from build_bridge.auth import device_auth_flow
from build_bridge.complications import ComplicationRegistry
from build_bridge.config import load_config
from build_bridge.e2ee import E2EEHandler
from build_bridge.storage import MessageStore
from build_bridge.ws import run_connection

if TYPE_CHECKING:
    from build_bridge.daemon import DaemonContext

LOG_DIR = Path.home() / ".config" / "build" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

# Console handler (existing behavior).
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)

# Rotating file handler — device client logs.
from logging.handlers import RotatingFileHandler
_file_handler = RotatingFileHandler(
    LOG_DIR / "device.log",
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,
)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_console_handler, _file_handler],
)
log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://getbuild.ing"


async def async_main(
    base_url: str,
    reset: bool = False,
    agent_port: int = DEFAULT_AGENT_PORT,
    daemon_ctx: DaemonContext | None = None,
) -> None:
    """Main async entry point."""
    config = None if reset else load_config()

    if not config:
        config = await device_auth_flow(base_url)

    log.info("Device: %s (%s)", config.device_name, config.device_id)
    log.info("Server: %s", config.base_url)

    # Initialize stores.
    store = MessageStore()
    agent_store = AgentStore()

    # Initialize E2EE handler.
    handler = E2EEHandler(config, store)

    # Initialize agent server with E2EE broadcast callback.
    complications = ComplicationRegistry(
        broadcast=handler.broadcast_to_sessions,
        agent_store=agent_store,
    )
    agent_server = AgentServer(
        store=agent_store,
        broadcast=handler.broadcast_to_sessions,
        port=agent_port,
        e2ee_store=store,
        complications=complications,
    )

    # Initialize agent spawner for managing agent worker processes.
    agent_spawner = AgentSpawner(
        store=agent_store,
        agent_port=agent_port,
    )

    # Wire the agent server and spawner into the E2EE handler.
    handler.set_agent_server(agent_server)
    handler.set_agent_spawner(agent_spawner)

    # Populate daemon context if running under the process manager.
    if daemon_ctx:
        daemon_ctx.agent_spawner = agent_spawner
        daemon_ctx.agent_server = agent_server
        daemon_ctx.agent_store = agent_store

    try:
        # Start the local agent WS server.
        await agent_server.start()
        log.info("Agent server ready on port %s", agent_port)

        # If the previous daemon exited via `--keep-agents`, adopt the
        # surviving agent processes from the snapshot instead of spawning
        # fresh ones. Consume the env var so a subsequent crash-restart
        # (without --keep-agents) falls back to the normal respawn path.
        keep_agents_restart = bool(os.environ.pop(KEEP_AGENTS_ENV, None))
        if keep_agents_restart:
            adopted = agent_spawner.rehydrate_from_snapshot(WORKER_SNAPSHOT_PATH)
            log.info("Adopted %d surviving agent(s) after --keep-agents restart", adopted)
        else:
            # Re-spawn agents for channels that were active/idle before restart.
            active_channels = agent_store.list_resumable_channels()
            if active_channels:
                log.info("Re-spawning agents for %d active channel(s)", len(active_channels))
                for ch in active_channels:
                    try:
                        await agent_spawner.spawn(
                            channel_id=ch.id,
                            harness=ch.harness,
                            model=ch.model,
                            system_prompt=ch.system_prompt,
                            working_directory=ch.working_directory,
                        )
                    except Exception as exc:
                        log.error("Failed to re-spawn agent on channel %s: %s", ch.id[:8], exc)

        # Run the relay connection. If running under the daemon, make it
        # interruptible via the shutdown event.
        if daemon_ctx:
            daemon_ctx.relay_connected = True  # run_connection auto-reconnects.

            def _request_restart():
                daemon_ctx.restart_requested = True
                daemon_ctx.shutdown_event.set()

            relay_task = asyncio.create_task(
                run_connection(config, e2e_handler=handler.handle_message, on_restart=_request_restart)
            )
            shutdown_task = asyncio.create_task(daemon_ctx.shutdown_event.wait())
            done, pending = await asyncio.wait(
                [relay_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        else:
            await run_connection(config, e2e_handler=handler.handle_message)

    except KeyboardInterrupt:
        pass
    finally:
        keep_agents = daemon_ctx.keep_agents_on_stop if daemon_ctx else False
        restarting = daemon_ctx.restart_requested if daemon_ctx else False
        if not keep_agents:
            await agent_spawner.stop_all(resumable=restarting)
        else:
            log.info("Keeping agents running (--keep-agents)")
            if restarting:
                # Persist worker state and flag the re-exec so the next
                # daemon adopts these processes instead of spawning fresh
                # ones (which would race the survivors' WS reconnects).
                count = agent_spawner.snapshot_to_disk(WORKER_SNAPSHOT_PATH)
                os.environ[KEEP_AGENTS_ENV] = "1"
                log.info(
                    "Snapshotted %d agent(s) for adoption after re-exec", count,
                )
        # When keeping agents alive, skip the broadcast of agent.shutdown —
        # that would tell each agent wrapper to exit cleanly, defeating the
        # whole point. Agents just see their WS drop and auto-reconnect.
        await agent_server.stop(notify_agents=not keep_agents)
        agent_store.close()
        store.close()

    log.info("Disconnected.")


# ---------------------------------------------------------------------------
# CLI entry point — subcommand dispatcher
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build device client — manages the local device daemon.",
    )
    sub = parser.add_subparsers(dest="command")

    # start (also default when no subcommand given)
    p_start = sub.add_parser("start", help="Start the device client daemon")
    p_start.add_argument(
        "--url",
        default=os.environ.get("BUILD_URL", DEFAULT_BASE_URL),
        help=f"Build server URL (default: {DEFAULT_BASE_URL})",
    )
    p_start.add_argument("--reset", action="store_true", help="Re-run device auth")
    p_start.add_argument(
        "--agent-port", type=int,
        default=int(os.environ.get("BUILD_AGENT_PORT", str(DEFAULT_AGENT_PORT))),
        help=f"Agent WebSocket port (default: {DEFAULT_AGENT_PORT})",
    )
    p_start.add_argument(
        "--no-watchdog", action="store_true",
        help="Run without crash recovery watchdog",
    )
    p_start.add_argument(
        "-f", "--foreground", action="store_true",
        help="Run in the foreground (default: detach into background)",
    )

    # stop
    p_stop = sub.add_parser("stop", help="Stop the daemon")
    p_stop.add_argument(
        "--keep-agents", action="store_true",
        help="Keep agent processes running after stopping the daemon",
    )

    # restart
    p_restart = sub.add_parser("restart", help="Restart the daemon")
    p_restart.add_argument("--keep-agents", action="store_true")

    # status
    sub.add_parser("status", help="Show daemon status")

    # logs
    p_logs = sub.add_parser("logs", help="Show daemon logs")
    p_logs.add_argument(
        "-f", "--follow", action="store_true",
        help="Follow new log output as it is written",
    )
    p_logs.add_argument(
        "-n", "--lines", type=int, default=50,
        help="Number of trailing lines to show (default: 50)",
    )

    # agents
    sub.add_parser("agents", help="List running agents")

    # agent-stop
    p_as = sub.add_parser("agent-stop", help="Stop an agent")
    p_as.add_argument("channel", help="Channel ID (or prefix)")

    # agent-restart
    p_ar = sub.add_parser("agent-restart", help="Restart an agent")
    p_ar.add_argument("channel", help="Channel ID (or prefix)")

    args = parser.parse_args()
    command = args.command

    # Default to start when no subcommand given.
    if command is None:
        # Check for legacy flags on the bare command.
        args.url = os.environ.get("BUILD_URL", DEFAULT_BASE_URL)
        args.reset = False
        args.agent_port = int(os.environ.get("BUILD_AGENT_PORT", str(DEFAULT_AGENT_PORT)))
        args.no_watchdog = False
        args.foreground = False
        command = "start"

    if command == "start":
        _cmd_start(args)
    elif command == "stop":
        _cmd_stop(args)
    elif command == "restart":
        _cmd_restart(args)
    elif command == "status":
        _cmd_status()
    elif command == "logs":
        _cmd_logs(args)
    elif command == "agents":
        _cmd_agents()
    elif command == "agent-stop":
        _cmd_agent_control("agent_stop", args.channel)
    elif command == "agent-restart":
        _cmd_agent_control("agent_restart", args.channel)


def _cmd_start(args: argparse.Namespace) -> None:
    from build_bridge.daemon import main_with_watchdog, run_daemon
    from build_bridge.ctl import is_running
    from build_bridge.config import load_config

    running, pid = is_running()
    if running:
        print(f"Daemon is already running (PID {pid}).")
        sys.exit(1)

    # First-time setup needs interactive auth — can't background.
    needs_auth = args.reset or load_config() is None
    if needs_auth and not args.foreground:
        print("First run needs interactive device auth — staying in the foreground.")
        print("After auth, stop and re-run `start` to detach.")
        args.foreground = True

    if not args.foreground:
        _spawn_detached(args)
        return

    try:
        if args.no_watchdog:
            asyncio.run(run_daemon(args.url, reset=args.reset, agent_port=args.agent_port))
        else:
            asyncio.run(main_with_watchdog(args.url, reset=args.reset, agent_port=args.agent_port))
    except KeyboardInterrupt:
        pass


def _spawn_detached(args: argparse.Namespace) -> None:
    """Re-exec ourselves in the background with --foreground, then return."""
    import subprocess
    import time
    from build_bridge.ctl import is_running

    cmd = [
        sys.executable, "-m", "build_bridge.cli", "start", "--foreground",
        "--url", args.url,
        "--agent-port", str(args.agent_port),
    ]
    if args.no_watchdog:
        cmd.append("--no-watchdog")

    print("Starting daemon in background...", end=" ", flush=True)
    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    # Poll for ~5s for the daemon to come up.
    for _ in range(50):
        time.sleep(0.1)
        running, pid = is_running()
        if running:
            print(f"running (PID {pid}).")
            print(f"Logs: {LOG_DIR / 'device.log'}")
            print("Tail with: build-device logs -f")
            return
    print("daemon did not become ready within 5s.")
    print(f"Check logs: {LOG_DIR / 'device.log'}")
    sys.exit(1)


def _cmd_stop(args: argparse.Namespace) -> None:
    from build_bridge.ctl import send_command, is_running, print_not_running

    running, pid = is_running()
    if not running:
        print_not_running()

    kill_agents = not getattr(args, "keep_agents", False)
    print(f"Stopping daemon (PID {pid})...", end=" ", flush=True)
    try:
        send_command({"cmd": "stop", "kill_agents": kill_agents})
        print("done.")
    except ConnectionError:
        print("failed (connection lost).")
        sys.exit(1)


def _cmd_restart(args: argparse.Namespace) -> None:
    from build_bridge.ctl import send_command, is_running, print_not_running

    running, pid = is_running()
    if not running:
        print_not_running()

    kill_agents = not getattr(args, "keep_agents", False)
    print(f"Restarting daemon (PID {pid})...", end=" ", flush=True)
    try:
        send_command({"cmd": "stop", "kill_agents": kill_agents, "restart": True})
        print("done.")
    except ConnectionError:
        print("failed (connection lost).")
        sys.exit(1)


def _cmd_status() -> None:
    from build_bridge.ctl import send_command, is_running, print_status, print_not_running

    running, pid = is_running()
    if not running:
        print("Build device client: not running")
        return

    try:
        resp = send_command({"cmd": "status"})
        if resp.get("ok"):
            print_status(resp)
        else:
            print(f"Error: {resp.get('error')}")
    except ConnectionError:
        print(f"Build device client: running (PID {pid}) but not responding")


def _cmd_logs(args: argparse.Namespace) -> None:
    import shutil
    import subprocess

    log_path = LOG_DIR / "device.log"
    if not log_path.exists():
        print(f"No log file found at {log_path}", file=sys.stderr)
        sys.exit(1)

    tail = shutil.which("tail")
    if not tail:
        print("`tail` not found on PATH; cannot display logs.", file=sys.stderr)
        sys.exit(1)

    cmd = [tail, "-n", str(args.lines)]
    if args.follow:
        # -F follows the file across rotation (RotatingFileHandler renames).
        cmd.append("-F")
    cmd.append(str(log_path))

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


def _cmd_agents() -> None:
    from build_bridge.ctl import send_command, print_agents, print_not_running

    try:
        resp = send_command({"cmd": "agents"})
        if resp.get("ok"):
            print_agents(resp)
        else:
            print(f"Error: {resp.get('error')}")
    except ConnectionError:
        print_not_running()


def _cmd_agent_control(cmd: str, channel: str) -> None:
    from build_bridge.ctl import send_command, print_not_running

    try:
        resp = send_command({"cmd": cmd, "channel": channel})
        if resp.get("ok"):
            if cmd == "agent_restart":
                agent = resp.get("agent", {})
                print(f"Restarted agent on channel {channel[:8]} (PID {agent.get('pid')})")
            else:
                print(f"Stopped agent on channel {channel[:8]}")
        else:
            print(f"Error: {resp.get('error')}")
            sys.exit(1)
    except ConnectionError:
        print_not_running()


if __name__ == "__main__":
    main()
