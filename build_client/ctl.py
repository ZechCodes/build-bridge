"""Control CLI client for the Build device client daemon.

Communicates with the running daemon over a Unix domain socket using
length-prefixed JSON messages.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
from pathlib import Path
from typing import Any

from build_client.daemon import PIDFILE, SOCKFILE


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from a socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


def send_command(cmd: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    """Send a command to the daemon and return the response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(SOCKFILE))
    except (FileNotFoundError, ConnectionRefusedError):
        raise ConnectionError("Daemon is not running")

    # Send length-prefixed JSON.
    data = json.dumps(cmd).encode()
    sock.sendall(struct.pack("!I", len(data)) + data)

    # Read response.
    header = _recv_exact(sock, 4)
    length = struct.unpack("!I", header)[0]
    resp_data = _recv_exact(sock, length)
    sock.close()
    return json.loads(resp_data)


def is_running() -> tuple[bool, int | None]:
    """Check if the daemon is running.

    Returns (running, pid). Tries socket first, falls back to pidfile.
    """
    if SOCKFILE.exists():
        try:
            resp = send_command({"cmd": "ping"}, timeout=3.0)
            return True, resp.get("pid")
        except (ConnectionError, OSError):
            pass

    # Fall back to pidfile.
    try:
        pid = int(PIDFILE.read_text().strip())
        os.kill(pid, 0)  # Check if process exists.
        return True, pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False, None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_uptime(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m}m"


def print_status(resp: dict[str, Any]) -> None:
    pid = resp.get("pid", "?")
    uptime = _format_uptime(resp.get("uptime_s", 0))
    relay = "connected" if resp.get("relay_connected") else "disconnected"
    agents = resp.get("agent_count", 0)
    print(f"Build device client: running (PID {pid})")
    print(f"  Uptime:  {uptime}")
    print(f"  Relay:   {relay}")
    print(f"  Agents:  {agents} active")


def print_agents(resp: dict[str, Any]) -> None:
    agents = resp.get("agents", [])
    if not agents:
        print("No agents running.")
        return
    # Header.
    print(f"{'CHANNEL':<12} {'HARNESS':<14} {'MODEL':<22} {'PID':<8} {'STATUS'}")
    print("-" * 70)
    for a in agents:
        ch = str(a.get("channel_id", ""))[:8]
        harness = a.get("harness", "?")
        model = a.get("model", "?")
        pid = a.get("pid", "?")
        status = a.get("status", "?")
        print(f"{ch:<12} {harness:<14} {model:<22} {str(pid):<8} {status}")


def print_not_running() -> None:
    print("Build device client is not running.", file=sys.stderr)
    sys.exit(1)
