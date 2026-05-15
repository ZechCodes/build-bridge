"""Terminal namespace — RELAY_PROTOCOL.md §5.8.

Browser-terminal handlers:

- `terminal_exec`: streams stdout/stderr of a shell command in batches.
- `terminal_kill`: SIGKILL the running process on a channel.
- `terminal_complete`: bash tab-completion candidates.

The per-channel registry of running processes is kept on the facade
(`facade._terminal_procs`) so `terminal_kill` can reach the live proc the
streaming task spawned in `terminal_exec`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from typing import Any

from build_bridge import relay_protocol as proto
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


# Marker the wrapping shell emits so we can recover the cwd after the
# command runs (so `cd foo` actually moves the terminal's cwd).
_SENTINEL = "__BUILD_CWD__"

# Streaming batch parameters: flush every 100 ms OR when the buffer hits
# 16 KB, whichever comes first.
_BATCH_INTERVAL_S = 0.1
_BATCH_MAX_BYTES = 16_384

# Kill any subprocess that produces no output for this long — almost
# certainly an interactive/TUI app that won't terminate here.
_NO_OUTPUT_TIMEOUT_S = 30.0


def _resolve_terminal_cwd(ctx: HandlerContext, channel_id: str, cwd: str) -> str:
    """Return the cwd for a terminal command, expanding ~/env vars."""
    if not cwd and ctx.agent_server:
        ch = ctx.agent_server.store.get_channel(channel_id)
        if ch and ch.working_directory:
            cwd = ch.working_directory
    if not cwd:
        cwd = os.getcwd()
    return os.path.expanduser(os.path.expandvars(cwd))


async def handle_terminal_exec(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Execute a shell command and stream output back to the browser."""
    # validator: channel_id + command (non-empty) required.
    channel_id = payload["channel_id"]
    command = payload["command"].strip()
    command_id = payload.get("command_id", "")
    cwd = payload.get("cwd", "")

    def _term_payload(**fields: Any) -> dict[str, Any]:
        """Build a terminal_output payload with channel_id and command_id."""
        p: dict[str, Any] = {"action": proto.TERMINAL_OUTPUT, "channel_id": channel_id}
        if command_id:
            p["command_id"] = command_id
        p.update(fields)
        return p

    cwd = _resolve_terminal_cwd(ctx, channel_id, cwd)

    # Wrap command to capture resulting cwd after execution.
    # NOTE: Do NOT wrap {command} in parentheses — that creates a subshell
    # which prevents `cd` from propagating to the pwd capture below.
    wrapped = (
        f'cd {shlex.quote(cwd)} && {command}\n__exit=$?\n'
        f'echo "{_SENTINEL}$(pwd)"\nexit $__exit'
    )

    try:
        proc = await asyncio.create_subprocess_shell(
            wrapped,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd if os.path.isdir(cwd) else None,
        )
        ctx.terminal_procs[channel_id] = (proc, command_id)
    except Exception as exc:
        await ctx.send_frame(session, ws, payload=_term_payload(
            data=f"Failed to execute: {exc}\n", done=True, exit_code=1, cwd=cwd,
        ))
        return

    collected_out: list[str] = []
    _batch_buf: list[str] = []
    _batch_lock = asyncio.Lock()

    async def _flush_batch() -> None:
        async with _batch_lock:
            if not _batch_buf:
                return
            data = "".join(_batch_buf)
            _batch_buf.clear()
        # Cap size — keep the tail (most recent output matters).
        if len(data) > _BATCH_MAX_BYTES:
            data = data[-_BATCH_MAX_BYTES:]
        await ctx.send_frame(session, ws, payload=_term_payload(
            data=data, done=False,
        ))

    async def _batch_timer() -> None:
        try:
            while True:
                await asyncio.sleep(_BATCH_INTERVAL_S)
                await _flush_batch()
        except asyncio.CancelledError:
            await _flush_batch()

    _output_stalled = False

    async def _read_output() -> None:
        nonlocal _output_stalled
        while True:
            try:
                chunk = await asyncio.wait_for(
                    proc.stdout.read(4096), timeout=_NO_OUTPUT_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                _output_stalled = True
                log.warning(
                    "No output for %.0fs on channel %s, killing process",
                    _NO_OUTPUT_TIMEOUT_S, channel_id[:8],
                )
                proc.kill()
                return
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            collected_out.append(text)
            async with _batch_lock:
                _batch_buf.append(text)

    timer_task = asyncio.create_task(_batch_timer())
    try:
        await _read_output()
        exit_code = await proc.wait()
    finally:
        timer_task.cancel()
        try:
            await timer_task
        except asyncio.CancelledError:
            pass
        ctx.terminal_procs.pop(channel_id, None)

    if _output_stalled:
        await ctx.send_frame(session, ws, payload=_term_payload(
            data=(
                "\r\nError: Command produced no output for 30 seconds and was killed. "
                "This may be an interactive/TUI application that requires a full "
                "terminal emulator.\r\n"
            ),
            done=True, exit_code=1, cwd=cwd,
        ))
        return

    # Extract cwd from sentinel line.
    result_cwd = cwd
    for line in reversed(collected_out):
        stripped = line.strip()
        if stripped.startswith(_SENTINEL):
            result_cwd = stripped[len(_SENTINEL):]
            break

    await ctx.send_frame(session, ws, payload=_term_payload(
        done=True, exit_code=exit_code, cwd=result_cwd,
    ))


async def handle_terminal_kill(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Kill the running terminal process on a channel, then send a done frame."""
    channel_id = payload.get("channel_id", "")
    log.info("Terminal kill requested for channel %s", channel_id[:8])
    entry = ctx.terminal_procs.pop(channel_id, None)
    if entry:
        proc, cmd_id = entry
        if proc.returncode is None:
            log.info(
                "Killing terminal process on channel %s (pid=%s)",
                channel_id[:8], proc.pid,
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    else:
        cmd_id = payload.get("command_id", "")

    # Always send a done frame to unstick the browser terminal.
    done_payload: dict[str, Any] = {
        "action": proto.TERMINAL_OUTPUT,
        "channel_id": channel_id,
        "data": "^C\r\n",
        "done": True,
        "exit_code": 130,
        "cwd": "",
    }
    if cmd_id:
        done_payload["command_id"] = cmd_id
    await ctx.send_frame(session, ws, payload=done_payload)


async def handle_terminal_complete(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Return tab-completion candidates for the browser terminal."""
    # validator: channel_id + partial + line required.
    channel_id = payload["channel_id"]
    partial = payload["partial"]
    line = payload["line"]
    cwd = payload.get("cwd", "")

    cwd = _resolve_terminal_cwd(ctx, channel_id, cwd)

    # Determine completion type: if the partial is the first word on the
    # line (i.e., it's a command), complete commands + files/dirs;
    # otherwise just files/dirs.
    line_before = line.strip()
    words = line_before.split()
    is_command_position = len(words) <= 1 and partial == line_before

    candidates: list[str] = []
    try:
        if is_command_position:
            script = (
                f'cd {shlex.quote(cwd)} 2>/dev/null; '
                f'compgen -c -- {shlex.quote(partial)} 2>/dev/null; '
                f'compgen -f -- {shlex.quote(partial)} 2>/dev/null'
            )
        else:
            script = (
                f'cd {shlex.quote(cwd)} 2>/dev/null; '
                f'compgen -f -- {shlex.quote(partial)} 2>/dev/null'
            )

        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd if os.path.isdir(cwd) else None,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
        raw = stdout.decode("utf-8", errors="replace").strip()
        if raw:
            seen: set[str] = set()
            for entry in raw.splitlines():
                entry = entry.strip()
                if entry and entry not in seen:
                    seen.add(entry)
                    candidates.append(entry)

        # Mark directories with trailing slash.
        enriched: list[str] = []
        for c in candidates:
            full = os.path.join(cwd, c) if not os.path.isabs(c) else c
            if os.path.isdir(full):
                enriched.append(c.rstrip("/") + "/")
            else:
                enriched.append(c)
        candidates = enriched

    except (asyncio.TimeoutError, Exception) as exc:
        log.debug("Tab completion failed: %s", exc)
        candidates = []

    # Cap results to avoid huge payloads.
    candidates = candidates[:100]

    await ctx.send_frame(session, ws, payload={
        "action": proto.TERMINAL_COMPLETIONS,
        "channel_id": channel_id,
        "partial": partial,
        "completions": candidates,
    })
