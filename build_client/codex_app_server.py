"""Codex app-server transport over stdio JSON-RPC."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

NotificationHandler = Callable[[dict[str, Any]], Awaitable[None] | None]
RequestHandler = Callable[[str | int, dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


class CodexAppServerError(RuntimeError):
    """Raised when the Codex app-server transport fails."""


class CodexAppServerClient:
    """Minimal JSON-RPC client for `codex app-server --listen stdio://`."""

    def __init__(
        self,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        command: list[str] | None = None,
    ) -> None:
        self._env = env
        self._cwd = cwd
        self._command = command or ["codex", "app-server", "--listen", "stdio://"]
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._notifications: dict[str, NotificationHandler] = {}
        self._requests: dict[str, RequestHandler] = {}
        self._next_id = 1
        self._write_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def on_notification(self, method: str, handler: NotificationHandler) -> None:
        self._notifications[method] = handler

    def on_request(self, method: str, handler: RequestHandler) -> None:
        self._requests[method] = handler

    async def start(self) -> None:
        if self.is_running:
            return

        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            cwd=self._cwd,
        )
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def initialize(
        self,
        *,
        client_name: str = "build-client",
        client_version: str = "0.1.0",
        experimental_api: bool = True,
        opt_out_notifications: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.send_request(
            "initialize",
            {
                "clientInfo": {"name": client_name, "version": client_version},
                "capabilities": {
                    "experimentalApi": experimental_api,
                    "optOutNotificationMethods": opt_out_notifications,
                },
            },
        )

    async def stop(self) -> None:
        process = self._process
        if not process:
            return

        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        for task in (self._stdout_task, self._stderr_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._fail_pending(CodexAppServerError("Codex app-server stopped"))
        self._process = None
        self._stdout_task = None
        self._stderr_task = None

    async def send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.is_running:
            raise CodexAppServerError("Codex app-server is not running")

        request_id = self._next_id
        self._next_id += 1

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future

        await self._send_json({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        })

        try:
            result = await future
            if not isinstance(result, dict):
                raise CodexAppServerError(f"{method} returned non-object result: {result!r}")
            return result
        finally:
            self._pending.pop(request_id, None)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise CodexAppServerError("Codex app-server stdin is unavailable")

        data = json.dumps(payload, separators=(",", ":")) + "\n"
        async with self._write_lock:
            self._process.stdin.write(data.encode("utf-8"))
            await self._process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self._process and self._process.stdout
        try:
            while True:
                raw = await self._process.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("Ignoring non-JSON app-server stdout line: %s", line)
                    continue

                await self._handle_message(message)
        finally:
            self._fail_pending(CodexAppServerError("Codex app-server stdout closed"))

    async def _read_stderr(self) -> None:
        assert self._process and self._process.stderr
        while True:
            raw = await self._process.stderr.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                log.info("[codex-app-server] %s", line)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            await self._handle_server_request(message)
            return

        if "method" in message:
            method = message["method"]
            params = message.get("params", {})
            handler = self._notifications.get(method)
            if not handler:
                return
            result = handler(params)
            if asyncio.iscoroutine(result):
                await result
            return

        request_id = message.get("id")
        future = self._pending.get(request_id)
        if not future or future.done():
            return

        if "error" in message:
            future.set_exception(CodexAppServerError(str(message["error"])))
            return

        future.set_result(message.get("result"))

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        method = message.get("method", "")
        request_id = message.get("id")
        params = message.get("params", {})
        handler = self._requests.get(method)

        if not handler:
            await self._send_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unhandled method: {method}"},
            })
            return

        try:
            result = handler(request_id, params)
            if asyncio.iscoroutine(result):
                result = await result
            await self._send_json({"jsonrpc": "2.0", "id": request_id, "result": result})
        except Exception as exc:
            log.exception("Codex request handler failed for %s", method)
            await self._send_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(exc)},
            })

    def _fail_pending(self, exc: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
