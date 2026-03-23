"""Wrapper-owned stdio MCP bridge for Build chat tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ENV_SOCKET = "BUILD_CHAT_BRIDGE_SOCKET"
ENV_TOKEN = "BUILD_CHAT_BRIDGE_TOKEN"


class BuildChatBridgeServer:
    """Local IPC bridge exposing ChatMCP methods to a child MCP process."""

    def __init__(self, *, chat_mcp: Any, socket_path: str, token: str) -> None:
        self._chat_mcp = chat_mcp
        self._socket_path = Path(socket_path)
        self._token = token
        self._server: asyncio.AbstractServer | None = None

    @property
    def socket_path(self) -> str:
        return str(self._socket_path)

    async def start(self) -> None:
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self._socket_path.exists():
            self._socket_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
        )

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._socket_path.exists():
            self._socket_path.unlink()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw = await reader.readline()
            if not raw:
                return
            request = json.loads(raw.decode("utf-8"))
            if request.get("token") != self._token:
                response = {"ok": False, "error": "invalid token"}
            else:
                response = {"ok": True, "result": await self._dispatch(request)}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}

        writer.write((json.dumps(response) + "\n").encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        method = request.get("method")
        payload = request.get("payload", {})

        if method == "read_unread":
            return await self._chat_mcp.handle_read_unread()
        if method == "send":
            return await self._chat_mcp.handle_send(payload.get("message", ""))

        raise ValueError(f"unknown bridge method: {method}")


class _BuildChatBridgeClient:
    def __init__(self, *, socket_path: str, token: str) -> None:
        self._socket_path = socket_path
        self._token = token

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        reader, writer = await asyncio.open_unix_connection(self._socket_path)
        writer.write((json.dumps({
            "token": self._token,
            "method": method,
            "payload": payload or {},
        }) + "\n").encode("utf-8"))
        await writer.drain()
        raw = await reader.readline()
        writer.close()
        await writer.wait_closed()

        if not raw:
            raise RuntimeError("bridge returned no response")

        response = json.loads(raw.decode("utf-8"))
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "bridge request failed"))
        return response.get("result", {})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    socket_path = os.environ.get(ENV_SOCKET)
    token = os.environ.get(ENV_TOKEN)
    if not socket_path or not token:
        raise SystemExit("BUILD_CHAT_BRIDGE_SOCKET and BUILD_CHAT_BRIDGE_TOKEN are required")

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("The 'mcp' package is required to run the Build chat bridge") from exc

    bridge = _BuildChatBridgeClient(socket_path=socket_path, token=token)
    mcp = FastMCP(name="build-chat")

    @mcp.tool(
        name="read_unread",
        description="Read unread messages from the Build user chat queue.",
    )
    async def read_unread() -> dict[str, Any]:
        return await bridge.call("read_unread")

    @mcp.tool(
        name="send",
        description="Send a user-visible message back to the Build browser UI.",
    )
    async def send(message: str) -> dict[str, Any]:
        return await bridge.call("send", {"message": message})

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
