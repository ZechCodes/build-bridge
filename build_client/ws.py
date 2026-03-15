"""WebSocket connection management and heartbeat loop."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from build_client.config import DeviceConfig

log = logging.getLogger(__name__)

# Reconnect backoff parameters.
INITIAL_BACKOFF_S = 2
MAX_BACKOFF_S = 60


def _sign_handshake(config: DeviceConfig) -> dict[str, str]:
    """Sign a WS handshake and return the extra headers."""
    key_bytes = base64.b64decode(config.private_key_b64)
    private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)

    timestamp_str = str(time.time())
    path = "/api/devices/ws"
    message = f"{timestamp_str}.GET.{path}".encode()
    signature = private_key.sign(message)

    return {
        "X-Device-Id": config.device_id,
        "X-Timestamp": timestamp_str,
        "X-Signature": base64.b64encode(signature).decode(),
    }


async def _heartbeat_loop(ws, interval: int) -> None:
    """Send heartbeats on the given interval."""
    while True:
        await asyncio.sleep(interval)
        try:
            await ws.send(json.dumps({"type": "heartbeat"}))
            log.debug("Heartbeat sent")
        except ConnectionClosed:
            return


async def _receive_loop(ws) -> None:
    """Read messages from the server."""
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            if msg_type == "authenticated":
                log.info(
                    "Authenticated (device_id=%s, heartbeat=%ss)",
                    msg.get("device_id"),
                    msg.get("heartbeat_interval_s"),
                )
            elif msg_type == "response":
                if not msg.get("ok"):
                    log.warning("Server response error: %s", msg.get("error"))
            elif msg_type == "error":
                log.error("Server error: %s", msg.get("error"))
            else:
                log.debug("Server message: %s", msg)
    except ConnectionClosed:
        pass


async def run_connection(config: DeviceConfig) -> None:
    """Establish WS connection and run heartbeat + receive loops.

    Reconnects automatically with exponential backoff on disconnection.
    """
    # Build the WS URL from the HTTP base URL.
    ws_base = config.base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/api/devices/ws"

    backoff = INITIAL_BACKOFF_S

    while True:
        headers = _sign_handshake(config)
        log.info("Connecting to %s ...", ws_url)

        try:
            async with ws_connect(ws_url, additional_headers=headers) as ws:
                # Wait for the authenticated message to get heartbeat interval.
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw)
                if msg.get("type") == "error":
                    log.error("Auth failed: %s", msg.get("error"))
                    raise SystemExit(f"Device auth rejected: {msg.get('error')}")

                interval = msg.get("heartbeat_interval_s", 30)
                log.info("Connected! Heartbeat interval: %ss", interval)

                # Reset backoff on successful connection.
                backoff = INITIAL_BACKOFF_S

                # Run heartbeat and receive loops concurrently.
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(_heartbeat_loop(ws, interval))
                    tg.create_task(_receive_loop(ws))

        except SystemExit:
            raise
        except Exception as e:
            log.warning("Connection lost: %s. Reconnecting in %ss...", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
