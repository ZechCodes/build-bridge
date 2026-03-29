"""WebSocket connection management, heartbeat loop, and E2EE message handling."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Callable, Coroutine, Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from build_client.config import DeviceConfig, save_config
from build_secure_transport import generate_transport_keypair

log = logging.getLogger(__name__)

# Reconnect backoff parameters.
INITIAL_BACKOFF_S = 2
MAX_BACKOFF_S = 60

# Type alias for E2EE message handler callback.
E2EHandler = Callable[[dict[str, Any], Any], Coroutine[Any, Any, None]]


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


def _ensure_transport_keypair(config: DeviceConfig) -> DeviceConfig:
    """Generate X25519 transport keypair if not present, persist to config."""
    if config.transport_private_key_b64 and config.transport_public_key_b64:
        return config

    log.info("Generating X25519 transport keypair for E2EE...")
    kp = generate_transport_keypair()
    config.transport_private_key_b64 = kp["private_key_b64"]
    config.transport_public_key_b64 = kp["public_key_b64"]
    save_config(config)
    log.info("Transport keypair saved to config")
    return config


async def _upload_transport_key(ws, config: DeviceConfig) -> None:
    """Upload X25519 transport public key to the server."""
    await ws.send(json.dumps({
        "type": "transport_key",
        "transport_public_key": config.transport_public_key_b64,
    }))
    log.info("Transport key uploaded to server")


async def _heartbeat_loop(ws, interval: int) -> None:
    """Send heartbeats on the given interval."""
    while True:
        await asyncio.sleep(interval)
        try:
            await ws.send(json.dumps({"type": "heartbeat"}))
            log.debug("Heartbeat sent")
        except ConnectionClosed:
            return


async def _receive_loop(
    ws,
    config: DeviceConfig,
    e2e_handler: E2EHandler | None = None,
    on_restart: Callable[[], None] | None = None,
) -> None:
    """Read messages from the server and dispatch E2EE messages."""
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
            elif msg_type == "restart":
                log.info("Restart command received from server")
                if on_restart:
                    on_restart()
                return
            elif msg_type in ("session_init", "e2ee_envelope"):
                # E2EE messages — dispatch to handler.
                if e2e_handler:
                    try:
                        await e2e_handler(msg, ws)
                    except Exception as exc:
                        log.error("E2EE handler error: %s", exc)
                else:
                    log.warning("E2EE message received but no handler registered: %s", msg_type)
            else:
                log.debug("Server message: %s", msg)
    except ConnectionClosed:
        pass


async def run_connection(
    config: DeviceConfig,
    e2e_handler: E2EHandler | None = None,
    on_restart: Callable[[], None] | None = None,
) -> None:
    """Establish WS connection and run heartbeat + receive loops.

    Reconnects automatically with exponential backoff on disconnection.

    Args:
        config: Device configuration with auth and transport keys.
        e2e_handler: Optional async callback for E2EE messages (session_init, e2ee_envelope).
            Called with (message_dict, websocket) so it can send responses.
        on_restart: Optional callback invoked when a restart command is received from the server.
    """
    # Ensure transport keypair exists.
    config = _ensure_transport_keypair(config)

    # Build the WS URL from the HTTP base URL.
    ws_base = config.base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/api/devices/ws"

    backoff = INITIAL_BACKOFF_S

    while True:
        headers = _sign_handshake(config)
        log.info("Connecting to %s ...", ws_url)

        try:
            async with ws_connect(ws_url, additional_headers=headers, max_size=2 * 1024 * 1024) as ws:
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

                # Upload transport key to server.
                await _upload_transport_key(ws, config)

                # Run heartbeat and receive loops concurrently.
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(_heartbeat_loop(ws, interval))
                    tg.create_task(_receive_loop(ws, config, e2e_handler, on_restart))

        except SystemExit:
            raise
        except Exception as e:
            log.warning("Connection lost: %s. Reconnecting in %ss...", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
