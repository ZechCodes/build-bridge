"""Device registration and SSE-based authorization flow."""

from __future__ import annotations

import json
import logging
import platform
import webbrowser

import httpx

from build_client.config import DeviceConfig, generate_keypair, save_config

log = logging.getLogger(__name__)


async def register_device(base_url: str, name: str, public_key_b64: str) -> dict:
    """POST /api/devices/register — start the pending registration.

    Returns ``{"code": "...", "auth_url": "..."}``.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{base_url}/api/devices/register",
            json={"name": name, "public_key": public_key_b64},
        )
        resp.raise_for_status()
        return resp.json()


async def wait_for_approval(base_url: str, code: str) -> dict:
    """Connect to the pending-device SSE stream and wait for approval.

    Returns the approval payload: ``{"type": "approved", "device_id": "...", ...}``.
    Raises ``SystemExit`` on expiry or denial.
    """
    url = f"{base_url}/api/devices/pending/{code}/events"
    log.info("Listening for approval on SSE stream...")

    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=15)) as client:
        async with client.stream("GET", url) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except (json.JSONDecodeError, ValueError):
                    continue

                if event.get("type") == "approved":
                    return event
                elif event.get("type") == "expired":
                    raise SystemExit(
                        "Registration expired before approval. Run again to retry."
                    )

    raise SystemExit("SSE stream closed unexpectedly. Run again to retry.")


async def dismiss_pending(base_url: str, code: str) -> None:
    """DELETE /api/devices/pending/{code} — acknowledge receipt of approval."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.delete(f"{base_url}/api/devices/pending/{code}")
    except Exception:
        pass  # Best-effort cleanup.


async def device_auth_flow(base_url: str) -> DeviceConfig:
    """Run the full interactive device auth flow.

    1. Generate keypair
    2. Register with the server (gets a pending code + auth URL)
    3. Open the auth URL in the browser
    4. Wait for the admin to approve via SSE
    5. Dismiss the notification
    6. Save and return the device config
    """
    device_name = platform.node() or "Unknown Device"
    keypair = generate_keypair()

    log.info("Registering device '%s'...", device_name)
    result = await register_device(
        base_url=base_url,
        name=device_name,
        public_key_b64=keypair["public_key_b64"],
    )

    code = result["code"]
    auth_url = result["auth_url"]

    print(f"\n  Authorize this device:\n  {auth_url}\n")

    # Try to open browser.
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    log.info("Waiting for approval...")
    approval = await wait_for_approval(base_url, code)
    log.info("Device approved! ID: %s", approval["device_id"])

    # Dismiss the pending notification.
    await dismiss_pending(base_url, code)

    config = DeviceConfig(
        device_id=approval["device_id"],
        device_name=device_name,
        private_key_b64=keypair["private_key_b64"],
        public_key_b64=keypair["public_key_b64"],
        base_url=base_url,
    )
    save_config(config)
    return config
