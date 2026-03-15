"""Device configuration — keypair generation, save/load to disk."""

from __future__ import annotations

import base64
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "build" / "device.json"


@dataclass
class DeviceConfig:
    """Persistent device configuration stored locally."""

    device_id: str
    device_name: str
    private_key_b64: str
    public_key_b64: str
    base_url: str


def generate_keypair() -> dict:
    """Generate a new Ed25519 keypair for device auth.

    Returns dict with 'private_key' (Ed25519PrivateKey object),
    'private_key_b64' and 'public_key_b64' (base64-encoded raw bytes).
    """
    private_key = Ed25519PrivateKey.generate()
    return {
        "private_key": private_key,
        "private_key_b64": base64.b64encode(private_key.private_bytes_raw()).decode(),
        "public_key_b64": base64.b64encode(
            private_key.public_key().public_bytes_raw()
        ).decode(),
    }


def save_config(config: DeviceConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Save device config to disk with restricted permissions (0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2))
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> DeviceConfig | None:
    """Load device config from disk. Returns None if missing or corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return DeviceConfig(**data)
    except Exception:
        return None
