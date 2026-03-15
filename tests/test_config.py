"""Tests for build_client.config — keypair generation and config persistence."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from build_client.config import (
    DeviceConfig,
    generate_keypair,
    load_config,
    save_config,
)


class TestGenerateKeypair:
    def test_returns_all_fields(self):
        kp = generate_keypair()
        assert "private_key" in kp
        assert "private_key_b64" in kp
        assert "public_key_b64" in kp

    def test_keys_are_base64(self):
        import base64

        kp = generate_keypair()
        # Should decode without error.
        raw_priv = base64.b64decode(kp["private_key_b64"])
        raw_pub = base64.b64decode(kp["public_key_b64"])
        assert len(raw_priv) == 32
        assert len(raw_pub) == 32

    def test_keypairs_are_unique(self):
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        assert kp1["public_key_b64"] != kp2["public_key_b64"]

    def test_sign_and_verify_roundtrip(self):
        import base64

        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        kp = generate_keypair()
        private_key = kp["private_key"]
        message = b"test message"
        signature = private_key.sign(message)

        pub_bytes = base64.b64decode(kp["public_key_b64"])
        pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
        # Should not raise.
        pub_key.verify(signature, message)


class TestConfigPersistence:
    def test_save_and_load(self, tmp_path: Path):
        path = tmp_path / "device.json"
        config = DeviceConfig(
            device_id="test-id",
            device_name="test-device",
            private_key_b64="privkey",
            public_key_b64="pubkey",
            base_url="https://example.com",
        )
        save_config(config, path)
        loaded = load_config(path)
        assert loaded is not None
        assert loaded.device_id == "test-id"
        assert loaded.device_name == "test-device"
        assert loaded.base_url == "https://example.com"

    def test_file_permissions(self, tmp_path: Path):
        path = tmp_path / "device.json"
        config = DeviceConfig(
            device_id="x",
            device_name="x",
            private_key_b64="x",
            public_key_b64="x",
            base_url="x",
        )
        save_config(config, path)
        mode = path.stat().st_mode
        assert mode & 0o777 == stat.S_IRUSR | stat.S_IWUSR  # 0o600

    def test_load_missing_file(self, tmp_path: Path):
        assert load_config(tmp_path / "nonexistent.json") is None

    def test_load_corrupt_file(self, tmp_path: Path):
        path = tmp_path / "device.json"
        path.write_text("not json")
        assert load_config(path) is None

    def test_roundtrip_preserves_all_fields(self, tmp_path: Path):
        path = tmp_path / "device.json"
        kp = generate_keypair()
        config = DeviceConfig(
            device_id="dev-123",
            device_name="my-laptop",
            private_key_b64=kp["private_key_b64"],
            public_key_b64=kp["public_key_b64"],
            base_url="https://getbuild.ing",
        )
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.device_id == config.device_id
        assert loaded.private_key_b64 == config.private_key_b64
        assert loaded.public_key_b64 == config.public_key_b64
