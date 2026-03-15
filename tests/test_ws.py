"""Tests for build_client.ws — handshake signing and connection logic."""

from __future__ import annotations

import base64
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from build_client.config import DeviceConfig, generate_keypair
from build_client.ws import _sign_handshake, INITIAL_BACKOFF_S, MAX_BACKOFF_S


def _make_config() -> DeviceConfig:
    kp = generate_keypair()
    return DeviceConfig(
        device_id="test-device-id",
        device_name="test-device",
        private_key_b64=kp["private_key_b64"],
        public_key_b64=kp["public_key_b64"],
        base_url="https://getbuild.ing",
    )


class TestSignHandshake:
    def test_returns_required_headers(self):
        config = _make_config()
        headers = _sign_handshake(config)
        assert "X-Device-Id" in headers
        assert "X-Timestamp" in headers
        assert "X-Signature" in headers

    def test_device_id_matches(self):
        config = _make_config()
        headers = _sign_handshake(config)
        assert headers["X-Device-Id"] == "test-device-id"

    def test_timestamp_is_recent(self):
        config = _make_config()
        headers = _sign_handshake(config)
        ts = float(headers["X-Timestamp"])
        assert abs(time.time() - ts) < 5

    def test_signature_verifies(self):
        config = _make_config()
        headers = _sign_handshake(config)

        # Reconstruct the signed message.
        timestamp_str = headers["X-Timestamp"]
        path = "/api/devices/ws"
        message = f"{timestamp_str}.GET.{path}".encode()

        # Verify with the public key.
        pub_bytes = base64.b64decode(config.public_key_b64)
        pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
        signature = base64.b64decode(headers["X-Signature"])
        pub_key.verify(signature, message)  # Raises on failure.

    def test_wrong_key_fails_verification(self):
        config = _make_config()
        headers = _sign_handshake(config)

        # Different keypair.
        other_kp = generate_keypair()
        other_pub = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(other_kp["public_key_b64"])
        )
        signature = base64.b64decode(headers["X-Signature"])
        timestamp_str = headers["X-Timestamp"]
        message = f"{timestamp_str}.GET./api/devices/ws".encode()

        import pytest
        with pytest.raises(Exception):
            other_pub.verify(signature, message)


class TestBackoffConstants:
    def test_initial_backoff(self):
        assert INITIAL_BACKOFF_S == 2

    def test_max_backoff(self):
        assert MAX_BACKOFF_S == 60

    def test_backoff_doubles(self):
        backoff = INITIAL_BACKOFF_S
        backoff = min(backoff * 2, MAX_BACKOFF_S)
        assert backoff == 4
        backoff = min(backoff * 2, MAX_BACKOFF_S)
        assert backoff == 8

    def test_backoff_caps_at_max(self):
        backoff = INITIAL_BACKOFF_S
        for _ in range(20):
            backoff = min(backoff * 2, MAX_BACKOFF_S)
        assert backoff == MAX_BACKOFF_S
