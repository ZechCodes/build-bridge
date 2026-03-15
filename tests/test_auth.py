"""Tests for build_client.auth — registration and SSE approval flow."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from build_client.auth import register_device, wait_for_approval, dismiss_pending


class TestRegisterDevice:
    @pytest.mark.asyncio
    async def test_sends_correct_payload(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": "abc123", "auth_url": "https://example.com/approve/abc123"}
        mock_resp.raise_for_status = MagicMock()

        with patch("build_client.auth.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await register_device(
                base_url="https://example.com",
                name="my-device",
                public_key_b64="cHVia2V5",
            )

            instance.post.assert_called_once()
            call_args = instance.post.call_args
            assert call_args[0][0] == "https://example.com/api/devices/register"
            assert call_args[1]["json"]["name"] == "my-device"
            assert call_args[1]["json"]["public_key"] == "cHVia2V5"
            assert result["code"] == "abc123"


def _make_sse_client(sse_lines: list[str]):
    """Build a mock httpx.AsyncClient whose .stream() yields the given SSE lines."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_stream(method, url):
        class FakeResponse:
            async def aiter_lines(self_inner):
                for line in sse_lines:
                    yield line
        yield FakeResponse()

    mock_client = AsyncMock()
    mock_client.stream = mock_stream
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestWaitForApproval:
    @pytest.mark.asyncio
    async def test_returns_on_approved_event(self):
        """Simulates an SSE stream with an approved event."""
        sse_lines = [
            "event: status",
            'data: {"type": "waiting", "device_name": "test"}',
            "",
            "event: notification",
            'data: {"type": "approved", "device_id": "dev-001", "device_name": "test"}',
            "",
        ]

        mock_client = _make_sse_client(sse_lines)
        with patch("build_client.auth.httpx.AsyncClient", return_value=mock_client):
            result = await wait_for_approval("https://example.com", "abc123")
            assert result["type"] == "approved"
            assert result["device_id"] == "dev-001"

    @pytest.mark.asyncio
    async def test_exits_on_expired_event(self):
        sse_lines = [
            'data: {"type": "expired"}',
        ]

        mock_client = _make_sse_client(sse_lines)
        with patch("build_client.auth.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(SystemExit, match="expired"):
                await wait_for_approval("https://example.com", "abc123")


class TestDismissPending:
    @pytest.mark.asyncio
    async def test_calls_delete(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("build_client.auth.httpx.AsyncClient", return_value=mock_client):
            await dismiss_pending("https://example.com", "abc123")
            mock_client.delete.assert_called_once_with(
                "https://example.com/api/devices/pending/abc123"
            )

    @pytest.mark.asyncio
    async def test_swallows_errors(self):
        """Dismiss is best-effort — errors should not propagate."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.delete.side_effect = Exception("network error")

        with patch("build_client.auth.httpx.AsyncClient", return_value=mock_client):
            # Should not raise.
            await dismiss_pending("https://example.com", "abc123")
