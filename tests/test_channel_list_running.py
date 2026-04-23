"""Tests for the `is_running` flag in _send_channel_list responses.

The flag is the authoritative source the browser uses to reconcile
`presenceStore.agentActive` after SSE blips / focus resyncs. It must be
True iff the spawner reports the worker subprocess alive AND the agent
store has status 'active' (i.e. mid-turn, not idle/error/closed)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from build_bridge.e2ee import E2EEHandler


class _FakeSession:
    session_id = "fake"
    session_key_b64 = "fake"


@pytest.fixture
def handler(tmp_path: Path, monkeypatch) -> E2EEHandler:
    cfg = SimpleNamespace()
    store = MagicMock()
    store.list_channels.return_value = []  # overridden per-test
    h = E2EEHandler(cfg, store)

    h._sent_frames = []

    async def fake_send(session, ws, payload):
        h._sent_frames.append(payload)

    h._send_frame = fake_send  # type: ignore[assignment]
    return h


def _channel(id_: str, name: str = "c") -> SimpleNamespace:
    return SimpleNamespace(id=id_, name=name, created_at="t0")


def _agent_ch(**overrides) -> SimpleNamespace:
    defaults = dict(
        harness="claude-code",
        model="claude-sonnet",
        working_directory="",
        effort="",
        plan_mode=False,
        auto_approve_tools=False,
        last_seen_at=None,
        status="active",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_is_running_true_when_worker_alive_and_status_active(handler):
    handler.store.list_channels.return_value = [_channel("ch_run")]

    agent_server = MagicMock()
    agent_server.store.get_channel.return_value = _agent_ch(status="active")
    handler._agent_server = agent_server

    spawner = MagicMock()
    spawner.is_running.return_value = True
    handler._agent_spawner = spawner

    await handler._send_channel_list(_FakeSession(), object())
    frame = next(f for f in handler._sent_frames if f.get("action") == "channel_list")
    entry = frame["channels"][0]
    assert entry["is_running"] is True


@pytest.mark.asyncio
async def test_is_running_false_when_worker_dead(handler):
    handler.store.list_channels.return_value = [_channel("ch_dead")]

    agent_server = MagicMock()
    agent_server.store.get_channel.return_value = _agent_ch(status="active")
    handler._agent_server = agent_server

    spawner = MagicMock()
    spawner.is_running.return_value = False  # worker not alive
    handler._agent_spawner = spawner

    await handler._send_channel_list(_FakeSession(), object())
    frame = next(f for f in handler._sent_frames if f.get("action") == "channel_list")
    assert frame["channels"][0]["is_running"] is False


@pytest.mark.asyncio
async def test_is_running_false_when_status_idle(handler):
    handler.store.list_channels.return_value = [_channel("ch_idle")]

    agent_server = MagicMock()
    agent_server.store.get_channel.return_value = _agent_ch(status="idle")
    handler._agent_server = agent_server

    spawner = MagicMock()
    spawner.is_running.return_value = True  # subprocess alive but waiting
    handler._agent_spawner = spawner

    await handler._send_channel_list(_FakeSession(), object())
    frame = next(f for f in handler._sent_frames if f.get("action") == "channel_list")
    assert frame["channels"][0]["is_running"] is False


@pytest.mark.asyncio
async def test_is_running_false_when_spawner_missing(handler):
    """Defensive: e2ee wired without a spawner should report `False`,
    not crash or omit the field."""
    handler.store.list_channels.return_value = [_channel("ch_nospawn")]

    agent_server = MagicMock()
    agent_server.store.get_channel.return_value = _agent_ch(status="active")
    handler._agent_server = agent_server

    handler._agent_spawner = None

    await handler._send_channel_list(_FakeSession(), object())
    frame = next(f for f in handler._sent_frames if f.get("action") == "channel_list")
    assert frame["channels"][0]["is_running"] is False


@pytest.mark.asyncio
async def test_stub_channels_without_agent_row_lack_is_running(handler):
    """Channels that haven't had a spawn — agent_server.store returns
    None — aren't enriched with is_running; the UI treats missing as
    'not running' by default."""
    handler.store.list_channels.return_value = [_channel("ch_stub")]

    agent_server = MagicMock()
    agent_server.store.get_channel.return_value = None
    handler._agent_server = agent_server

    spawner = MagicMock()
    spawner.is_running.return_value = False
    handler._agent_spawner = spawner

    await handler._send_channel_list(_FakeSession(), object())
    frame = next(f for f in handler._sent_frames if f.get("action") == "channel_list")
    entry = frame["channels"][0]
    # Only the e2ee-store fields are present.
    assert set(entry.keys()) == {"id", "name", "created_at"}
