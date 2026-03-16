"""Tests for build_client.agent_server — BAP WebSocket server integration tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect as ws_connect

from build_client.agent_protocol import (
    AGENT_CONFIGURED,
    AGENT_ERROR,
    AGENT_HELLO,
    CHAT_RESPONSE,
    ACTIVITY_DELTA,
    ACTIVITY_END,
    TOOL_USE,
    TOOL_RESULT,
    PROTOCOL_VERSION,
    make_envelope,
)
from build_client.agent_server import AgentServer
from build_client.agent_store import AgentStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> AgentStore:
    s = AgentStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def broadcast_log() -> list[tuple[str, dict[str, Any]]]:
    """Collects broadcast calls for assertion."""
    return []


@pytest.fixture
def broadcast_callback(broadcast_log):
    async def cb(channel_id: str, payload: dict[str, Any]) -> None:
        broadcast_log.append((channel_id, payload))
    return cb


@pytest.fixture
async def server(store, broadcast_callback):
    """Start an agent server on a random available port."""
    srv = AgentServer(store=store, broadcast=broadcast_callback, port=0)
    # Use port 0 to let OS assign a free port.
    # We need to work around this — let's pick a high random port.
    import random
    port = random.randint(19000, 29000)
    srv._port = port
    await srv.start()
    yield srv, port
    await srv.stop()


def _hello_envelope(
    agent_id: str = "agt_test1",
    harness: str = "claude-code",
    capabilities: list[str] | None = None,
    model: str = "claude-sonnet-4-20250514",
    reconnect: bool = False,
) -> dict[str, Any]:
    """Build an agent.hello envelope."""
    return make_envelope(AGENT_HELLO, {
        "agent_id": agent_id,
        "harness": harness,
        "capabilities": capabilities or ["chat", "activity", "tools"],
        "model": model,
        "reconnect": reconnect,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandshake:
    async def test_hello_and_configured(self, server, store):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            # Send agent.hello.
            await ws.send(json.dumps(_hello_envelope()))

            # Should receive agent.configured.
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)

            assert msg["type"] == AGENT_CONFIGURED
            assert msg["v"] == PROTOCOL_VERSION
            assert "channel_id" in msg["payload"]
            assert "system_prompt" in msg["payload"]
            assert "chat_instructions" in msg["payload"]
            assert "history" in msg["payload"]
            assert "chat" in msg["payload"]["history"]
            assert "activity" in msg["payload"]["history"]

            # Channel should exist in DB.
            ch = store.get_channel(msg["payload"]["channel_id"])
            assert ch is not None
            assert ch.agent_id == "agt_test1"
            assert ch.harness == "claude-code"
            assert ch.status == "active"

    async def test_non_hello_first_message_rejected(self, server):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            # Send a non-hello message.
            await ws.send(json.dumps(make_envelope("chat.response", {"content": "hi"})))

            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == AGENT_ERROR
            assert msg["payload"]["fatal"] is True
            assert "agent.hello" in msg["payload"]["message"]

    async def test_invalid_json_rejected(self, server):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send("not json at all")

            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == AGENT_ERROR
            assert msg["payload"]["fatal"] is True

    async def test_missing_hello_fields_rejected(self, server):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            # Hello without agent_id.
            bad_hello = make_envelope(AGENT_HELLO, {
                "harness": "claude-code",
                "capabilities": ["chat"],
                "model": "test",
                "reconnect": False,
            })
            await ws.send(json.dumps(bad_hello))

            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == AGENT_ERROR
            assert "agent_id" in msg["payload"]["message"]


class TestChatResponse:
    async def test_chat_response_stored_and_broadcast(self, server, store, broadcast_log):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            # Handshake.
            await ws.send(json.dumps(_hello_envelope()))
            configured = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            channel_id = configured["payload"]["channel_id"]

            # Send chat.response.
            response = make_envelope(CHAT_RESPONSE, {"content": "I'll fix the bug."})
            await ws.send(json.dumps(response))

            # Give the server a moment to process.
            await asyncio.sleep(0.1)

            # Chat message should be stored.
            history = store.get_chat_history(channel_id)
            assert len(history) == 1
            assert history[0].role == "assistant"
            assert history[0].content == "I'll fix the bug."

            # Should have been broadcast to browser.
            assert any(
                b[1]["event_type"] == "chat.response"
                and b[1]["event"]["content"] == "I'll fix the bug."
                for b in broadcast_log
            )


class TestActivity:
    async def test_activity_delta_stored_and_broadcast(self, server, store, broadcast_log):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps(_hello_envelope()))
            configured = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            channel_id = configured["payload"]["channel_id"]

            # Send activity.delta.
            delta = make_envelope(ACTIVITY_DELTA, {
                "delta": {"type": "text", "text": "Analyzing the code..."},
                "index": 0,
            })
            await ws.send(json.dumps(delta))
            await asyncio.sleep(0.1)

            # Activity should be stored.
            activity = store.get_activity_history(channel_id)
            assert len(activity) == 1
            assert activity[0].type == "text"

            # Should have been broadcast.
            assert any(
                b[1]["event_type"] == "activity.delta"
                for b in broadcast_log
            )

    async def test_activity_end_updates_status(self, server, store, broadcast_log):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps(_hello_envelope()))
            configured = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            channel_id = configured["payload"]["channel_id"]

            # Send activity.end.
            end = make_envelope(ACTIVITY_END, {
                "reason": "complete",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            })
            await ws.send(json.dumps(end))
            await asyncio.sleep(0.1)

            # Channel status should be idle.
            ch = store.get_channel(channel_id)
            assert ch.status == "idle"


class TestToolUse:
    async def test_tool_use_and_result_stored(self, server, store, broadcast_log):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps(_hello_envelope()))
            configured = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            channel_id = configured["payload"]["channel_id"]

            # Send tool.use.
            tu = make_envelope(TOOL_USE, {
                "tool_use_id": "tu_001",
                "name": "read_file",
                "input": {"path": "/src/main.py"},
            })
            await ws.send(json.dumps(tu))
            await asyncio.sleep(0.1)

            # Tool use should be stored.
            tools = store.get_tool_uses(channel_id)
            assert len(tools) == 1
            assert tools[0].name == "read_file"
            assert tools[0].output is None

            # Send tool.result.
            tr = make_envelope(TOOL_RESULT, {
                "tool_use_id": "tu_001",
                "content": "def main(): pass",
                "is_error": False,
            })
            await ws.send(json.dumps(tr))
            await asyncio.sleep(0.1)

            # Tool result should be stored.
            tools = store.get_tool_uses(channel_id)
            assert tools[0].output == "def main(): pass"
            assert tools[0].is_error is False
            assert tools[0].completed_at is not None

            # Both should have been broadcast.
            tool_events = [b for b in broadcast_log if "tool." in b[1].get("event_type", "")]
            assert len(tool_events) == 2


class TestCapabilityEnforcement:
    async def test_chat_without_capability_rejected(self, server):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            # Connect with only activity capability.
            hello = _hello_envelope(capabilities=["activity"])
            await ws.send(json.dumps(hello))
            await asyncio.wait_for(ws.recv(), timeout=5)  # configured

            # Send chat.response without chat capability.
            response = make_envelope(CHAT_RESPONSE, {"content": "test"})
            await ws.send(json.dumps(response))

            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == AGENT_ERROR
            assert msg["payload"]["code"] == "capability_mismatch"


class TestDirectionEnforcement:
    async def test_client_to_agent_type_rejected(self, server):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps(_hello_envelope()))
            await asyncio.wait_for(ws.recv(), timeout=5)  # configured

            # Agent sends a client-to-agent message type.
            bad = make_envelope("chat.message", {"role": "user", "content": "hi"})
            await ws.send(json.dumps(bad))

            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == AGENT_ERROR
            assert msg["payload"]["code"] == "protocol_violation"


class TestReconnection:
    async def test_reconnect_restores_channel(self, server, store):
        srv, port = server

        # First connection — creates channel.
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps(_hello_envelope(agent_id="agt_reconnect")))
            configured = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            channel_id = configured["payload"]["channel_id"]

            # Store a chat message.
            store.store_chat_message("msg_1", channel_id, "user", "Hello agent")

        # Agent disconnects — wait for cleanup.
        await asyncio.sleep(0.2)

        # Reconnect with same agent_id.
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            hello = _hello_envelope(agent_id="agt_reconnect", reconnect=True)
            await ws.send(json.dumps(hello))

            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == AGENT_CONFIGURED

            # Should get same channel.
            assert msg["payload"]["channel_id"] == channel_id

            # Should include chat history.
            chat = msg["payload"]["history"]["chat"]
            assert len(chat) == 1
            assert chat[0]["content"] == "Hello agent"

    async def test_reconnect_unknown_agent_fails(self, server):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            hello = _hello_envelope(agent_id="agt_unknown", reconnect=True)
            await ws.send(json.dumps(hello))

            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == AGENT_ERROR
            assert "reconnect" in msg["payload"]["code"]


class TestSendChatMessage:
    async def test_send_to_connected_agent(self, server, store):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps(_hello_envelope()))
            configured = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            channel_id = configured["payload"]["channel_id"]

            # Send a chat message from user to agent.
            sent = await srv.send_chat_message(channel_id, "Fix the auth bug")
            assert sent

            # Agent should receive chat.message.
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == "chat.message"
            assert msg["payload"]["role"] == "user"
            assert msg["payload"]["content"] == "Fix the auth bug"

            # User message should be stored.
            history = store.get_chat_history(channel_id)
            assert len(history) == 1
            assert history[0].role == "user"

    async def test_send_to_disconnected_channel(self, server):
        srv, port = server
        sent = await srv.send_chat_message("ch_nonexistent", "Hello?")
        assert not sent


class TestBrowserNotifications:
    async def test_connect_and_disconnect_notifications(self, server, broadcast_log):
        srv, port = server
        async with ws_connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps(_hello_envelope()))
            await asyncio.wait_for(ws.recv(), timeout=5)

        # Wait for disconnect cleanup.
        await asyncio.sleep(0.2)

        # Should have connect and disconnect notifications.
        event_types = [b[1]["event_type"] for b in broadcast_log]
        assert "agent.connected" in event_types
        assert "agent.disconnected" in event_types
