"""Tests for build_client.agent_wrapper — BAP wrapper integration tests."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect as ws_connect

from build_client.agent_protocol import (
    ACTIVITY_DELTA,
    ACTIVITY_END,
    ACTIVITY_PING,
    AGENT_CONFIGURED,
    AGENT_ERROR,
    AGENT_GOODBYE,
    AGENT_HELLO,
    AGENT_SHUTDOWN,
    CHAT_CANCEL,
    CHAT_MESSAGE,
    CHAT_RESPONSE,
    TOOL_RESULT,
    TOOL_USE,
    make_envelope,
)
from build_client.agent_server import AgentServer
from build_client.agent_store import AgentStore
from build_client.agent_wrapper import AgentWrapper, CHAT_MCP_TOOLS


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
    return []


@pytest.fixture
def broadcast_callback(broadcast_log):
    async def cb(channel_id: str, payload: dict[str, Any]) -> None:
        broadcast_log.append((channel_id, payload))
    return cb


@pytest.fixture
async def server(store, broadcast_callback):
    port = random.randint(19000, 29000)
    srv = AgentServer(store=store, broadcast=broadcast_callback, port=port)
    await srv.start()
    yield srv, port
    await srv.stop()


@pytest.fixture
def wrapper_factory():
    """Factory to create wrapper instances for a given port."""
    wrappers = []

    def factory(port: int, **kwargs) -> AgentWrapper:
        w = AgentWrapper(port=port, **kwargs)
        wrappers.append(w)
        return w

    yield factory

    # Cleanup.
    for w in wrappers:
        if w.is_connected:
            asyncio.get_event_loop().run_until_complete(w.disconnect())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnect:
    async def test_connect_and_handshake(self, server, wrapper_factory):
        srv, port = server
        wrapper = wrapper_factory(port)

        config = await wrapper.connect()
        assert config is not None
        assert config.channel_id is not None
        assert wrapper.is_connected
        assert wrapper.channel_id == config.channel_id

        await wrapper.disconnect()

    async def test_connect_sets_agent_id(self, server, wrapper_factory):
        srv, port = server
        wrapper = wrapper_factory(port, agent_id="agt_custom123")

        await wrapper.connect()
        assert wrapper.agent_id == "agt_custom123"
        await wrapper.disconnect()

    async def test_connect_receives_chat_instructions(self, server, wrapper_factory):
        srv, port = server
        wrapper = wrapper_factory(port)

        config = await wrapper.connect()
        assert "read_unread" in config.chat_instructions
        assert "send" in config.chat_instructions
        await wrapper.disconnect()


class TestChatMCPIntegration:
    async def test_incoming_chat_message_queued(self, server, wrapper_factory):
        srv, port = server
        wrapper = wrapper_factory(port)

        await wrapper.connect()
        channel_id = wrapper.channel_id

        # Start the receive loop in background.
        run_task = asyncio.create_task(wrapper.run())

        # Send a user message via the server.
        await srv.send_chat_message(channel_id, "Fix the auth bug")

        # Wait for the message to be queued.
        has_msg = await wrapper.chat_mcp.wait_for_unread(timeout=2.0)
        assert has_msg
        assert wrapper.chat_mcp.unread_count == 1

        # Read it via MCP tool.
        result = await wrapper.chat_mcp.handle_read_unread()
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Fix the auth bug"

        await wrapper.disconnect()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    async def test_send_emits_chat_response(self, server, wrapper_factory, broadcast_log):
        srv, port = server
        wrapper = wrapper_factory(port)

        await wrapper.connect()

        # Call send via Chat MCP.
        result = await wrapper.chat_mcp.handle_send("Bug is fixed!")
        assert result == {"status": "sent"}

        # Give server time to process.
        await asyncio.sleep(0.1)

        # Should have been broadcast to browser.
        assert any(
            b[1]["event_type"] == "chat.response"
            and b[1]["event"]["content"] == "Bug is fixed!"
            for b in broadcast_log
        )

        await wrapper.disconnect()


class TestActivityEmission:
    async def test_emit_activity_delta(self, server, wrapper_factory, broadcast_log):
        srv, port = server
        wrapper = wrapper_factory(port)
        await wrapper.connect()

        await wrapper.emit_activity_delta("text", "Analyzing code...")
        await asyncio.sleep(0.1)

        assert any(
            b[1]["event_type"] == "activity.delta"
            and b[1]["event"]["delta"]["text"] == "Analyzing code..."
            for b in broadcast_log
        )
        await wrapper.disconnect()

    async def test_emit_activity_delta_auto_index(self, server, wrapper_factory, broadcast_log):
        srv, port = server
        wrapper = wrapper_factory(port)
        await wrapper.connect()

        await wrapper.emit_activity_delta("text", "First")
        await wrapper.emit_activity_delta("text", "Second")
        await asyncio.sleep(0.1)

        deltas = [b for b in broadcast_log if b[1]["event_type"] == "activity.delta"]
        assert deltas[0][1]["event"]["index"] == 0
        assert deltas[1][1]["event"]["index"] == 1

        await wrapper.disconnect()

    async def test_emit_activity_ping(self, server, wrapper_factory, broadcast_log):
        srv, port = server
        wrapper = wrapper_factory(port)
        await wrapper.connect()

        await wrapper.emit_activity_ping()
        await asyncio.sleep(0.1)

        assert any(b[1]["event_type"] == "activity.ping" for b in broadcast_log)
        await wrapper.disconnect()

    async def test_emit_activity_end(self, server, wrapper_factory, broadcast_log):
        srv, port = server
        wrapper = wrapper_factory(port)
        await wrapper.connect()

        await wrapper.emit_activity_end("complete", usage={"input_tokens": 100, "output_tokens": 50})
        await asyncio.sleep(0.1)

        end_events = [b for b in broadcast_log if b[1]["event_type"] == "activity.end"]
        assert len(end_events) >= 1
        assert end_events[0][1]["event"]["reason"] == "complete"
        assert end_events[0][1]["event"]["usage"]["input_tokens"] == 100

        await wrapper.disconnect()

    async def test_activity_end_resets_index(self, server, wrapper_factory, broadcast_log):
        srv, port = server
        wrapper = wrapper_factory(port)
        await wrapper.connect()

        await wrapper.emit_activity_delta("text", "Turn 1")
        await wrapper.emit_activity_end("complete")
        await wrapper.emit_activity_delta("text", "Turn 2")
        await asyncio.sleep(0.1)

        deltas = [b for b in broadcast_log if b[1]["event_type"] == "activity.delta"]
        # Both turns should start at index 0.
        assert deltas[0][1]["event"]["index"] == 0
        assert deltas[1][1]["event"]["index"] == 0

        await wrapper.disconnect()


class TestToolEmission:
    async def test_emit_tool_use_and_result(self, server, wrapper_factory, broadcast_log):
        srv, port = server
        wrapper = wrapper_factory(port)
        await wrapper.connect()

        await wrapper.emit_tool_use("tu_1", "read_file", {"path": "/src/main.py"})
        await wrapper.emit_tool_result("tu_1", "def main(): pass", is_error=False)
        await asyncio.sleep(0.1)

        tool_events = [b for b in broadcast_log if "tool." in b[1].get("event_type", "")]
        assert len(tool_events) == 2

        await wrapper.disconnect()

    async def test_chat_mcp_tools_filtered(self, server, wrapper_factory, broadcast_log):
        """§8.4: Chat MCP tools must not appear in tool.* namespace."""
        srv, port = server
        wrapper = wrapper_factory(port)
        await wrapper.connect()

        # These should be silently filtered.
        await wrapper.emit_tool_use("tu_mcp1", "read_unread", {})
        await wrapper.emit_tool_use("tu_mcp2", "send", {"message": "hi"})
        await wrapper.emit_tool_result("tu_mcp1", '{"messages": []}', tool_name="read_unread")
        await wrapper.emit_tool_result("tu_mcp2", '{"status": "sent"}', tool_name="send")

        # These should go through.
        await wrapper.emit_tool_use("tu_real", "edit_file", {"path": "/src/main.py"})
        await asyncio.sleep(0.1)

        tool_events = [b for b in broadcast_log if "tool." in b[1].get("event_type", "")]
        assert len(tool_events) == 1  # Only the edit_file
        assert tool_events[0][1]["event"]["name"] == "edit_file"

        await wrapper.disconnect()


class TestCancelAndShutdown:
    async def test_cancel_fires_callback(self, server, wrapper_factory):
        srv, port = server
        cancel_called = asyncio.Event()

        async def on_cancel():
            cancel_called.set()

        wrapper = wrapper_factory(port, on_cancel=on_cancel)
        await wrapper.connect()
        channel_id = wrapper.channel_id

        run_task = asyncio.create_task(wrapper.run())

        # Send cancel from server.
        await srv.send_cancel(channel_id)

        # Wait for callback.
        await asyncio.wait_for(cancel_called.wait(), timeout=2.0)
        assert cancel_called.is_set()

        await wrapper.disconnect()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    async def test_shutdown_sends_goodbye(self, server, wrapper_factory, store):
        srv, port = server
        shutdown_reason = []

        async def on_shutdown(reason: str):
            shutdown_reason.append(reason)

        wrapper = wrapper_factory(port, on_shutdown=on_shutdown)
        await wrapper.connect()

        run_task = asyncio.create_task(wrapper.run())

        # Stop the server (sends agent.shutdown to all agents).
        await srv.stop()

        # Wait for run to complete.
        await asyncio.wait_for(run_task, timeout=5.0)

        assert not wrapper.is_connected


class TestDisconnect:
    async def test_graceful_disconnect(self, server, wrapper_factory):
        srv, port = server
        wrapper = wrapper_factory(port)
        await wrapper.connect()
        assert wrapper.is_connected

        await wrapper.disconnect("completed")
        assert not wrapper.is_connected


class TestReconnection:
    async def test_reconnect_restores_channel(self, server, wrapper_factory, store):
        srv, port = server

        # First connection.
        wrapper1 = wrapper_factory(port, agent_id="agt_reconnect_test")
        config1 = await wrapper1.connect()
        channel_id = config1.channel_id

        # Send a chat response to create history.
        await wrapper1.chat_mcp.handle_send("First response")
        await asyncio.sleep(0.1)

        await wrapper1.disconnect()
        await asyncio.sleep(0.2)

        # Reconnect.
        wrapper2 = wrapper_factory(port, agent_id="agt_reconnect_test", reconnect=True)
        config2 = await wrapper2.connect()

        assert config2.channel_id == channel_id
        assert len(config2.chat_history) >= 1

        await wrapper2.disconnect()
