"""Tests for build_client.build_agent — Claude Code integration."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import pytest

from build_client.agent_protocol import CHAT_RESPONSE, ACTIVITY_DELTA, TOOL_USE
from build_client.agent_server import AgentServer
from build_client.agent_store import AgentStore
from build_client.agent_wrapper import AgentWrapper
from build_client.build_agent import (
    CHAT_CONTEXT,
    build_agent_options,
    handle_response_message,
    make_chat_tools,
    make_pre_tool_hook,
    make_post_tool_hook,
    make_stop_hook,
    _describe_tool,
)


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
async def wrapper(server):
    srv, port = server
    w = AgentWrapper(port=port, harness="claude-code", model="claude-sonnet-4-20250514")
    await w.connect()
    yield w
    await w.disconnect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDescribeTool:
    def test_read(self):
        assert "main.py" in _describe_tool("Read", {"file_path": "/src/main.py"})

    def test_edit(self):
        assert "auth.py" in _describe_tool("Edit", {"file_path": "/src/auth.py"})

    def test_write(self):
        assert "config.py" in _describe_tool("Write", {"file_path": "/src/config.py"})

    def test_bash(self):
        desc = _describe_tool("Bash", {"command": "git status"})
        assert "git status" in desc

    def test_bash_truncation(self):
        long_cmd = "x" * 100
        desc = _describe_tool("Bash", {"command": long_cmd})
        assert len(desc) < 100

    def test_glob(self):
        assert "*.py" in _describe_tool("Glob", {"pattern": "*.py"})

    def test_grep(self):
        assert "TODO" in _describe_tool("Grep", {"pattern": "TODO"})

    def test_agent(self):
        assert "fix bug" in _describe_tool("Agent", {"description": "fix bug"})

    def test_unknown_tool(self):
        assert "CustomTool" in _describe_tool("CustomTool", {})


class TestChatTools:
    async def test_read_unread_tool(self, wrapper):
        tools = make_chat_tools(wrapper)
        read_tool = next(t for t in tools if t.name == "read_unread")

        # Queue a message and read it via the tool.
        await wrapper.chat_mcp.queue_message("Hello agent")
        result = read_tool.handler  # The raw handler
        # Use handle_read_unread directly since we can't call SDK tools in tests.
        mcp_result = await wrapper.chat_mcp.handle_read_unread()
        assert len(mcp_result["messages"]) == 1
        assert mcp_result["messages"][0]["content"] == "Hello agent"

    async def test_send_tool(self, wrapper, broadcast_log):
        result = await wrapper.chat_mcp.handle_send("Bug is fixed!")
        assert result == {"status": "sent"}

        await asyncio.sleep(0.1)
        assert any(
            b[1]["event_type"] == "chat.response"
            for b in broadcast_log
        )


class TestPreToolHook:
    async def test_emits_tool_use(self, wrapper, broadcast_log):
        hook = make_pre_tool_hook(wrapper)

        result = await hook(
            {"tool_name": "Read", "tool_input": {"file_path": "/src/main.py"}},
            "tu_test_1",
            {},
        )

        assert result == {"continue_": True}
        await asyncio.sleep(0.1)

        tool_events = [b for b in broadcast_log if b[1].get("event_type") == "tool.use"]
        assert len(tool_events) == 1
        assert tool_events[0][1]["event"]["name"] == "Read"

    async def test_skips_chat_mcp_tools(self, wrapper, broadcast_log):
        hook = make_pre_tool_hook(wrapper)

        result = await hook(
            {"tool_name": "mcp__build_chat__send", "tool_input": {"message": "hi"}},
            "tu_mcp_1",
            {},
        )

        assert result == {"continue_": True}
        await asyncio.sleep(0.1)

        tool_events = [b for b in broadcast_log if b[1].get("event_type") == "tool.use"]
        assert len(tool_events) == 0


class TestPostToolHook:
    async def test_emits_tool_result(self, wrapper, broadcast_log):
        # Emit tool.use first so there's something to complete.
        await wrapper.emit_tool_use("tu_test_1", "Read", {"file_path": "/src/main.py"})
        await asyncio.sleep(0.05)

        hook = make_post_tool_hook(wrapper)
        result = await hook(
            {"tool_name": "Read", "tool_input": {"file_path": "/src/main.py"}},
            "tu_test_1",
            {},
        )

        assert result == {}
        await asyncio.sleep(0.1)

        result_events = [b for b in broadcast_log if b[1].get("event_type") == "tool.result"]
        assert len(result_events) >= 1

    async def test_skips_chat_mcp_tools(self, wrapper, broadcast_log):
        hook = make_post_tool_hook(wrapper)

        result = await hook(
            {"tool_name": "mcp__build_chat__read_unread", "tool_input": {}},
            "tu_mcp_1",
            {},
        )

        assert result == {}
        await asyncio.sleep(0.1)

        result_events = [b for b in broadcast_log if b[1].get("event_type") == "tool.result"]
        assert len(result_events) == 0


class TestStopHook:
    async def test_emits_activity_end(self, wrapper, broadcast_log):
        hook = make_stop_hook(wrapper)

        result = await hook({}, None, {})
        assert result == {}
        await asyncio.sleep(0.1)

        end_events = [b for b in broadcast_log if b[1].get("event_type") == "activity.end"]
        assert len(end_events) == 1
        assert end_events[0][1]["event"]["reason"] == "waiting"


class TestBuildAgentOptions:
    def test_options_structure(self, wrapper):
        options = build_agent_options(wrapper)

        assert "build_chat" in options.mcp_servers
        assert "PreToolUse" in options.hooks
        assert "PostToolUse" in options.hooks
        assert "Stop" in options.hooks
        assert "PreCompact" in options.hooks
        assert options.permission_mode == "plan"


class TestChatContext:
    def test_context_mentions_tools(self):
        assert "mcp__build_chat__send" in CHAT_CONTEXT
        assert "mcp__build_chat__read_unread" in CHAT_CONTEXT

    def test_context_warns_about_visibility(self):
        assert "CANNOT see" in CHAT_CONTEXT


class TestEndToEndFlow:
    async def test_user_message_to_agent_send(self, server, store, broadcast_log):
        """Full flow: user sends message → agent reads → agent sends response."""
        srv, port = server

        wrapper = AgentWrapper(port=port, harness="claude-code", model="claude-sonnet-4-20250514")
        config = await wrapper.connect()
        channel_id = config.channel_id

        # Start receive loop.
        run_task = asyncio.create_task(wrapper.run())

        # User sends a message via the server.
        await srv.send_chat_message(channel_id, "What's the status?")

        # Wait for it to be queued.
        has_msg = await wrapper.chat_mcp.wait_for_unread(timeout=2.0)
        assert has_msg

        # Agent reads unread.
        result = await wrapper.chat_mcp.handle_read_unread()
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "What's the status?"

        # Agent sends a response.
        await wrapper.chat_mcp.handle_send("Everything is running smoothly!")
        await asyncio.sleep(0.1)

        # Verify it was broadcast to browser.
        responses = [
            b for b in broadcast_log
            if b[1].get("event_type") == "chat.response"
            and b[1]["event"]["content"] == "Everything is running smoothly!"
        ]
        assert len(responses) == 1

        # Verify it was stored in DB.
        history = store.get_chat_history(channel_id)
        assert any(m.role == "user" and m.content == "What's the status?" for m in history)
        assert any(m.role == "assistant" and m.content == "Everything is running smoothly!" for m in history)

        await wrapper.disconnect()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
