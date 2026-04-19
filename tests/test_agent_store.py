"""Tests for build_bridge.agent_store — SQLite storage for BAP data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_bridge.agent_store import AgentStore


@pytest.fixture
def store(tmp_path: Path) -> AgentStore:
    """Create a fresh AgentStore with a temp DB."""
    s = AgentStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


class TestChannels:
    def test_create_channel(self, store: AgentStore):
        ch = store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        assert ch.id == "ch_1"
        assert ch.agent_id == "agt_1"
        assert ch.harness == "claude-code"
        assert ch.model == "claude-sonnet"
        assert ch.status == "active"
        assert ch.system_prompt == ""

    def test_get_channel(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        ch = store.get_channel("ch_1")
        assert ch is not None
        assert ch.id == "ch_1"

    def test_get_channel_missing(self, store: AgentStore):
        assert store.get_channel("nonexistent") is None

    def test_get_channel_by_agent_id(self, store: AgentStore):
        store.create_channel("ch_1", "agt_abc", "claude-code", "claude-sonnet")
        ch = store.get_channel_by_agent_id("agt_abc")
        assert ch is not None
        assert ch.id == "ch_1"

    def test_get_channel_by_agent_id_missing(self, store: AgentStore):
        assert store.get_channel_by_agent_id("agt_missing") is None

    def test_update_channel_status(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        store.update_channel_status("ch_1", "idle")
        ch = store.get_channel("ch_1")
        assert ch.status == "idle"

    def test_list_active_channels(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        store.create_channel("ch_2", "agt_2", "codex", "o3")
        store.update_channel_status("ch_2", "closed")

        active = store.list_active_channels()
        assert len(active) == 1
        assert active[0].id == "ch_1"

    def test_touch_channel_updates_timestamp(self, store: AgentStore):
        ch = store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        original_updated = ch.updated_at

        # Touch and re-read.
        import time
        time.sleep(0.01)
        store.touch_channel("ch_1")
        ch2 = store.get_channel("ch_1")
        assert ch2.updated_at >= original_updated


class TestChatMessages:
    def test_store_and_retrieve(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        store.store_chat_message("msg_1", "ch_1", "user", "Hello")
        store.store_chat_message("msg_2", "ch_1", "assistant", "Hi there")

        history = store.get_chat_history("ch_1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there"

    def test_empty_history(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        assert store.get_chat_history("ch_1") == []

    def test_upsert_on_duplicate_id(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        store.store_chat_message("msg_1", "ch_1", "user", "First")
        store.store_chat_message("msg_1", "ch_1", "user", "Updated")

        history = store.get_chat_history("ch_1")
        assert len(history) == 1
        assert history[0].content == "Updated"


class TestActivityLog:
    def test_store_and_retrieve(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        store.store_activity("ch_1", "text", {"text": "Thinking..."})
        store.store_activity("ch_1", "thinking", {"text": "Let me analyze..."})

        history = store.get_activity_history("ch_1")
        assert len(history) == 2
        assert history[0].type == "text"
        data0 = json.loads(history[0].data)
        assert data0["text"] == "Thinking..."

    def test_empty_activity(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        assert store.get_activity_history("ch_1") == []


class TestToolUses:
    def test_store_tool_use(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        tu = store.store_tool_use("tu_1", "ch_1", "read_file", {"path": "/src/main.py"})
        assert tu.id == "tu_1"
        assert tu.name == "read_file"
        assert tu.output is None
        assert tu.is_error is None
        assert tu.completed_at is None

    def test_store_tool_result(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        store.store_tool_use("tu_1", "ch_1", "read_file", {"path": "/src/main.py"})
        store.store_tool_result("tu_1", "file contents here", is_error=False)

        tools = store.get_tool_uses("ch_1")
        assert len(tools) == 1
        assert tools[0].output == "file contents here"
        assert tools[0].is_error is False
        assert tools[0].completed_at is not None

    def test_tool_result_error(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        store.store_tool_use("tu_1", "ch_1", "bash", {"command": "rm -rf /"})
        store.store_tool_result("tu_1", "Permission denied", is_error=True)

        tools = store.get_tool_uses("ch_1")
        assert tools[0].is_error is True

    def test_get_tool_uses_ordered(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        store.store_tool_use("tu_1", "ch_1", "read_file", {"path": "a.py"})
        store.store_tool_use("tu_2", "ch_1", "edit_file", {"path": "b.py"})

        tools = store.get_tool_uses("ch_1")
        assert len(tools) == 2
        assert tools[0].id == "tu_1"
        assert tools[1].id == "tu_2"

    def test_empty_tool_uses(self, store: AgentStore):
        store.create_channel("ch_1", "agt_1", "claude-code", "claude-sonnet")
        assert store.get_tool_uses("ch_1") == []
