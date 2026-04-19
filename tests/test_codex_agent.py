"""Tests for the Codex Build agent runtime helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from build_bridge.codex_agent import (
    _build_history_context,
    _render_plan_text,
    _tool_input_for_item,
    _tool_name_for_item,
    _tool_result_from_item,
    create_isolated_codex_home,
)


class TestHistoryContext:
    def test_empty_history(self):
        assert _build_history_context([]) == ""

    def test_renders_recent_messages(self):
        text = _build_history_context([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        assert "Recent conversation history" in text
        assert "[user] hello" in text
        assert "[assistant] hi" in text


class TestPlanRendering:
    def test_renders_explanation_and_steps(self):
        text = _render_plan_text(
            "Need to wire Codex app-server.",
            [
                {"step": "Create client", "status": "completed"},
                {"step": "Wire bridge", "status": "inProgress"},
                {"step": "Run tests", "status": "pending"},
            ],
        )
        assert "Need to wire Codex app-server." in text
        assert "[x] Create client" in text
        assert "[>] Wire bridge" in text
        assert "[ ] Run tests" in text


class TestToolMapping:
    def test_command_execution_maps_to_bash(self):
        item = {
            "type": "commandExecution",
            "command": "git status",
            "cwd": "/repo",
            "commandActions": [],
        }
        assert _tool_name_for_item(item) == "Bash"
        tool_input = _tool_input_for_item(item)
        assert tool_input["command"] == "git status"
        assert tool_input["cwd"] == "/repo"

    def test_dynamic_tool_result_uses_content_items(self):
        content, is_error = _tool_result_from_item(
            {
                "type": "dynamicToolCall",
                "contentItems": [{"type": "inputText", "text": "done"}],
                "success": True,
            },
            "",
        )
        assert content == "done"
        assert is_error is False

    def test_command_result_uses_buffer_and_exit_code(self):
        content, is_error = _tool_result_from_item(
            {
                "type": "commandExecution",
                "aggregatedOutput": "",
                "exitCode": 1,
                "status": "completed",
            },
            "stderr: failed",
        )
        assert content == "stderr: failed"
        assert is_error is True


class TestIsolatedHome:
    def test_writes_bridge_config_and_copies_auth(self, tmp_path: Path, monkeypatch):
        real_home = tmp_path / "real-home"
        (real_home / ".codex").mkdir(parents=True)
        (real_home / ".codex" / "auth.json").write_text('{"token":"x"}', encoding="utf-8")
        monkeypatch.setenv("HOME", str(real_home))

        runtime_home = create_isolated_codex_home(
            bridge_socket="/tmp/build-bridge.sock",
            bridge_token="secret",
            trusted_project="/tmp/project",
        )
        try:
            config = (runtime_home / ".codex" / "config.toml").read_text(encoding="utf-8")
            assert "[mcp_servers.build-chat]" in config
            assert 'BUILD_CHAT_BRIDGE_TOKEN = "secret"' in config
            assert '[projects."/tmp/project"]' in config

            auth = json.loads((runtime_home / ".codex" / "auth.json").read_text(encoding="utf-8"))
            assert auth["token"] == "x"
        finally:
            shutil.rmtree(runtime_home, ignore_errors=True)
