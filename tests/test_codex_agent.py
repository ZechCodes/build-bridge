"""Tests for the Codex Build agent runtime helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_bridge.chat_mcp import ChatMCP
from build_bridge.codex_agent import (
    CodexHarnessRuntime,
    _build_history_context,
    _extract_proposed_plan_text,
    _render_plan_text,
    _turn_id_from_params,
    _tool_input_for_item,
    _tool_name_for_item,
    _tool_result_from_item,
    create_isolated_codex_home,
)


class _FakePlanWrapper:
    def __init__(self, interaction_result: dict):
        self.interaction_result = interaction_result
        self.interactions: list[dict] = []
        self.activity_ends: list[str] = []
        self.state_updates: list[dict] = []
        self.sent_messages: list[str] = []
        self.chat_mcp = ChatMCP(on_send=self._on_send)
        self.pending_model = None
        self.pending_effort = None
        self.pending_plan_mode = None

    async def _on_send(self, message: str, suggested_actions: list[str] | None = None):
        self.sent_messages.append(message)

    async def emit_activity_end(self, reason: str):
        self.activity_ends.append(reason)

    async def request_interaction(self, **kwargs):
        self.interactions.append(kwargs)
        return self.interaction_result

    async def emit_state_update(self, **kwargs):
        self.state_updates.append(kwargs)


class _FakeCodexClient:
    def __init__(self):
        self.requests: list[tuple[str, dict]] = []

    async def send_request(self, method: str, params: dict, timeout=None):
        self.requests.append((method, params))
        return {}


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

    def test_extracts_turn_id_from_supported_shapes(self):
        assert _turn_id_from_params({"turnId": "turn-a"}) == "turn-a"
        assert _turn_id_from_params({"turn_id": "turn-b"}) == "turn-b"
        assert _turn_id_from_params({"turn": {"id": "turn-c"}}) == "turn-c"
        assert _turn_id_from_params({"turn": {"turnId": "turn-d"}}) == "turn-d"

    def test_extracts_codex_proposed_plan_markup(self):
        text = "<proposed_plan>\n\n# Plan\n\nDo the work.\n</proposed_plan>"
        assert _extract_proposed_plan_text(text) == "# Plan\n\nDo the work."

    def test_extracts_unclosed_codex_proposed_plan_markup(self):
        text = "<proposed_plan>\n\n# Plan\n\nDo the work."
        assert _extract_proposed_plan_text(text) == "# Plan\n\nDo the work."


class TestPlanReviewFlow:
    @pytest.mark.asyncio
    async def test_proposed_plan_send_is_captured_instead_of_sent_as_chat(self, tmp_path: Path):
        wrapper = _FakePlanWrapper({"cancelled": True})
        runtime = CodexHarnessRuntime(
            wrapper=wrapper,
            client=SimpleNamespace(),
            config=SimpleNamespace(auto_approve_tools=False),
            model="gpt-5.3-codex",
            initial_prompt=None,
            working_directory=str(tmp_path),
        )
        runtime.turn_id = "turn-1"

        await wrapper.chat_mcp.handle_send("<proposed_plan>\n\n# Plan\n\nDo the work.")
        await runtime._on_turn_completed({"turn": {"id": "turn-1", "status": "completed"}})
        await runtime._plan_review_task

        assert wrapper.sent_messages == []
        assert len(wrapper.interactions) == 1
        assert wrapper.interactions[0]["kind"] == "plan_review"
        assert wrapper.interactions[0]["plan"] == "# Plan\n\nDo the work."
        assert runtime.plan_mode is True

    @pytest.mark.asyncio
    async def test_normal_send_still_reaches_chat(self, tmp_path: Path):
        wrapper = _FakePlanWrapper({"cancelled": True})
        CodexHarnessRuntime(
            wrapper=wrapper,
            client=SimpleNamespace(),
            config=SimpleNamespace(auto_approve_tools=False),
            model="gpt-5.3-codex",
            initial_prompt=None,
            working_directory=str(tmp_path),
        )

        await wrapper.chat_mcp.handle_send("Normal response.")

        assert wrapper.sent_messages == ["Normal response."]

    @pytest.mark.asyncio
    async def test_approve_clears_pending_plan_mode_before_continuing(self, tmp_path: Path):
        wrapper = _FakePlanWrapper({"selected_option": "approve"})
        wrapper.pending_plan_mode = True
        client = _FakeCodexClient()
        runtime = CodexHarnessRuntime(
            wrapper=wrapper,
            client=client,
            config=SimpleNamespace(auto_approve_tools=False),
            model="gpt-5.3-codex",
            initial_prompt=None,
            working_directory=str(tmp_path),
        )
        runtime.thread_id = "thread-1"
        runtime.plan_mode = True
        runtime.latest_plan_text = "Do the work."

        await runtime._handle_plan_review()

        assert runtime.plan_mode is False
        assert wrapper.pending_plan_mode is None
        assert wrapper.state_updates == [{"plan_mode": False}]
        assert client.requests
        method, params = client.requests[-1]
        assert method == "turn/start"
        assert params["collaborationMode"]["mode"] == "default"

    @pytest.mark.asyncio
    async def test_full_plan_mode_lifecycle_starts_plan_then_exits_after_approval(self, tmp_path: Path):
        wrapper = _FakePlanWrapper({"selected_option": "approve"})
        client = _FakeCodexClient()
        runtime = CodexHarnessRuntime(
            wrapper=wrapper,
            client=client,
            config=SimpleNamespace(auto_approve_tools=False),
            model="gpt-5.3-codex",
            initial_prompt=None,
            working_directory=str(tmp_path),
        )
        runtime.thread_id = "thread-1"

        wrapper.pending_plan_mode = True
        await runtime._start_or_steer("Plan this change.")

        first_method, first_params = client.requests[-1]
        assert first_method == "turn/start"
        assert first_params["collaborationMode"]["mode"] == "plan"
        assert runtime.plan_mode is True
        assert wrapper.pending_plan_mode is None

        await runtime._on_turn_started({"turn": {"id": "turn-1", "status": "inProgress"}})
        await wrapper.chat_mcp.handle_send("<proposed_plan>\n\n# Plan\n\nDo the work.</proposed_plan>")
        await runtime._on_turn_completed({"turn": {"id": "turn-1", "status": "completed"}})
        await runtime._plan_review_task

        assert wrapper.sent_messages == []
        assert len(wrapper.interactions) == 1
        assert wrapper.interactions[0]["kind"] == "plan_review"
        assert runtime.plan_mode is False
        assert wrapper.state_updates[-1] == {"plan_mode": False}

        second_method, second_params = client.requests[-1]
        assert second_method == "turn/start"
        assert second_params["input"][0]["text"] == "The user approved the plan. Proceed with the implementation."
        assert second_params["collaborationMode"]["mode"] == "default"

    @pytest.mark.asyncio
    async def test_completed_turn_without_id_still_requests_plan_review(self, tmp_path: Path):
        wrapper = _FakePlanWrapper({"cancelled": True})
        runtime = CodexHarnessRuntime(
            wrapper=wrapper,
            client=SimpleNamespace(),
            config=SimpleNamespace(auto_approve_tools=False),
            model="gpt-5.3-codex",
            initial_prompt=None,
            working_directory=str(tmp_path),
        )
        runtime.plan_mode = True

        await runtime._on_turn_plan_updated({
            "turnId": "turn-1",
            "plan": [{"step": "Patch bridge", "status": "pending"}],
        })
        await runtime._on_turn_completed({"turn": {"status": "completed"}})
        await runtime._plan_review_task

        assert wrapper.interactions
        assert wrapper.interactions[0]["kind"] == "plan_review"
        assert wrapper.interactions[0]["plan"] == "[ ] Patch bridge"

    @pytest.mark.asyncio
    async def test_completed_turn_id_mismatch_still_requests_latest_plan_review(self, tmp_path: Path):
        wrapper = _FakePlanWrapper({"cancelled": True})
        runtime = CodexHarnessRuntime(
            wrapper=wrapper,
            client=SimpleNamespace(),
            config=SimpleNamespace(auto_approve_tools=False),
            model="gpt-5.3-codex",
            initial_prompt=None,
            working_directory=str(tmp_path),
        )
        runtime.plan_mode = True

        await runtime._on_turn_plan_updated({
            "turnId": "plan-event-id",
            "plan": [{"step": "Patch bridge", "status": "pending"}],
        })
        await runtime._on_turn_completed({"turn": {"id": "completed-event-id", "status": "completed"}})
        await runtime._plan_review_task

        assert len(wrapper.interactions) == 1
        assert wrapper.interactions[0]["kind"] == "plan_review"

    @pytest.mark.asyncio
    async def test_duplicate_completed_turn_does_not_request_duplicate_review(self, tmp_path: Path):
        wrapper = _FakePlanWrapper({"cancelled": True})
        runtime = CodexHarnessRuntime(
            wrapper=wrapper,
            client=SimpleNamespace(),
            config=SimpleNamespace(auto_approve_tools=False),
            model="gpt-5.3-codex",
            initial_prompt=None,
            working_directory=str(tmp_path),
        )
        runtime.plan_mode = True

        await runtime._on_turn_plan_updated({
            "turnId": "turn-1",
            "plan": [{"step": "Patch bridge", "status": "pending"}],
        })
        await runtime._on_turn_completed({"turn": {"id": "turn-1", "status": "completed"}})
        await runtime._plan_review_task
        await runtime._on_turn_completed({"turn": {"id": "turn-1", "status": "completed"}})

        assert len(wrapper.interactions) == 1


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
