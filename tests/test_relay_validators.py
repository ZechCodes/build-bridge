"""Tests for build_bridge.relay_protocol payload validators.

Each validator gets:
- A happy-path test asserting (True, "") on a minimal valid payload.
- Negative tests for each required field missing AND for type mismatches.

A `make_validation_error` test covers the error-frame correlation-token echo.

A registry-level test asserts that every action constant in the protocol has
an entry in `VALIDATORS` (no silently-unvalidated actions).
"""

from __future__ import annotations

import pytest

from build_bridge import relay_protocol as proto


# ---------------------------------------------------------------------------
# Registry coverage — every dispatched action must have a validator.
# ---------------------------------------------------------------------------


# Action constants that ARE dispatched by E2EESession._handle_data_frame.
# Response/event action names (CHANNEL_LIST, CHANNEL_CREATED, etc.) are
# device → browser only and don't need validators.
DISPATCHED_ACTIONS = {
    proto.LIST_CHANNELS, proto.CREATE_CHANNEL, proto.RENAME_CHANNEL,
    proto.UPDATE_CHANNEL, proto.DELETE_CHANNEL,
    proto.GET_MESSAGES, proto.MESSAGE, proto.RETRY_MESSAGE,
    proto.MARK_READ, proto.MARK_SEEN,
    proto.GET_ACTIVITY,
    proto.LIST_HARNESSES, proto.LIST_WORKERS,
    proto.START_AGENT, proto.STOP_AGENT, proto.CANCEL, proto.RESTART_AGENT,
    proto.INTERACTION_RESPONSE,
    proto.RESET_SESSION, proto.COMPACT_SESSION,
    proto.GET_COMPLICATIONS, proto.COMPLICATION_ACTION,
    proto.TERMINAL_EXEC, proto.TERMINAL_KILL, proto.TERMINAL_COMPLETE,
    proto.FILES_LIST, proto.FILES_CHANGES, proto.FILES_COMMITS,
    proto.FILE_READ, proto.FILE_DIFF,
    proto.URL_FETCH,
    proto.UPLOAD_CHUNK, proto.UPLOAD_COMPLETE,
}


def test_every_dispatched_action_has_a_validator():
    missing = DISPATCHED_ACTIONS - set(proto.VALIDATORS.keys())
    assert missing == set(), f"actions without validators: {missing}"


def test_no_unknown_validators_in_registry():
    """VALIDATORS shouldn't contain entries for actions that aren't dispatched."""
    extra = set(proto.VALIDATORS.keys()) - DISPATCHED_ACTIONS
    assert extra == set(), f"VALIDATORS has entries for non-dispatched actions: {extra}"


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


class TestValidateListChannels:
    def test_empty_payload_is_valid(self):
        assert proto.validate_list_channels({}) == (True, "")

    def test_extra_fields_are_ignored(self):
        assert proto.validate_list_channels({"foo": "bar"}) == (True, "")


class TestValidateCreateChannel:
    def test_valid_minimum(self):
        assert proto.validate_create_channel({"name": "general"}) == (True, "")

    def test_valid_with_options(self):
        ok, _ = proto.validate_create_channel({
            "name": "general",
            "harness": "claude-code",
            "model": "claude-sonnet-4",
            "effort": "medium",
            "system_prompt": "be helpful",
            "working_directory": "/tmp",
            "auto_approve_tools": True,
        })
        assert ok

    def test_missing_name(self):
        ok, err = proto.validate_create_channel({})
        assert not ok
        assert "name" in err

    def test_empty_name(self):
        ok, err = proto.validate_create_channel({"name": "   "})
        assert not ok
        assert "name" in err

    def test_wrong_type_name(self):
        ok, err = proto.validate_create_channel({"name": 123})
        assert not ok
        assert "name" in err

    def test_wrong_type_auto_approve(self):
        ok, err = proto.validate_create_channel({
            "name": "g", "auto_approve_tools": "yes",
        })
        assert not ok
        assert "auto_approve_tools" in err


class TestValidateRenameChannel:
    def test_valid(self):
        assert proto.validate_rename_channel(
            {"channel_id": "abc", "name": "new"}
        ) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_rename_channel({"name": "new"})
        assert not ok

    def test_missing_name(self):
        ok, _ = proto.validate_rename_channel({"channel_id": "abc"})
        assert not ok

    def test_empty_name(self):
        ok, _ = proto.validate_rename_channel({"channel_id": "abc", "name": ""})
        assert not ok


class TestValidateUpdateChannel:
    def test_valid_minimum(self):
        assert proto.validate_update_channel({"channel_id": "abc"}) == (True, "")

    def test_valid_with_options(self):
        ok, _ = proto.validate_update_channel({
            "channel_id": "abc",
            "working_directory": "/tmp",
            "model": "claude-sonnet-4",
            "effort": "low",
            "harness": "codex",
            "auto_approve_tools": False,
        })
        assert ok

    def test_missing_channel_id(self):
        ok, _ = proto.validate_update_channel({})
        assert not ok


class TestValidateDeleteChannel:
    def test_valid(self):
        assert proto.validate_delete_channel({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_delete_channel({})
        assert not ok


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestValidateGetMessages:
    def test_valid_minimum(self):
        assert proto.validate_get_messages({"channel_id": "abc"}) == (True, "")

    def test_valid_with_pagination(self):
        ok, _ = proto.validate_get_messages({
            "channel_id": "abc", "limit": 100, "before": 1234567890.5,
        })
        assert ok

    def test_missing_channel_id(self):
        ok, _ = proto.validate_get_messages({})
        assert not ok

    def test_wrong_limit_type(self):
        ok, _ = proto.validate_get_messages({"channel_id": "abc", "limit": "10"})
        assert not ok

    def test_wrong_before_type(self):
        ok, _ = proto.validate_get_messages({"channel_id": "abc", "before": "yesterday"})
        assert not ok


class TestValidateMessage:
    def test_valid_with_content(self):
        assert proto.validate_message(
            {"channel_id": "abc", "content": "hello"}
        ) == (True, "")

    def test_valid_with_attachments_only(self):
        ok, _ = proto.validate_message({
            "channel_id": "abc",
            "attachments": [{"file_id": "f1", "filename": "x.png"}],
        })
        assert ok

    def test_missing_both_content_and_attachments(self):
        ok, err = proto.validate_message({"channel_id": "abc"})
        assert not ok
        assert "content" in err or "attachments" in err

    def test_empty_content_with_no_attachments(self):
        ok, _ = proto.validate_message({"channel_id": "abc", "content": ""})
        assert not ok

    def test_missing_channel_id(self):
        ok, _ = proto.validate_message({"content": "hi"})
        assert not ok

    def test_attachment_missing_file_id(self):
        ok, _ = proto.validate_message({
            "channel_id": "abc", "attachments": [{"filename": "x.png"}],
        })
        assert not ok

    def test_attachment_not_list(self):
        ok, _ = proto.validate_message({
            "channel_id": "abc", "content": "hi", "attachments": "f1",
        })
        assert not ok


class TestValidateRetryMessage:
    def test_valid(self):
        assert proto.validate_retry_message(
            {"channel_id": "abc", "message_id": "m1"}
        ) == (True, "")

    def test_missing_message_id(self):
        ok, _ = proto.validate_retry_message({"channel_id": "abc"})
        assert not ok


class TestValidateMarkRead:
    def test_valid(self):
        assert proto.validate_mark_read(
            {"channel_id": "abc", "message_ids": ["m1", "m2"]}
        ) == (True, "")

    def test_empty_list_is_valid(self):
        ok, _ = proto.validate_mark_read({"channel_id": "abc", "message_ids": []})
        assert ok

    def test_missing_message_ids(self):
        ok, _ = proto.validate_mark_read({"channel_id": "abc"})
        assert not ok

    def test_non_list_message_ids(self):
        ok, _ = proto.validate_mark_read(
            {"channel_id": "abc", "message_ids": "m1"}
        )
        assert not ok

    def test_non_string_element(self):
        ok, _ = proto.validate_mark_read(
            {"channel_id": "abc", "message_ids": ["m1", 42]}
        )
        assert not ok


class TestValidateMarkSeen:
    def test_valid(self):
        assert proto.validate_mark_seen({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_mark_seen({})
        assert not ok


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


class TestValidateGetActivity:
    def test_valid(self):
        assert proto.validate_get_activity({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_get_activity({})
        assert not ok


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class TestValidateListHarnesses:
    def test_valid(self):
        assert proto.validate_list_harnesses({}) == (True, "")


class TestValidateListWorkers:
    def test_valid(self):
        assert proto.validate_list_workers({}) == (True, "")


class TestValidateStartAgent:
    def test_valid(self):
        assert proto.validate_start_agent(
            {"channel_id": "abc", "harness": "claude-code"}
        ) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_start_agent({"harness": "claude-code"})
        assert not ok

    def test_missing_harness(self):
        ok, _ = proto.validate_start_agent({"channel_id": "abc"})
        assert not ok


class TestValidateStopAgent:
    def test_valid(self):
        assert proto.validate_stop_agent({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_stop_agent({})
        assert not ok


class TestValidateCancel:
    def test_valid(self):
        assert proto.validate_cancel({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_cancel({})
        assert not ok


class TestValidateRestartAgent:
    def test_valid(self):
        assert proto.validate_restart_agent({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_restart_agent({})
        assert not ok


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------


class TestValidateInteractionResponse:
    def test_valid_with_selected_option(self):
        ok, _ = proto.validate_interaction_response({
            "channel_id": "abc", "interaction_id": "i1", "selected_option": "yes",
        })
        assert ok

    def test_valid_with_freeform(self):
        ok, _ = proto.validate_interaction_response({
            "channel_id": "abc", "interaction_id": "i1",
            "freeform_response": "user text",
        })
        assert ok

    def test_valid_with_no_answer_fields(self):
        # Handler is loose about which answer field is present today.
        ok, _ = proto.validate_interaction_response(
            {"channel_id": "abc", "interaction_id": "i1"}
        )
        assert ok

    def test_missing_interaction_id(self):
        ok, _ = proto.validate_interaction_response({"channel_id": "abc"})
        assert not ok

    def test_wrong_selected_options_type(self):
        ok, _ = proto.validate_interaction_response({
            "channel_id": "abc", "interaction_id": "i1",
            "selected_options": "yes",
        })
        assert not ok

    def test_wrong_step_answers_type(self):
        ok, _ = proto.validate_interaction_response({
            "channel_id": "abc", "interaction_id": "i1",
            "step_answers": "yes",
        })
        assert not ok


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestValidateResetSession:
    def test_valid(self):
        assert proto.validate_reset_session({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_reset_session({})
        assert not ok


class TestValidateCompactSession:
    def test_valid(self):
        assert proto.validate_compact_session({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_compact_session({})
        assert not ok


# ---------------------------------------------------------------------------
# Complications
# ---------------------------------------------------------------------------


class TestValidateGetComplications:
    def test_valid(self):
        assert proto.validate_get_complications({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_get_complications({})
        assert not ok


class TestValidateComplicationAction:
    def test_valid(self):
        ok, _ = proto.validate_complication_action({
            "channel_id": "abc", "complication_id": "git:/repo", "option_id": "push",
        })
        assert ok

    @pytest.mark.parametrize("missing", ["channel_id", "complication_id", "option_id"])
    def test_missing_required_field(self, missing):
        payload = {
            "channel_id": "abc",
            "complication_id": "git:/repo",
            "option_id": "push",
        }
        del payload[missing]
        ok, _ = proto.validate_complication_action(payload)
        assert not ok


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


class TestValidateTerminalExec:
    def test_valid(self):
        ok, _ = proto.validate_terminal_exec(
            {"channel_id": "abc", "command": "echo hi"}
        )
        assert ok

    def test_missing_command(self):
        ok, _ = proto.validate_terminal_exec({"channel_id": "abc"})
        assert not ok

    def test_empty_command(self):
        ok, _ = proto.validate_terminal_exec(
            {"channel_id": "abc", "command": "   "}
        )
        assert not ok


class TestValidateTerminalKill:
    def test_valid(self):
        assert proto.validate_terminal_kill({"channel_id": "abc"}) == (True, "")

    def test_missing_channel_id(self):
        ok, _ = proto.validate_terminal_kill({})
        assert not ok


class TestValidateTerminalComplete:
    def test_valid(self):
        ok, _ = proto.validate_terminal_complete({
            "channel_id": "abc", "partial": "ls", "line": "ls -",
        })
        assert ok

    def test_valid_empty_partial(self):
        # `partial` and `line` allow empty strings.
        ok, _ = proto.validate_terminal_complete({
            "channel_id": "abc", "partial": "", "line": "",
        })
        assert ok

    def test_missing_partial(self):
        ok, _ = proto.validate_terminal_complete({"channel_id": "abc", "line": ""})
        assert not ok


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class TestValidateFilesList:
    def test_valid(self):
        assert proto.validate_files_list({"channel_id": "abc"}) == (True, "")

    def test_valid_with_path(self):
        ok, _ = proto.validate_files_list({"channel_id": "abc", "path": "src"})
        assert ok

    def test_missing_channel_id(self):
        ok, _ = proto.validate_files_list({})
        assert not ok


class TestValidateFilesChanges:
    def test_valid(self):
        assert proto.validate_files_changes({"channel_id": "abc"}) == (True, "")

    def test_valid_scoped(self):
        ok, _ = proto.validate_files_changes({
            "channel_id": "abc",
            "repo_path": "subrepo",
            "newer_ref": "main",
            "older_ref": "HEAD~3",
        })
        assert ok


class TestValidateFilesCommits:
    def test_valid(self):
        assert proto.validate_files_commits({"channel_id": "abc"}) == (True, "")

    def test_valid_with_limit(self):
        ok, _ = proto.validate_files_commits({"channel_id": "abc", "limit": 10})
        assert ok

    def test_wrong_limit_type(self):
        ok, _ = proto.validate_files_commits({"channel_id": "abc", "limit": "10"})
        assert not ok


class TestValidateFileRead:
    def test_valid(self):
        ok, _ = proto.validate_file_read(
            {"channel_id": "abc", "path": "README.md"}
        )
        assert ok

    def test_missing_path(self):
        ok, _ = proto.validate_file_read({"channel_id": "abc"})
        assert not ok


class TestValidateFileDiff:
    def test_valid(self):
        ok, _ = proto.validate_file_diff(
            {"channel_id": "abc", "path": "README.md"}
        )
        assert ok

    def test_valid_staged(self):
        ok, _ = proto.validate_file_diff(
            {"channel_id": "abc", "path": "README.md", "staged": True}
        )
        assert ok

    def test_wrong_staged_type(self):
        ok, _ = proto.validate_file_diff(
            {"channel_id": "abc", "path": "README.md", "staged": "yes"}
        )
        assert not ok


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------


class TestValidateUrlFetch:
    def test_valid(self):
        assert proto.validate_url_fetch(
            {"url": "http://localhost:3000"}
        ) == (True, "")

    def test_valid_post(self):
        ok, _ = proto.validate_url_fetch({
            "url": "http://localhost:3000",
            "method": "POST",
            "body": '{"a":1}',
            "content_type": "application/json",
        })
        assert ok

    def test_method_lowercase_ok(self):
        # The handler upper-cases the method, so the validator accepts mixed case.
        ok, _ = proto.validate_url_fetch({"url": "http://x", "method": "post"})
        assert ok

    def test_invalid_method(self):
        ok, _ = proto.validate_url_fetch({"url": "http://x", "method": "PATCH"})
        assert not ok

    def test_missing_url(self):
        ok, _ = proto.validate_url_fetch({})
        assert not ok


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


class TestValidateUploadChunk:
    BASE = {
        "file_id": "f1",
        "channel_id": "c1",
        "chunk_index": 0,
        "total_size": 1024,
        "total_chunks": 1,
        "filename": "x.png",
        "data": "aGVsbG8=",
    }

    def test_valid(self):
        ok, _ = proto.validate_upload_chunk(self.BASE)
        assert ok

    def test_valid_with_optional_fields(self):
        ok, _ = proto.validate_upload_chunk({
            **self.BASE,
            "mime_type": "image/png",
            "dest_dir": "uploads",
        })
        assert ok

    @pytest.mark.parametrize("missing", [
        "file_id", "channel_id", "chunk_index", "total_size",
        "total_chunks", "filename", "data",
    ])
    def test_missing_required_field(self, missing):
        payload = dict(self.BASE)
        del payload[missing]
        ok, _ = proto.validate_upload_chunk(payload)
        assert not ok

    def test_wrong_chunk_index_type(self):
        ok, _ = proto.validate_upload_chunk({**self.BASE, "chunk_index": "0"})
        assert not ok

    def test_chunk_index_must_not_be_bool(self):
        # bool is a subclass of int but doesn't make sense here.
        ok, _ = proto.validate_upload_chunk({**self.BASE, "chunk_index": True})
        assert not ok


class TestValidateUploadComplete:
    def test_valid(self):
        ok, _ = proto.validate_upload_complete(
            {"file_id": "f1", "channel_id": "c1", "sha256": "abc"}
        )
        assert ok

    @pytest.mark.parametrize("missing", ["file_id", "channel_id", "sha256"])
    def test_missing_required_field(self, missing):
        payload = {"file_id": "f1", "channel_id": "c1", "sha256": "abc"}
        del payload[missing]
        ok, _ = proto.validate_upload_complete(payload)
        assert not ok


# ---------------------------------------------------------------------------
# make_validation_error
# ---------------------------------------------------------------------------


class TestMakeValidationError:
    def test_basic_shape(self):
        out = proto.make_validation_error(
            "rename_channel", "name required", {"channel_id": "abc"}
        )
        assert out == {
            "action": proto.ERROR,
            "error": "name required",
            "request_action": "rename_channel",
            "channel_id": "abc",
        }

    def test_echoes_correlation_tokens(self):
        out = proto.make_validation_error("terminal_exec", "command required", {
            "channel_id": "c1",
            "command_id": "cmd1",
            "file_id": "should_not_apply_here_but_echoed",
        })
        assert out["request_action"] == "terminal_exec"
        assert out["channel_id"] == "c1"
        assert out["command_id"] == "cmd1"
        assert out["file_id"] == "should_not_apply_here_but_echoed"

    def test_skips_non_string_correlation_tokens(self):
        # Non-strings (e.g. int channel_id) are silently dropped — the
        # browser wouldn't be able to route them anyway.
        out = proto.make_validation_error("rename_channel", "...", {"channel_id": 123})
        assert "channel_id" not in out


# ---------------------------------------------------------------------------
# End-to-end: dispatcher emits the error frame
# ---------------------------------------------------------------------------


class TestDispatcherEmitsValidationError:
    """E2EESession._handle_data_frame should emit a validation_error frame
    when the payload fails validation, and NOT call the handler."""

    @pytest.mark.asyncio
    async def test_missing_channel_id_in_rename(self, monkeypatch, tmp_path):
        from build_bridge.e2ee import E2EEHandler
        from build_bridge.relay_session import ActiveSession
        from build_bridge.storage import MessageStore
        from types import SimpleNamespace

        cfg = SimpleNamespace()
        store = MessageStore(db_path=tmp_path / "msg.db")
        handler = E2EEHandler(cfg, store)

        sent = []

        async def fake_send(session, ws, payload):
            sent.append(payload)

        handler._send_frame = fake_send  # type: ignore[assignment]

        session = ActiveSession(session_id="s1", session_key_b64="x" * 44)
        handler._sessions["s1"] = session

        # Frame missing the required `name` field for rename_channel.
        frame = {
            "frame_type": "data",
            "sender": "client",
            "payload": {"action": "rename_channel", "channel_id": "abc"},
        }
        await handler._session_mgr._handle_data_frame(session, frame, object())

        assert len(sent) == 1
        msg = sent[0]
        assert msg["action"] == "error"
        assert msg["request_action"] == "rename_channel"
        assert msg["channel_id"] == "abc"
        assert "name" in msg["error"]
