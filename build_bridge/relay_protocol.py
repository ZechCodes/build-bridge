"""Build Relay Protocol — action name + error code constants and payload validators.

The wire format and full semantics are defined in build-bridge/RELAY_PROTOCOL.md.
This module exposes:

- Action / error / frame-field name constants so the dispatcher and handler
  modules don't have to repeat string literals.
- Per-action payload validators (one `validate_<action>` function per request
  action) plus a `VALIDATORS` registry the dispatcher consults before invoking
  a handler.
- `make_validation_error` — the canonical shape for validation-failure frames
  emitted on the wire.

Mirrors agent_protocol.py's style: plain `(bool, str)` validators, no schemas
DSL. The markdown spec is the source of truth for semantics; these validators
codify field name / type / required-ness.
"""

from __future__ import annotations

from typing import Any, Callable


# ---------------------------------------------------------------------------
# Request action names (browser → device)
# ---------------------------------------------------------------------------

# Channels
LIST_CHANNELS = "list_channels"
CREATE_CHANNEL = "create_channel"
RENAME_CHANNEL = "rename_channel"
UPDATE_CHANNEL = "update_channel"
DELETE_CHANNEL = "delete_channel"

# Messages
GET_MESSAGES = "get_messages"
MESSAGE = "message"
RETRY_MESSAGE = "retry_message"
MARK_READ = "mark_read"
MARK_SEEN = "mark_seen"

# Activity
GET_ACTIVITY = "get_activity"

# Agents
LIST_HARNESSES = "list_harnesses"
LIST_WORKERS = "list_workers"
START_AGENT = "start_agent"
STOP_AGENT = "stop_agent"
CANCEL = "cancel"
RESTART_AGENT = "restart_agent"

# Interactions
INTERACTION_RESPONSE = "interaction_response"

# Sessions
RESET_SESSION = "reset_session"
COMPACT_SESSION = "compact_session"

# Complications
GET_COMPLICATIONS = "get_complications"
COMPLICATION_ACTION = "complication:action"

# Terminal
TERMINAL_EXEC = "terminal_exec"
TERMINAL_KILL = "terminal_kill"
TERMINAL_COMPLETE = "terminal_complete"

# Files
FILES_LIST = "files_list"
FILES_CHANGES = "files_changes"
FILES_COMMITS = "files_commits"
FILE_READ = "file_read"
FILE_DIFF = "file_diff"
CHAT_IMAGE_FETCH = "chat_image_fetch"

# URL
URL_FETCH = "url_fetch"

# Uploads
UPLOAD_CHUNK = "upload_chunk"
UPLOAD_COMPLETE = "upload_complete"


# ---------------------------------------------------------------------------
# Response / event action names (device → browser)
# ---------------------------------------------------------------------------

# Channels
CHANNEL_LIST = "channel_list"
CHANNEL_CREATED = "channel_created"
CHANNEL_RENAMED = "channel_renamed"
CHANNEL_UPDATED = "channel_updated"
CHANNEL_DELETED = "channel_deleted"

# Messages
MESSAGES = "messages"
DELIVERED = "delivered"
DELIVERY_FAILED = "delivery_failed"
READ = "read"

# Activity
ACTIVITY_HISTORY = "activity_history"

# Agents
HARNESS_LIST = "harness_list"
WORKER_LIST = "worker_list"
AGENT_STARTED = "agent_started"
AGENT_STOPPED = "agent_stopped"
AGENT_RESTARTED = "agent_restarted"
CANCEL_ACK = "cancel_ack"

# Sessions
SESSION_RESET = "session_reset"
COMPACT_STARTED = "compact_started"

# Complications
COMPLICATIONS = "complications"
COMPLICATION_UPDATE = "complication:update"
COMPLICATION_REMOVE = "complication:remove"

# Terminal
TERMINAL_OUTPUT = "terminal_output"
TERMINAL_COMPLETIONS = "terminal_completions"

# Files
FILES_LIST_RESULT = "files_list_result"
FILES_CHANGES_RESULT = "files_changes_result"
FILES_COMMITS_RESULT = "files_commits_result"
FILE_READ_RESULT = "file_read_result"
FILE_DIFF_RESULT = "file_diff_result"
CHAT_IMAGE_RESULT = "chat_image_result"

# URL
URL_FETCH_RESULT = "url_fetch_result"

# Uploads
CHUNK_ACK = "chunk_ack"
UPLOAD_ACCEPTED = "upload_accepted"
UPLOAD_ERROR = "upload_error"

# Cross-cutting
SYSTEM_MESSAGE = "system_message"
AGENT_EVENT = "agent_event"
ERROR = "error"


# ---------------------------------------------------------------------------
# Frame field values (inner decrypted frame)
# ---------------------------------------------------------------------------

FRAME_TYPE_DATA = "data"
FRAME_TYPE_CLOSE = "close"

SENDER_CLIENT = "client"
SENDER_DEVICE = "device"

ROUTE_TO_CLIENT = "client"
ROUTE_TO_DEVICE = "device"


# ---------------------------------------------------------------------------
# Outer WebSocket message types
# ---------------------------------------------------------------------------

WS_SESSION_INIT = "session_init"
WS_SESSION_ACCEPT = "session_accept"
WS_E2EE_ENVELOPE = "e2ee_envelope"


# ---------------------------------------------------------------------------
# Default action when payload.action is absent (legacy behaviour)
# ---------------------------------------------------------------------------

DEFAULT_ACTION = MESSAGE


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
#
# Each validator takes the inner-frame `payload` dict and returns
# `(is_valid, error_message)`. The dispatcher in
# `E2EESession._handle_data_frame` consults the `VALIDATORS` registry; if a
# validator returns `(False, msg)`, the dispatcher emits an error frame
# (see `make_validation_error`) and skips the handler.
#
# Conventions:
# - "required" means the field must be present, not None, the right type, and
#   non-empty where the type supports emptiness (str/list/dict).
# - Optional fields are only type-checked when present; absent is fine.
# - Validators do NOT enforce business rules (channel must exist, agent must
#   be running, path must be safe) — those remain handler concerns.

ValidatorFn = Callable[[dict[str, Any]], "tuple[bool, str]"]


def _require_str(payload: dict[str, Any], field: str, *, allow_empty: bool = False) -> str:
    """Return "" if valid, else an error message. Helper for validators."""
    if field not in payload:
        return f"missing required field: {field}"
    val = payload[field]
    if not isinstance(val, str):
        return f"{field} must be a string"
    if not allow_empty and not val.strip():
        return f"{field} required"
    return ""


def _check_optional_str(payload: dict[str, Any], field: str) -> str:
    if field in payload and payload[field] is not None and not isinstance(payload[field], str):
        return f"{field} must be a string"
    return ""


def _check_optional_bool(payload: dict[str, Any], field: str) -> str:
    if field in payload and payload[field] is not None and not isinstance(payload[field], bool):
        return f"{field} must be a boolean"
    return ""


def _check_optional_int(payload: dict[str, Any], field: str) -> str:
    # Accept bool subclass of int? No — booleans should be bools, not ints.
    if field in payload and payload[field] is not None:
        val = payload[field]
        if isinstance(val, bool) or not isinstance(val, int):
            return f"{field} must be an integer"
    return ""


def _check_optional_number(payload: dict[str, Any], field: str) -> str:
    """For float-or-int fields like `before` (timestamp)."""
    if field in payload and payload[field] is not None:
        val = payload[field]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return f"{field} must be a number"
    return ""


def _check_required_int(payload: dict[str, Any], field: str) -> str:
    if field not in payload:
        return f"missing required field: {field}"
    val = payload[field]
    if isinstance(val, bool) or not isinstance(val, int):
        return f"{field} must be an integer"
    return ""


# ---------------------------------------------------------------------------
# Channels (§5.1)
# ---------------------------------------------------------------------------


def validate_list_channels(payload: dict[str, Any]) -> tuple[bool, str]:
    return True, ""


def validate_create_channel(payload: dict[str, Any]) -> tuple[bool, str]:
    err = _require_str(payload, "name")
    if err:
        return False, err
    for f in ("harness", "model", "effort", "system_prompt", "working_directory"):
        if e := _check_optional_str(payload, f):
            return False, e
    if e := _check_optional_bool(payload, "auto_approve_tools"):
        return False, e
    return True, ""


def validate_rename_channel(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "name"):
        return False, err
    return True, ""


def validate_update_channel(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    for f in ("working_directory", "model", "effort", "harness"):
        if e := _check_optional_str(payload, f):
            return False, e
    if e := _check_optional_bool(payload, "auto_approve_tools"):
        return False, e
    return True, ""


def validate_delete_channel(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    return True, ""


# ---------------------------------------------------------------------------
# Messages (§5.2)
# ---------------------------------------------------------------------------


def validate_get_messages(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if e := _check_optional_int(payload, "limit"):
        return False, e
    if e := _check_optional_number(payload, "before"):
        return False, e
    return True, ""


def validate_message(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    content = payload.get("content")
    attachments = payload.get("attachments")
    has_content = isinstance(content, str) and bool(content)
    has_attachments = isinstance(attachments, list) and bool(attachments)
    if not has_content and not has_attachments:
        return False, "message requires content or attachments"
    if content is not None and not isinstance(content, str):
        return False, "content must be a string"
    if attachments is not None:
        if not isinstance(attachments, list):
            return False, "attachments must be a list"
        for i, att in enumerate(attachments):
            if not isinstance(att, dict):
                return False, f"attachments[{i}] must be an object"
            if "file_id" not in att or not isinstance(att["file_id"], str):
                return False, f"attachments[{i}].file_id must be a string"
    if e := _check_optional_bool(payload, "plan_mode"):
        return False, e
    if e := _check_optional_str(payload, "model"):
        return False, e
    if e := _check_optional_str(payload, "effort"):
        return False, e
    return True, ""


def validate_retry_message(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "message_id"):
        return False, err
    return True, ""


def validate_mark_read(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if "message_ids" not in payload:
        return False, "missing required field: message_ids"
    mids = payload["message_ids"]
    if not isinstance(mids, list):
        return False, "message_ids must be a list"
    for i, m in enumerate(mids):
        if not isinstance(m, str):
            return False, f"message_ids[{i}] must be a string"
    return True, ""


def validate_mark_seen(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    return True, ""


# ---------------------------------------------------------------------------
# Activity (§5.3)
# ---------------------------------------------------------------------------


def validate_get_activity(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    return True, ""


# ---------------------------------------------------------------------------
# Agents (§5.4)
# ---------------------------------------------------------------------------


def validate_list_harnesses(payload: dict[str, Any]) -> tuple[bool, str]:
    return True, ""


def validate_list_workers(payload: dict[str, Any]) -> tuple[bool, str]:
    return True, ""


def validate_start_agent(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "harness"):
        return False, err
    for f in ("model", "system_prompt", "working_directory"):
        if e := _check_optional_str(payload, f):
            return False, e
    return True, ""


def validate_stop_agent(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    return True, ""


def validate_cancel(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    return True, ""


def validate_restart_agent(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    return True, ""


# ---------------------------------------------------------------------------
# Interactions (§5.5)
# ---------------------------------------------------------------------------


def validate_interaction_response(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "interaction_id"):
        return False, err
    # Answer fields are all optional and loose-typed per spec drift note —
    # the handler currently accepts any combination. Just type-check.
    if e := _check_optional_str(payload, "selected_option"):
        return False, e
    if e := _check_optional_str(payload, "freeform_response"):
        return False, e
    if "selected_options" in payload and payload["selected_options"] is not None:
        if not isinstance(payload["selected_options"], list):
            return False, "selected_options must be a list"
    if "step_answers" in payload and payload["step_answers"] is not None:
        if not isinstance(payload["step_answers"], dict):
            return False, "step_answers must be an object"
    return True, ""


# ---------------------------------------------------------------------------
# Sessions (§5.6)
# ---------------------------------------------------------------------------


def validate_reset_session(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    return True, ""


def validate_compact_session(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    return True, ""


# ---------------------------------------------------------------------------
# Complications (§5.7)
# ---------------------------------------------------------------------------


def validate_get_complications(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    return True, ""


def validate_complication_action(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "complication_id"):
        return False, err
    if err := _require_str(payload, "option_id"):
        return False, err
    return True, ""


# ---------------------------------------------------------------------------
# Terminal (§5.8)
# ---------------------------------------------------------------------------


def validate_terminal_exec(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "command"):
        return False, err
    if e := _check_optional_str(payload, "command_id"):
        return False, e
    if e := _check_optional_str(payload, "cwd"):
        return False, e
    return True, ""


def validate_terminal_kill(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if e := _check_optional_str(payload, "command_id"):
        return False, e
    return True, ""


def validate_terminal_complete(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "partial", allow_empty=True):
        return False, err
    if err := _require_str(payload, "line", allow_empty=True):
        return False, err
    if e := _check_optional_str(payload, "cwd"):
        return False, e
    return True, ""


# ---------------------------------------------------------------------------
# Files (§5.9)
# ---------------------------------------------------------------------------


def validate_files_list(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if e := _check_optional_str(payload, "path"):
        return False, e
    return True, ""


def validate_files_changes(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    for f in ("repo_path", "newer_ref", "older_ref"):
        if e := _check_optional_str(payload, f):
            return False, e
    return True, ""


def validate_files_commits(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if e := _check_optional_str(payload, "repo_path"):
        return False, e
    if e := _check_optional_int(payload, "limit"):
        return False, e
    return True, ""


def validate_file_read(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "path"):
        return False, err
    if e := _check_optional_int(payload, "offset"):
        return False, e
    if e := _check_optional_int(payload, "limit"):
        return False, e
    return True, ""


def validate_file_diff(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "path"):
        return False, err
    if e := _check_optional_bool(payload, "staged"):
        return False, e
    for f in ("repo_path", "newer_ref", "older_ref"):
        if e := _check_optional_str(payload, f):
            return False, e
    return True, ""


def validate_chat_image_fetch(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "path"):
        return False, err
    return True, ""


# ---------------------------------------------------------------------------
# URL (§5.10)
# ---------------------------------------------------------------------------


def validate_url_fetch(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "url"):
        return False, err
    if "method" in payload and payload["method"] is not None:
        m = payload["method"]
        if not isinstance(m, str):
            return False, "method must be a string"
        if m.upper() not in ("GET", "POST"):
            return False, "method must be GET or POST"
    for f in ("request_id", "tab_id", "body", "content_type"):
        if e := _check_optional_str(payload, f):
            return False, e
    return True, ""


# ---------------------------------------------------------------------------
# Uploads (§5.11)
# ---------------------------------------------------------------------------


def validate_upload_chunk(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "file_id"):
        return False, err
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _check_required_int(payload, "chunk_index"):
        return False, err
    if err := _check_required_int(payload, "total_size"):
        return False, err
    if err := _check_required_int(payload, "total_chunks"):
        return False, err
    if err := _require_str(payload, "filename"):
        return False, err
    if err := _require_str(payload, "data", allow_empty=True):
        return False, err
    if e := _check_optional_str(payload, "mime_type"):
        return False, e
    if e := _check_optional_str(payload, "dest_dir"):
        return False, e
    return True, ""


def validate_upload_complete(payload: dict[str, Any]) -> tuple[bool, str]:
    if err := _require_str(payload, "file_id"):
        return False, err
    if err := _require_str(payload, "channel_id"):
        return False, err
    if err := _require_str(payload, "sha256"):
        return False, err
    return True, ""


# ---------------------------------------------------------------------------
# Registry — action name -> validator function
# ---------------------------------------------------------------------------

VALIDATORS: dict[str, ValidatorFn] = {
    # Channels
    LIST_CHANNELS: validate_list_channels,
    CREATE_CHANNEL: validate_create_channel,
    RENAME_CHANNEL: validate_rename_channel,
    UPDATE_CHANNEL: validate_update_channel,
    DELETE_CHANNEL: validate_delete_channel,
    # Messages
    GET_MESSAGES: validate_get_messages,
    MESSAGE: validate_message,
    RETRY_MESSAGE: validate_retry_message,
    MARK_READ: validate_mark_read,
    MARK_SEEN: validate_mark_seen,
    # Activity
    GET_ACTIVITY: validate_get_activity,
    # Agents
    LIST_HARNESSES: validate_list_harnesses,
    LIST_WORKERS: validate_list_workers,
    START_AGENT: validate_start_agent,
    STOP_AGENT: validate_stop_agent,
    CANCEL: validate_cancel,
    RESTART_AGENT: validate_restart_agent,
    # Interactions
    INTERACTION_RESPONSE: validate_interaction_response,
    # Sessions
    RESET_SESSION: validate_reset_session,
    COMPACT_SESSION: validate_compact_session,
    # Complications
    GET_COMPLICATIONS: validate_get_complications,
    COMPLICATION_ACTION: validate_complication_action,
    # Terminal
    TERMINAL_EXEC: validate_terminal_exec,
    TERMINAL_KILL: validate_terminal_kill,
    TERMINAL_COMPLETE: validate_terminal_complete,
    # Files
    FILES_LIST: validate_files_list,
    FILES_CHANGES: validate_files_changes,
    FILES_COMMITS: validate_files_commits,
    FILE_READ: validate_file_read,
    FILE_DIFF: validate_file_diff,
    CHAT_IMAGE_FETCH: validate_chat_image_fetch,
    # URL
    URL_FETCH: validate_url_fetch,
    # Uploads
    UPLOAD_CHUNK: validate_upload_chunk,
    UPLOAD_COMPLETE: validate_upload_complete,
}


# Tokens echoed back in a validation_error so the browser can route the
# response to the right request. Order matters for stable serialization.
_CORRELATION_TOKENS = ("message_id", "channel_id", "command_id", "file_id", "request_id")


def make_validation_error(
    action: str,
    err: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the error frame for a validator failure.

    Matches the existing spec §7 generic-error shape (`{action: "error",
    error: "..."}`) plus a `request_action` field naming the action that
    failed, plus any correlation tokens (`channel_id`, `command_id`,
    `file_id`, `request_id`, `message_id`) that were present in the
    request payload — so the browser can route the response.
    """
    out: dict[str, Any] = {
        "action": ERROR,
        "error": err,
        "request_action": action,
    }
    for token in _CORRELATION_TOKENS:
        if token in payload and isinstance(payload[token], str):
            out[token] = payload[token]
    return out
