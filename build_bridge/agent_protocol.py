"""Build Agent Protocol — message types, envelope helpers, and validation.

Implements the BAP v1.0-draft message envelope format, type constants,
direction constraints, capability mapping, and validation functions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


# Protocol version.
PROTOCOL_VERSION = 1

# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

# Agent namespace (no capability required).
AGENT_HELLO = "agent.hello"
AGENT_CONFIGURED = "agent.configured"
AGENT_SHUTDOWN = "agent.shutdown"
AGENT_GOODBYE = "agent.goodbye"
AGENT_ERROR = "agent.error"

# Chat namespace (requires "chat" capability).
CHAT_MESSAGE = "chat.message"
CHAT_RESPONSE = "chat.response"
CHAT_CANCEL = "chat.cancel"

# Activity namespace (requires "activity" capability).
ACTIVITY_DELTA = "activity.delta"
ACTIVITY_PING = "activity.ping"
ACTIVITY_END = "activity.end"

# Tool namespace (requires "tools" capability).
TOOL_USE = "tool.use"
TOOL_RESULT = "tool.result"

# Interaction namespace (requires "interactions" capability).
INTERACTION_REQUEST = "interaction.request"
INTERACTION_RESPONSE = "interaction.response"

# Agent state (no capability required — part of agent namespace).
AGENT_STATE_UPDATE = "agent.state_update"
AGENT_SYSTEM_MESSAGE = "agent.system_message"
# Workspace filesystem change events (no capability required).
# Payload: {"paths": ["rel/path.ts", ...]}
AGENT_FILE_CHANGES = "agent.file_changes"

# ---------------------------------------------------------------------------
# Direction constraints
# ---------------------------------------------------------------------------

AGENT_TO_CLIENT: frozenset[str] = frozenset({
    AGENT_HELLO, AGENT_GOODBYE, AGENT_STATE_UPDATE, AGENT_SYSTEM_MESSAGE,
    AGENT_FILE_CHANGES,
    CHAT_RESPONSE,
    ACTIVITY_DELTA, ACTIVITY_PING, ACTIVITY_END,
    TOOL_USE, TOOL_RESULT,
    INTERACTION_REQUEST,
})

CLIENT_TO_AGENT: frozenset[str] = frozenset({
    AGENT_CONFIGURED, AGENT_SHUTDOWN,
    CHAT_MESSAGE, CHAT_CANCEL,
    INTERACTION_RESPONSE,
})

BIDIRECTIONAL: frozenset[str] = frozenset({AGENT_ERROR})

ALL_TYPES: frozenset[str] = AGENT_TO_CLIENT | CLIENT_TO_AGENT | BIDIRECTIONAL

# ---------------------------------------------------------------------------
# Capability -> message type mapping
# ---------------------------------------------------------------------------

CAPABILITY_TYPES: dict[str, frozenset[str]] = {
    "chat": frozenset({CHAT_MESSAGE, CHAT_RESPONSE, CHAT_CANCEL}),
    "activity": frozenset({ACTIVITY_DELTA, ACTIVITY_PING, ACTIVITY_END}),
    "tools": frozenset({TOOL_USE, TOOL_RESULT}),
    "interactions": frozenset({INTERACTION_REQUEST, INTERACTION_RESPONSE}),
}

AGENT_NAMESPACE_TYPES: frozenset[str] = frozenset({
    AGENT_HELLO, AGENT_CONFIGURED, AGENT_SHUTDOWN, AGENT_GOODBYE, AGENT_ERROR,
    AGENT_STATE_UPDATE, AGENT_FILE_CHANGES,
})

# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

ERROR_INVALID_MESSAGE = "invalid_message"
ERROR_UNKNOWN_TYPE = "unknown_type"
ERROR_PROTOCOL_VIOLATION = "protocol_violation"
ERROR_CAPABILITY_MISMATCH = "capability_mismatch"
ERROR_VERSION_MISMATCH = "version_mismatch"
ERROR_INTERNAL_ERROR = "internal_error"
ERROR_RECONNECT_FAILED = "reconnect_failed"

# ---------------------------------------------------------------------------
# Shutdown reasons
# ---------------------------------------------------------------------------

SHUTDOWN_USER_CLOSED = "user_closed"
SHUTDOWN_CLIENT_SHUTDOWN = "client_shutdown"
SHUTDOWN_TIMEOUT = "timeout"
SHUTDOWN_ERROR = "error"

# ---------------------------------------------------------------------------
# Activity end reasons
# ---------------------------------------------------------------------------

END_COMPLETE = "complete"
END_CANCELLED = "cancelled"
END_ERROR = "error"
END_LIMIT = "limit"
END_WAITING = "waiting"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_id() -> str:
    """Generate a unique, sortable message ID."""
    return f"msg_{uuid.uuid4().hex[:24]}"


def now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def make_envelope(
    msg_type: str,
    payload: dict[str, Any],
    ref: str | None = None,
    msg_id: str | None = None,
) -> dict[str, Any]:
    """Create a BAP message envelope."""
    return {
        "v": PROTOCOL_VERSION,
        "id": msg_id or generate_id(),
        "ref": ref,
        "type": msg_type,
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_envelope(data: Any) -> tuple[bool, str]:
    """Validate envelope structure. Returns (valid, error_message)."""
    if not isinstance(data, dict):
        return False, "Message must be a JSON object"

    if "v" not in data:
        return False, "Missing required field: v"
    if data["v"] != PROTOCOL_VERSION:
        return False, f"Unsupported protocol version: {data['v']}"

    if "id" not in data or not isinstance(data["id"], str):
        return False, "Missing or invalid field: id"

    if "type" not in data or not isinstance(data["type"], str):
        return False, "Missing or invalid field: type"

    if "payload" not in data or not isinstance(data["payload"], dict):
        return False, "Missing or invalid field: payload"

    return True, ""


def validate_agent_hello(payload: dict[str, Any]) -> tuple[bool, str]:
    """Validate agent.hello payload fields."""
    required = ("agent_id", "harness", "capabilities", "model", "reconnect")
    for field in required:
        if field not in payload:
            return False, f"agent.hello missing required field: {field}"

    if not isinstance(payload["capabilities"], list):
        return False, "agent.hello capabilities must be an array"

    if not isinstance(payload["reconnect"], bool):
        return False, "agent.hello reconnect must be a boolean"

    return True, ""


def check_capability(
    msg_type: str,
    capabilities: set[str],
) -> tuple[bool, str]:
    """Check whether a message type is allowed given the agent's capabilities.

    Agent namespace types require no capability.
    Returns (allowed, error_message).
    """
    if msg_type in AGENT_NAMESPACE_TYPES:
        return True, ""

    for cap, types in CAPABILITY_TYPES.items():
        if msg_type in types:
            if cap in capabilities:
                return True, ""
            return False, f"Message type {msg_type} requires capability '{cap}'"

    # Unknown type in a known namespace — silently allow per §11.2.
    namespace = msg_type.split(".")[0] if "." in msg_type else ""
    if namespace in ("agent", "chat", "activity", "tool"):
        return True, ""

    return False, f"Unknown message type namespace: {msg_type}"
