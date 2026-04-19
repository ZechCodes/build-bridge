"""Tests for build_bridge.agent_protocol — envelope, validation, capabilities."""

from __future__ import annotations

import pytest

from build_bridge.agent_protocol import (
    ACTIVITY_DELTA,
    AGENT_CONFIGURED,
    AGENT_ERROR,
    AGENT_HELLO,
    AGENT_TO_CLIENT,
    ALL_TYPES,
    BIDIRECTIONAL,
    CAPABILITY_TYPES,
    CHAT_MESSAGE,
    CHAT_RESPONSE,
    CLIENT_TO_AGENT,
    ERROR_INVALID_MESSAGE,
    PROTOCOL_VERSION,
    TOOL_USE,
    check_capability,
    generate_id,
    make_envelope,
    now_iso,
    validate_agent_hello,
    validate_envelope,
)


class TestGenerateId:
    def test_returns_string(self):
        assert isinstance(generate_id(), str)

    def test_starts_with_prefix(self):
        assert generate_id().startswith("msg_")

    def test_unique(self):
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100


class TestNowIso:
    def test_returns_iso_string(self):
        ts = now_iso()
        assert "T" in ts
        assert "+" in ts or "Z" in ts or ts.endswith("+00:00")


class TestMakeEnvelope:
    def test_basic_envelope(self):
        env = make_envelope("agent.hello", {"agent_id": "agt_1"})
        assert env["v"] == PROTOCOL_VERSION
        assert env["type"] == "agent.hello"
        assert env["payload"]["agent_id"] == "agt_1"
        assert env["ref"] is None
        assert env["id"].startswith("msg_")

    def test_with_ref(self):
        env = make_envelope("agent.configured", {}, ref="msg_hello123")
        assert env["ref"] == "msg_hello123"

    def test_custom_id(self):
        env = make_envelope("chat.message", {}, msg_id="custom_id_1")
        assert env["id"] == "custom_id_1"


class TestValidateEnvelope:
    def test_valid_envelope(self):
        env = make_envelope("agent.hello", {"agent_id": "a"})
        valid, err = validate_envelope(env)
        assert valid
        assert err == ""

    def test_not_a_dict(self):
        valid, err = validate_envelope("not a dict")
        assert not valid
        assert "JSON object" in err

    def test_missing_version(self):
        valid, err = validate_envelope({"id": "x", "type": "x", "payload": {}})
        assert not valid
        assert "v" in err

    def test_wrong_version(self):
        valid, err = validate_envelope({"v": 99, "id": "x", "type": "x", "payload": {}})
        assert not valid
        assert "version" in err

    def test_missing_id(self):
        valid, err = validate_envelope({"v": 1, "type": "x", "payload": {}})
        assert not valid
        assert "id" in err

    def test_missing_type(self):
        valid, err = validate_envelope({"v": 1, "id": "x", "payload": {}})
        assert not valid
        assert "type" in err

    def test_missing_payload(self):
        valid, err = validate_envelope({"v": 1, "id": "x", "type": "x"})
        assert not valid
        assert "payload" in err

    def test_payload_not_dict(self):
        valid, err = validate_envelope({"v": 1, "id": "x", "type": "x", "payload": "str"})
        assert not valid
        assert "payload" in err


class TestValidateAgentHello:
    def _valid_payload(self):
        return {
            "agent_id": "agt_1",
            "harness": "claude-code",
            "capabilities": ["chat", "activity", "tools"],
            "model": "claude-sonnet-4-20250514",
            "reconnect": False,
        }

    def test_valid(self):
        valid, err = validate_agent_hello(self._valid_payload())
        assert valid
        assert err == ""

    def test_missing_agent_id(self):
        p = self._valid_payload()
        del p["agent_id"]
        valid, err = validate_agent_hello(p)
        assert not valid
        assert "agent_id" in err

    def test_missing_capabilities(self):
        p = self._valid_payload()
        del p["capabilities"]
        valid, err = validate_agent_hello(p)
        assert not valid

    def test_capabilities_not_list(self):
        p = self._valid_payload()
        p["capabilities"] = "chat"
        valid, err = validate_agent_hello(p)
        assert not valid
        assert "array" in err

    def test_reconnect_not_bool(self):
        p = self._valid_payload()
        p["reconnect"] = "true"
        valid, err = validate_agent_hello(p)
        assert not valid
        assert "boolean" in err


class TestCheckCapability:
    def test_agent_namespace_needs_no_capability(self):
        for msg_type in (AGENT_HELLO, AGENT_CONFIGURED, AGENT_ERROR):
            allowed, err = check_capability(msg_type, set())
            assert allowed, f"{msg_type} should be allowed without capabilities"

    def test_chat_requires_chat_capability(self):
        allowed, _ = check_capability(CHAT_RESPONSE, {"activity", "tools"})
        assert not allowed

        allowed, _ = check_capability(CHAT_RESPONSE, {"chat"})
        assert allowed

    def test_activity_requires_activity_capability(self):
        allowed, _ = check_capability(ACTIVITY_DELTA, set())
        assert not allowed

        allowed, _ = check_capability(ACTIVITY_DELTA, {"activity"})
        assert allowed

    def test_tools_require_tools_capability(self):
        allowed, _ = check_capability(TOOL_USE, {"chat"})
        assert not allowed

        allowed, _ = check_capability(TOOL_USE, {"tools"})
        assert allowed

    def test_unknown_type_in_known_namespace_allowed(self):
        """Per §11.2, unknown types in known namespaces are silently ignored."""
        allowed, _ = check_capability("chat.typing", {"chat"})
        assert allowed

    def test_unknown_namespace_rejected(self):
        allowed, err = check_capability("foo.bar", set())
        assert not allowed
        assert "namespace" in err


class TestDirectionSets:
    def test_no_overlap_between_directions(self):
        assert AGENT_TO_CLIENT & CLIENT_TO_AGENT == frozenset()
        assert AGENT_TO_CLIENT & BIDIRECTIONAL == frozenset()
        assert CLIENT_TO_AGENT & BIDIRECTIONAL == frozenset()

    def test_all_types_is_union(self):
        assert ALL_TYPES == AGENT_TO_CLIENT | CLIENT_TO_AGENT | BIDIRECTIONAL

    def test_agent_hello_is_agent_to_client(self):
        assert AGENT_HELLO in AGENT_TO_CLIENT

    def test_chat_message_is_client_to_agent(self):
        assert CHAT_MESSAGE in CLIENT_TO_AGENT

    def test_agent_error_is_bidirectional(self):
        assert AGENT_ERROR in BIDIRECTIONAL
