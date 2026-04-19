"""Tests for build_bridge.harness_registry — harness detection and serialization."""

from __future__ import annotations

from unittest.mock import patch

from build_bridge.harness_registry import (
    KNOWN_HARNESSES,
    HarnessInfo,
    ModelInfo,
    detect_installed,
    get_harness,
    serialize_harnesses,
)


class TestKnownHarnesses:
    def test_has_claude_code(self):
        ids = [h.id for h in KNOWN_HARNESSES]
        assert "claude-code" in ids

    def test_has_codex(self):
        ids = [h.id for h in KNOWN_HARNESSES]
        assert "codex" in ids

    def test_claude_code_has_models(self):
        cc = next(h for h in KNOWN_HARNESSES if h.id == "claude-code")
        assert len(cc.models) >= 2
        model_ids = [m.id for m in cc.models]
        assert any("sonnet" in m for m in model_ids)

    def test_codex_has_models(self):
        cx = next(h for h in KNOWN_HARNESSES if h.id == "codex")
        assert len(cx.models) >= 2

    def test_default_models_are_valid(self):
        for harness in KNOWN_HARNESSES:
            model_ids = [m.id for m in harness.models]
            assert harness.default_model in model_ids, (
                f"{harness.id} default_model {harness.default_model!r} not in {model_ids}"
            )


class TestDetectInstalled:
    def test_detects_installed_binary(self):
        with patch("build_bridge.harness_registry.shutil.which") as mock_which:
            mock_which.side_effect = lambda name: "/usr/bin/claude" if name == "claude" else None
            harnesses = detect_installed()
            cc = next(h for h in harnesses if h.id == "claude-code")
            cx = next(h for h in harnesses if h.id == "codex")
            assert cc.installed is True
            assert cx.installed is False

    def test_returns_all_known_harnesses(self):
        with patch("build_bridge.harness_registry.shutil.which", return_value=None):
            harnesses = detect_installed()
            assert len(harnesses) == len(KNOWN_HARNESSES)

    def test_none_installed_when_no_binaries(self):
        with patch("build_bridge.harness_registry.shutil.which", return_value=None):
            harnesses = detect_installed()
            assert all(not h.installed for h in harnesses)

    def test_all_installed_when_all_binaries(self):
        with patch("build_bridge.harness_registry.shutil.which", return_value="/usr/bin/something"):
            harnesses = detect_installed()
            assert all(h.installed for h in harnesses)


class TestGetHarness:
    def test_get_existing(self):
        h = get_harness("claude-code")
        assert h is not None
        assert h.id == "claude-code"

    def test_get_nonexistent(self):
        h = get_harness("nonexistent")
        assert h is None


class TestSerializeHarnesses:
    def test_serializes_structure(self):
        harnesses = [
            HarnessInfo(
                id="test",
                name="Test Harness",
                description="A test harness",
                binary="test-bin",
                models=[ModelInfo("m1", "Model 1", "test-provider")],
                default_model="m1",
                installed=True,
            ),
        ]
        result = serialize_harnesses(harnesses)
        assert len(result) == 1
        assert result[0]["id"] == "test"
        assert result[0]["name"] == "Test Harness"
        assert result[0]["installed"] is True
        assert len(result[0]["models"]) == 1
        assert result[0]["models"][0]["id"] == "m1"
        assert result[0]["models"][0]["provider"] == "test-provider"
        assert result[0]["default_model"] == "m1"

    def test_serializes_all_known(self):
        result = serialize_harnesses(KNOWN_HARNESSES)
        assert len(result) == len(KNOWN_HARNESSES)
        for item in result:
            assert "id" in item
            assert "models" in item
            assert isinstance(item["models"], list)
