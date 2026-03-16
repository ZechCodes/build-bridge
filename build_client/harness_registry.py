"""Harness registry — detects available coding harnesses and their models.

Provides the browser with a list of installable harnesses so users can
configure channels with the right harness + model combination.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    """A model available for a harness."""

    id: str
    name: str
    provider: str  # "anthropic", "openai", etc.


@dataclass
class HarnessInfo:
    """A coding harness available on this device."""

    id: str
    name: str
    description: str
    models: list[ModelInfo]
    default_model: str
    installed: bool = False


# ---------------------------------------------------------------------------
# Known harnesses and their models
# ---------------------------------------------------------------------------

_CLAUDE_CODE_MODELS = [
    ModelInfo("claude-sonnet-4-20250514", "Claude Sonnet 4", "anthropic"),
    ModelInfo("claude-opus-4-20250514", "Claude Opus 4", "anthropic"),
    ModelInfo("claude-haiku-3-5-20241022", "Claude Haiku 3.5", "anthropic"),
]

_CODEX_MODELS = [
    ModelInfo("o4-mini", "o4-mini", "openai"),
    ModelInfo("o3", "o3", "openai"),
    ModelInfo("codex-mini-latest", "Codex Mini", "openai"),
]

KNOWN_HARNESSES: list[HarnessInfo] = [
    HarnessInfo(
        id="claude-code",
        name="Claude Code",
        description="Anthropic's agentic coding tool with planning, tool use, and MCP support.",
        models=_CLAUDE_CODE_MODELS,
        default_model="claude-sonnet-4-20250514",
    ),
    HarnessInfo(
        id="codex",
        name="Codex CLI",
        description="OpenAI's open-source coding agent.",
        models=_CODEX_MODELS,
        default_model="codex-mini-latest",
    ),
]

# Maps harness id → binary name to check on PATH.
_HARNESS_BINARIES: dict[str, str] = {
    "claude-code": "claude",
    "codex": "codex",
}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_installed() -> list[HarnessInfo]:
    """Detect which harnesses are installed on this device.

    Returns all known harnesses with their `installed` flag set based on
    whether the corresponding binary is found on PATH.
    """
    result = []
    for harness in KNOWN_HARNESSES:
        binary = _HARNESS_BINARIES.get(harness.id)
        installed = shutil.which(binary) is not None if binary else False
        result.append(HarnessInfo(
            id=harness.id,
            name=harness.name,
            description=harness.description,
            models=harness.models,
            default_model=harness.default_model,
            installed=installed,
        ))
    return result


def get_harness(harness_id: str) -> HarnessInfo | None:
    """Get a harness by ID."""
    for harness in KNOWN_HARNESSES:
        if harness.id == harness_id:
            return harness
    return None


def serialize_harnesses(harnesses: list[HarnessInfo]) -> list[dict]:
    """Serialize harness list for E2EE transport to browser."""
    return [
        {
            "id": h.id,
            "name": h.name,
            "description": h.description,
            "models": [
                {"id": m.id, "name": m.name, "provider": m.provider}
                for m in h.models
            ],
            "default_model": h.default_model,
            "installed": h.installed,
        }
        for h in harnesses
    ]
