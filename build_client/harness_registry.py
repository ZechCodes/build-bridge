"""Harness registry — detects available coding harnesses and their models.

Loads harness definitions from JSON files in the harnesses/ directory,
detects which are installed, and provides lookup/serialization for the
browser UI.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_HARNESSES_DIR = Path(__file__).parent / "harnesses"


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
    binary: str
    models: list[ModelInfo]
    default_model: str
    installed: bool = False


def _load_harnesses() -> list[HarnessInfo]:
    """Load harness definitions from JSON files in the harnesses/ directory."""
    harnesses: list[HarnessInfo] = []
    if not _HARNESSES_DIR.is_dir():
        log.warning("Harnesses directory not found: %s", _HARNESSES_DIR)
        return harnesses

    for path in sorted(_HARNESSES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            models = [
                ModelInfo(id=m["id"], name=m["name"], provider=m["provider"])
                for m in data.get("models", [])
            ]
            harnesses.append(HarnessInfo(
                id=data["id"],
                name=data["name"],
                description=data.get("description", ""),
                binary=data.get("binary", ""),
                models=models,
                default_model=data.get("default_model", models[0].id if models else ""),
            ))
        except Exception as exc:
            log.warning("Failed to load harness from %s: %s", path, exc)

    return harnesses


KNOWN_HARNESSES: list[HarnessInfo] = _load_harnesses()


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
        installed = shutil.which(harness.binary) is not None if harness.binary else False
        result.append(HarnessInfo(
            id=harness.id,
            name=harness.name,
            description=harness.description,
            binary=harness.binary,
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
