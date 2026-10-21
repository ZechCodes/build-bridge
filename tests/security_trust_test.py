"""Regression checks for the October 2026 trust-store remediation."""
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lock = tomllib.loads((ROOT / "uv.lock").read_text())
versions = [p["version"] for p in lock["package"] if p["name"] == "certifi"]
assert versions and all(tuple(map(int, v.split("."))) >= (2026, 10, 19) for v in versions), versions

manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
assert manifest.get("tool", {}).get("uv", {}).get("required-version") == ">=0.13.4", "Require patched uv before dependency installation"
