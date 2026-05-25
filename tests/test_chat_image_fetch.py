"""handle_chat_image_fetch — lazy chat-image fetch over the relay.

The bridge persists chat messages with body-less `<build-image>` tags
to keep chat_messages.content small. The dashboard then asks for the
image bytes on demand via the chat_image.fetch action. These tests
exercise the happy path (chunked data-URI back) and the rejection
paths (non-image extension, oversized file, path escape, missing
file).
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from build_bridge.e2ee import E2EEHandler
from build_bridge.relay_handlers import files as files_handlers


class _FakeSession:
    session_id = "fake"
    session_key_b64 = "fake"


@pytest.fixture
def handler() -> E2EEHandler:
    h = E2EEHandler(SimpleNamespace(), MagicMock())
    h._sent_frames = []

    async def fake_send(session, ws, payload):
        h._sent_frames.append(payload)

    h._send_frame = fake_send  # type: ignore[assignment]
    h._agent_server = MagicMock()
    return h


def _set_channel_cwd(h: E2EEHandler, channel_id: str, cwd: Path) -> None:
    ch = SimpleNamespace(id=channel_id, working_directory=str(cwd), harness="")
    h._agent_server.store.get_channel = lambda cid: ch if cid == channel_id else None


# A minimal 1×1 PNG (red pixel), 67 bytes.
_TINY_PNG = bytes.fromhex(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
    "0000000D4944415478DA63F8FFFFFFFF0000070003FF00018E65B7560000000049"
    "454E44AE426082"
)


@pytest.mark.asyncio
async def test_image_returned_as_chunked_data_uri(handler: E2EEHandler, tmp_path: Path):
    _set_channel_cwd(handler, "ch1", tmp_path)
    (tmp_path / "shot.png").write_bytes(_TINY_PNG)

    await handler._handle_chat_image_fetch(_FakeSession(), {
        "channel_id": "ch1",
        "path": "shot.png",
    }, None)

    results = [f for f in handler._sent_frames if f["action"] == "chat_image_result"]
    assert results, "no chat_image_result frames sent"
    assert all(f.get("chunk_total") == len(results) for f in results)
    assembled = "".join(f["content"] for f in sorted(results, key=lambda f: f["chunk_index"]))
    head, _, body = assembled.partition(",")
    assert head == "data:image/png;base64"
    assert base64.b64decode(body) == _TINY_PNG


@pytest.mark.asyncio
async def test_non_image_extension_rejected(handler: E2EEHandler, tmp_path: Path):
    _set_channel_cwd(handler, "ch1", tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n")

    await handler._handle_chat_image_fetch(_FakeSession(), {
        "channel_id": "ch1",
        "path": "notes.txt",
    }, None)

    [frame] = [f for f in handler._sent_frames if f["action"] == "chat_image_result"]
    assert frame.get("error") == "Not an image"


@pytest.mark.asyncio
async def test_oversized_image_rejected(
    handler: E2EEHandler, tmp_path: Path, monkeypatch,
):
    _set_channel_cwd(handler, "ch1", tmp_path)
    (tmp_path / "big.png").write_bytes(b"\x00" * 1024)
    # Force the cap below the file size without writing megabytes.
    monkeypatch.setattr(files_handlers, "_MAX_IMAGE_SIZE", 128)

    await handler._handle_chat_image_fetch(_FakeSession(), {
        "channel_id": "ch1",
        "path": "big.png",
    }, None)

    [frame] = [f for f in handler._sent_frames if f["action"] == "chat_image_result"]
    assert frame.get("error") == "Image too large"


@pytest.mark.asyncio
async def test_path_escape_rejected(handler: E2EEHandler, tmp_path: Path):
    sibling = tmp_path.parent / "outside.png"
    sibling.write_bytes(_TINY_PNG)
    _set_channel_cwd(handler, "ch1", tmp_path)

    await handler._handle_chat_image_fetch(_FakeSession(), {
        "channel_id": "ch1",
        "path": "../outside.png",
    }, None)

    [frame] = [f for f in handler._sent_frames if f["action"] == "chat_image_result"]
    assert frame.get("error") == "Path outside working directory"


@pytest.mark.asyncio
async def test_missing_file_returns_error(handler: E2EEHandler, tmp_path: Path):
    _set_channel_cwd(handler, "ch1", tmp_path)

    await handler._handle_chat_image_fetch(_FakeSession(), {
        "channel_id": "ch1",
        "path": "ghost.png",
    }, None)

    [frame] = [f for f in handler._sent_frames if f["action"] == "chat_image_result"]
    assert "error" in frame and frame["error"]
