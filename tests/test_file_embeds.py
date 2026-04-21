"""Unit tests for AgentServer._resolve_file_embeds — the [[file]] /
[[diff]] marker expansion. Exercises the image branch that returns
`<build-image>` with base64 content."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from build_bridge.agent_server import AgentServer
from build_bridge.agent_store import AgentStore


@pytest.fixture
def store(tmp_path: Path) -> AgentStore:
    s = AgentStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def server(store: AgentStore) -> AgentServer:
    async def _noop_broadcast(channel_id: str, payload: dict) -> None:
        return None
    return AgentServer(store=store, broadcast=_noop_broadcast, port=0)


@pytest.fixture
def channel(store: AgentStore, tmp_path: Path) -> str:
    channel_id = "chan-embed-test"
    store.create_channel(
        channel_id=channel_id,
        agent_id="agt-x",
        harness="claude-code",
        model="m1",
        working_directory=str(tmp_path),
    )
    return channel_id


# A minimal valid 1×1 PNG (red pixel) — 67 bytes.
_TINY_PNG = bytes.fromhex(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
    "0000000D4944415478DA63F8FFFFFFFF0000070003FF00018E65B7560000000049"
    "454E44AE426082"
)


async def test_text_file_yields_build_file(server, channel, tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("hi there\n")
    abs_path = str(p)
    out = await server._resolve_file_embeds(channel, f"See [[file]]({abs_path}) please")
    assert "<build-file" in out
    assert f'path="{abs_path}"' in out
    assert 'lang="txt"' in out
    assert "hi there" in out


async def test_image_yields_build_image_with_base64(server, channel, tmp_path):
    p = tmp_path / "pixel.png"
    p.write_bytes(_TINY_PNG)
    abs_path = str(p)
    out = await server._resolve_file_embeds(channel, f"look: [[file]]({abs_path})")
    assert "<build-image" in out
    assert f'path="{abs_path}"' in out
    assert 'mime="image/png"' in out
    # Extract the base64 body and confirm it decodes back to the bytes.
    head = f'<build-image path="{abs_path}" mime="image/png">\n'
    tail = "\n</build-image>"
    start = out.index(head) + len(head)
    end = out.index(tail, start)
    decoded = base64.b64decode(out[start:end])
    assert decoded == _TINY_PNG


async def test_image_extensions_detected(server, channel, tmp_path):
    # jpg, gif, webp, svg all map through _IMAGE_MIME and produce
    # <build-image>. Contents can be empty for this test — we only
    # care that the right tag fires.
    for ext, mime in [
        ("jpg", "image/jpeg"),
        ("jpeg", "image/jpeg"),
        ("gif", "image/gif"),
        ("webp", "image/webp"),
        ("svg", "image/svg+xml"),
    ]:
        p = tmp_path / f"file.{ext}"
        p.write_bytes(b"binary bytes")
        out = await server._resolve_file_embeds(channel, f"[[file]]({p})")
        assert f'mime="{mime}"' in out, f"ext={ext} produced {out!r}"


async def test_oversize_image_falls_back(server, channel, tmp_path, monkeypatch):
    p = tmp_path / "big.png"
    p.write_bytes(b"\x00" * 64)
    # Shrink the cap to force the fallback without writing 2 MiB to disk.
    monkeypatch.setattr(AgentServer, "_MAX_IMAGE_EMBED_BYTES", 16)
    out = await server._resolve_file_embeds(channel, f"[[file]]({p})")
    assert out == f"[Image too large: {p}]"


async def test_missing_file_falls_back(server, channel, tmp_path):
    missing = tmp_path / "nope.png"
    out = await server._resolve_file_embeds(channel, f"[[file]]({missing})")
    assert out == f"[Could not read: {missing}]"


async def test_relative_file_path_rejected(server, channel, tmp_path):
    # Even if the file exists under the channel cwd, a relative path
    # must be rejected with a clear [Path must be absolute: …] error.
    (tmp_path / "rel.txt").write_text("never resolved\n")
    out = await server._resolve_file_embeds(channel, "[[file]](rel.txt)")
    assert out == "[Path must be absolute: rel.txt]"


async def test_relative_diff_path_rejected(server, channel, tmp_path):
    (tmp_path / "rel.txt").write_text("hi\n")
    out = await server._resolve_file_embeds(channel, "[[diff]](rel.txt)")
    assert out == "[Path must be absolute: rel.txt]"


async def test_relative_diff_pair_rejected(server, channel, tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    out = await server._resolve_file_embeds(channel, "[[diff]](a.txt|b.txt)")
    assert out == "[Path must be absolute: a.txt|b.txt]"
