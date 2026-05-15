"""Uploads namespace — RELAY_PROTOCOL.md §5.11.

Chunked file upload pipeline. Chunks are deposited to
`~/.config/build/uploads/tmp/<file_id>/` and assembled on `upload_complete`
with a SHA-256 verify. Final placement is either the scratch uploads dir or
a destination directory inside the channel's workspace, depending on
whether the client supplied `dest_dir` on chunk 0.

The `_UPLOADS_BASE` and `_MAX_FILE_SIZE` class attributes live on the
`E2EEHandler` facade (test_upload_dest_dir.py monkeypatches them), so we
read them via the context's facade reference at call time.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from build_bridge import relay_protocol as proto
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


def upload_tmp_dir(facade: Any, file_id: str) -> Path:
    """`~/.config/build/uploads/tmp/<file_id>/`."""
    return facade._UPLOADS_BASE / "tmp" / file_id


def upload_final_dir(facade: Any, channel_id: str, file_id: str) -> Path:
    """`~/.config/build/uploads/<channel_id>/<file_id>/`."""
    return facade._UPLOADS_BASE / channel_id / file_id


def sanitize_filename(name: str) -> str:
    """Remove path separators and leading dots from a filename."""
    name = os.path.basename(name)
    name = name.lstrip(".")
    return name or "upload"


async def handle_upload_chunk(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Receive and store a single file chunk, then ack."""
    # validator: all of file_id, channel_id, chunk_index, total_size,
    # total_chunks, filename, data are required and type-checked.
    file_id = payload["file_id"]
    channel_id = payload["channel_id"]
    chunk_index = payload["chunk_index"]
    total_size = payload["total_size"]
    data_b64 = payload["data"]

    facade = ctx.facade
    if total_size > facade._MAX_FILE_SIZE:
        await ctx.send_frame(session, ws, payload={
            "action": proto.UPLOAD_ERROR,
            "file_id": file_id,
            "error": f"file too large (max {facade._MAX_FILE_SIZE // (1024 * 1024)} MB)",
        })
        return

    try:
        chunk_data = base64.b64decode(data_b64 + "==")  # pad for standard base64
    except Exception:
        # Try libsodium-style no-padding base64.
        try:
            padded = data_b64 + "=" * (-len(data_b64) % 4)
            chunk_data = base64.b64decode(padded)
        except Exception as exc:
            await ctx.send_frame(session, ws, payload={
                "action": proto.UPLOAD_ERROR,
                "file_id": file_id,
                "error": f"invalid chunk data: {exc}",
            })
            return

    tmp_dir = upload_tmp_dir(facade, file_id)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = tmp_dir / f"chunk_{chunk_index}"
    chunk_path.write_bytes(chunk_data)

    # Store metadata on first chunk.
    if chunk_index == 0:
        meta = {
            "filename": sanitize_filename(payload.get("filename", "upload")),
            "mime_type": payload.get("mime_type", "application/octet-stream"),
            "total_size": total_size,
            "total_chunks": payload.get("total_chunks", 1),
            "channel_id": channel_id,
            # Optional destination directory relative to the channel's
            # working_directory. When set, the assembled file lands in the
            # workspace instead of the scratch uploads registry. Sanitised
            # + path-safety checked at upload_complete time.
            "dest_dir": payload.get("dest_dir", "") or "",
        }
        (tmp_dir / "meta.json").write_text(json.dumps(meta))

    log.info(
        "Received chunk %d for upload %s (%d bytes)",
        chunk_index, file_id[:8], len(chunk_data),
    )

    await ctx.send_frame(session, ws, payload={
        "action": proto.CHUNK_ACK,
        "file_id": file_id,
        "chunk_index": chunk_index,
    })


async def handle_upload_complete(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Assemble chunks, verify hash, and move to final location."""
    file_id = payload.get("file_id", "")
    channel_id = payload.get("channel_id", "")
    expected_sha256 = payload.get("sha256", "")

    facade = ctx.facade
    tmp_dir = upload_tmp_dir(facade, file_id)
    meta_path = tmp_dir / "meta.json"

    if not meta_path.exists():
        await ctx.send_frame(session, ws, payload={
            "action": proto.UPLOAD_ERROR,
            "file_id": file_id,
            "error": "no chunks received for this upload",
        })
        return

    meta = json.loads(meta_path.read_text())
    total_chunks = meta["total_chunks"]
    filename = meta["filename"]

    # Assemble chunks.
    assembled = bytearray()
    for i in range(total_chunks):
        chunk_path = tmp_dir / f"chunk_{i}"
        if not chunk_path.exists():
            await ctx.send_frame(session, ws, payload={
                "action": proto.UPLOAD_ERROR,
                "file_id": file_id,
                "error": f"missing chunk {i}",
            })
            return
        assembled.extend(chunk_path.read_bytes())

    # Verify SHA-256.
    actual_sha256 = hashlib.sha256(assembled).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        log.error(
            "Upload %s hash mismatch: expected %s, got %s",
            file_id[:8], expected_sha256[:12], actual_sha256[:12],
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        await ctx.send_frame(session, ws, payload={
            "action": proto.UPLOAD_ERROR,
            "file_id": file_id,
            "error": "SHA-256 mismatch — file corrupted in transit",
        })
        return

    # Choose destination: if the client specified a dest_dir, land the file
    # inside the channel's working directory. Otherwise keep the legacy
    # scratch-uploads path.
    dest_dir = (meta.get("dest_dir") or "").strip()
    final_path: Path
    if dest_dir:
        cwd = ctx.get_channel_cwd(channel_id)
        resolved = ctx.resolve_safe_path(cwd, os.path.join(dest_dir, filename))
        if resolved is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            await ctx.send_frame(session, ws, payload={
                "action": proto.UPLOAD_ERROR,
                "file_id": file_id,
                "error": "destination outside workspace",
            })
            return
        final_path = Path(resolved)
        if final_path.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            await ctx.send_frame(session, ws, payload={
                "action": proto.UPLOAD_ERROR,
                "file_id": file_id,
                "error": "file exists",
                "path": str(final_path),
            })
            return
        final_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        final_dir = upload_final_dir(facade, channel_id, file_id)
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / filename

    final_path.write_bytes(assembled)

    # Clean up temp.
    shutil.rmtree(tmp_dir, ignore_errors=True)

    log.info(
        "Upload complete: %s → %s (%d bytes, sha256=%s)",
        file_id[:8], final_path, len(assembled), actual_sha256[:12],
    )

    await ctx.send_frame(session, ws, payload={
        "action": proto.UPLOAD_ACCEPTED,
        "file_id": file_id,
        "filename": filename,
        "size": len(assembled),
        "path": str(final_path),
    })
