"""Files namespace — RELAY_PROTOCOL.md §5.9.

Read-side filesystem and git operations for the browser file viewer:

- `files_list`: directory listing with two-phase git enrichment.
- `files_changes`: changed files across all repos in the working dir, or
  scoped to one repo with optional ref range.
- `files_commits`: recent commits for one repo.
- `file_read`: file contents with binary / image / text branching.
- `file_diff`: per-file unified diff with optional ref range.

All paths are resolved relative to the channel's `working_directory` and
rejected if they escape via `..` or symlinks (see
`HandlerContext.resolve_safe_path`).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

from build_bridge import relay_protocol as proto
from build_bridge.complications import (
    _run_git,
    find_git_repo,
    git_branch_info,
    git_remote_name,
)
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


_MAX_FILE_READ = 100_000  # ~100KB text, stays under 256KB encrypted envelope
_BINARY_CHECK_SIZE = 8192
_MAX_DIFF_SIZE = 100_000
_MAX_DIR_ENTRIES = 1000
_MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".bmp"})


# ---------------------------------------------------------------------------
# Git output parsers
# ---------------------------------------------------------------------------


def _parse_porcelain_status(output: str) -> dict[str, tuple[str, str]]:
    """Parse git status --porcelain=v1 → {rel_path: (staged, unstaged)}."""
    result: dict[str, tuple[str, str]] = {}
    for line in output.splitlines():
        if len(line) < 4:
            continue
        x, y = line[0], line[1]
        path = line[3:]
        # Handle renames: "R  old -> new"
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        result[path] = (x, y)
    return result


def _parse_numstat_per_file(output: str) -> dict[str, tuple[int, int]]:
    """Parse git diff --numstat → {path: (insertions, deletions)}."""
    result: dict[str, tuple[int, int]] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ins_s, dels_s, path = parts[0], parts[1], parts[2]
        if ins_s == "-":  # binary
            result[path] = (0, 0)
            continue
        try:
            result[path] = (int(ins_s), int(dels_s))
        except ValueError:
            continue
    return result


def _parse_name_status(output: str) -> dict[str, str]:
    """Parse git diff --name-status → {path: status}."""
    result: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        path = parts[-1]
        result[path] = status[0] if status else "M"
    return result


# ---------------------------------------------------------------------------
# Repo path helpers
# ---------------------------------------------------------------------------


def _repo_display_path(repo: str, cwd: str) -> str:
    """Compute a display path for `repo` relative to `cwd` ("." if equal)."""
    try:
        display_path = str(Path(repo).resolve().relative_to(Path(cwd).resolve()))
    except ValueError:
        display_path = str(Path(repo).resolve())
    return "." if display_path == "." else display_path


def _resolve_repo_path(
    ctx: HandlerContext, cwd: str, repo_path: str | None,
) -> str | None:
    """Resolve a repo path under cwd and confirm it is a git repo."""
    if repo_path:
        resolved = ctx.resolve_safe_path(cwd, repo_path)
        if resolved is None:
            return None
        repo = find_git_repo(resolved)
        if repo and Path(repo).resolve() == Path(resolved).resolve():
            return repo
        if (Path(resolved) / ".git").is_dir():
            return str(Path(resolved).resolve())
        return None
    return find_git_repo(cwd)


async def _validate_commit_ref(repo: str, ref: str) -> str | None:
    if ref == "HEAD":
        return ref
    out, ok = await _run_git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    return ref if ok and out else None


async def _repo_metadata(
    repo: str, cwd: str, entries: list[dict[str, Any]],
) -> dict[str, Any]:
    remote, branch_info = await asyncio.gather(
        git_remote_name(repo),
        git_branch_info(repo),
    )
    branch_name, _, _ = branch_info
    return {
        "path": _repo_display_path(repo, cwd),
        "remote": remote,
        "branch": branch_name,
        "entries": entries,
    }


def _change_entry(
    repo: str, cwd: str, repo_rel: str, status: str, ins: int, dels: int,
) -> dict[str, Any]:
    repo_root = Path(repo).resolve()
    cwd_resolved = Path(cwd).resolve()
    abs_path = repo_root / repo_rel
    try:
        cwd_rel = str(abs_path.relative_to(cwd_resolved))
    except ValueError:
        cwd_rel = repo_rel
    try:
        mtime = abs_path.stat().st_mtime
    except OSError:
        mtime = 0
    return {
        "name": os.path.basename(repo_rel),
        "path": cwd_rel,
        "type": "file",
        "size": 0,
        "modified": mtime,
        "git_status": status,
        "staged_status": None,
        "insertions": ins,
        "deletions": dels,
        "is_gitignored": False,
    }


def _find_all_git_repos(cwd: str) -> list[str]:
    """Find all git repos at cwd itself and 1-2 levels deep."""
    repos: list[str] = []
    cwd_path = Path(cwd).resolve()

    if (cwd_path / ".git").is_dir():
        repos.append(str(cwd_path))

    try:
        for entry in os.scandir(cwd_path):
            if not entry.is_dir(follow_symlinks=False) or entry.name.startswith("."):
                continue
            sub = Path(entry.path)
            if (sub / ".git").is_dir():
                repos.append(str(sub))
                continue
            # Level 2.
            try:
                for entry2 in os.scandir(sub):
                    if not entry2.is_dir(follow_symlinks=False) or entry2.name.startswith("."):
                        continue
                    if (Path(entry2.path) / ".git").is_dir():
                        repos.append(str(Path(entry2.path)))
            except (OSError, PermissionError):
                pass
    except (OSError, PermissionError):
        pass

    return repos


# ---------------------------------------------------------------------------
# files_list
# ---------------------------------------------------------------------------


async def handle_files_list(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """List directory contents. Bare listing first, then a git-enriched follow-up."""
    channel_id = payload["channel_id"]  # validator: required
    rel_path = payload.get("path", "")
    cwd = ctx.get_channel_cwd(channel_id)
    resolved = ctx.resolve_safe_path(cwd, rel_path) if rel_path else cwd
    if resolved is None:
        await ctx.send_frame(session, ws, payload={
            "action": proto.ERROR,
            "channel_id": channel_id,
            "error": "Path outside working directory",
        })
        return

    if not os.path.isdir(resolved):
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILES_LIST_RESULT,
            "channel_id": channel_id,
            "path": rel_path,
            "error": "Not a directory",
            "entries": [],
        })
        return

    # Scan directory entries.
    try:
        raw_entries: list[dict[str, Any]] = []
        with os.scandir(resolved) as it:
            for entry in it:
                try:
                    is_dir = entry.is_dir(follow_symlinks=True)
                    is_symlink = entry.is_symlink()
                    # Skip symlinks that escape the working directory.
                    if is_symlink:
                        link_target = str(Path(entry.path).resolve())
                        base_resolved = str(Path(cwd).resolve())
                        if (link_target != base_resolved and
                                not link_target.startswith(base_resolved + os.sep)):
                            continue
                    stat = entry.stat(follow_symlinks=True)
                    entry_rel = os.path.join(rel_path, entry.name) if rel_path else entry.name
                    raw_entries.append({
                        "name": entry.name,
                        "path": entry_rel,
                        "type": "dir" if is_dir else "file",
                        "size": stat.st_size if not is_dir else 0,
                        "modified": stat.st_mtime,
                        "git_status": None,
                        "staged_status": None,
                        "insertions": 0,
                        "deletions": 0,
                        "is_gitignored": False,
                    })
                except OSError:
                    continue
    except OSError as exc:
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILES_LIST_RESULT,
            "channel_id": channel_id,
            "path": rel_path,
            "error": str(exc),
            "entries": [],
        })
        return

    # Sort: directories first, then alphabetical.
    raw_entries.sort(key=lambda e: (0 if e["type"] == "dir" else 1, e["name"].lower()))

    truncated = len(raw_entries) > _MAX_DIR_ENTRIES
    if truncated:
        raw_entries = raw_entries[:_MAX_DIR_ENTRIES]

    # Fast path: send the bare directory listing immediately so the client
    # can render the tree. Git enrichment (status/numstat/check-ignore —
    # 4 subprocesses, up to several seconds on a large monorepo) runs in
    # the background and pushes a second `files_list_result` frame for
    # the same path when ready. The client overwrites the first frame
    # keyed by path.
    await ctx.send_frame(session, ws, payload={
        "action": proto.FILES_LIST_RESULT,
        "channel_id": channel_id,
        "path": rel_path,
        "entries": raw_entries,
        "truncated": truncated,
    })

    repo = find_git_repo(resolved)
    if repo:
        asyncio.create_task(_enrich_files_list_with_git(
            ctx, session, ws, channel_id, rel_path, resolved, repo, raw_entries, truncated,
        ))


async def _enrich_files_list_with_git(
    ctx: HandlerContext,
    session: ActiveSession,
    ws: Any,
    channel_id: str,
    rel_path: str,
    resolved: str,
    repo: str,
    raw_entries: list[dict[str, Any]],
    truncated: bool,
) -> None:
    """Run git status / numstat / check-ignore in parallel and send an
    enriched `files_list_result` frame on top of the bare listing."""
    try:
        repo_root = Path(repo).resolve()
        try:
            dir_rel_to_repo = str(Path(resolved).resolve().relative_to(repo_root))
            if dir_rel_to_repo == ".":
                dir_rel_to_repo = ""
        except ValueError:
            dir_rel_to_repo = ""

        entry_abs_paths = [os.path.join(resolved, e["name"]) for e in raw_entries]
        stdin_data = "\n".join(entry_abs_paths).encode() if entry_abs_paths else b""

        status_task         = _run_git(repo, ["status", "--porcelain=v1", "--", resolved])
        numstat_task        = _run_git(repo, ["diff", "--numstat", "--", resolved])
        numstat_cached_task = _run_git(repo, ["diff", "--cached", "--numstat", "--", resolved])

        async def _noop_git() -> tuple[str, bool]:
            return ("", False)

        ignore_task = (
            _run_git(repo, ["check-ignore", "--stdin"], stdin_data=stdin_data)
            if stdin_data else _noop_git()
        )

        results = await asyncio.gather(
            status_task, numstat_task, numstat_cached_task, ignore_task,
            return_exceptions=True,
        )
        status_out         = results[0] if not isinstance(results[0], BaseException) else ("", False)
        numstat_out        = results[1] if not isinstance(results[1], BaseException) else ("", False)
        numstat_cached_out = results[2] if not isinstance(results[2], BaseException) else ("", False)
        ignore_out         = results[3] if not isinstance(results[3], BaseException) else ("", False)

        status_map         = _parse_porcelain_status(status_out[0])
        numstat_map        = _parse_numstat_per_file(numstat_out[0])
        numstat_cached_map = _parse_numstat_per_file(numstat_cached_out[0])
        ignored_set = (
            {line.strip() for line in ignore_out[0].splitlines() if line.strip()}
            if ignore_out[0] else set()
        )

        for entry in raw_entries:
            entry_repo_rel = (
                os.path.join(dir_rel_to_repo, entry["name"])
                if dir_rel_to_repo else entry["name"]
            )

            if entry_repo_rel in status_map:
                staged, unstaged = status_map[entry_repo_rel]
                entry["staged_status"] = staged if staged.strip() else None
                entry["git_status"]    = unstaged if unstaged.strip() else None
            elif entry["type"] == "dir":
                prefix = entry_repo_rel + "/"
                has_staged = has_unstaged = False
                for p, (s, u) in status_map.items():
                    if p.startswith(prefix):
                        if s.strip():
                            has_staged = True
                        if u.strip():
                            has_unstaged = True
                        if has_staged and has_unstaged:
                            break
                if has_staged:
                    entry["staged_status"] = "M"
                if has_unstaged:
                    entry["git_status"] = "M"

            ins, dels = 0, 0
            if entry_repo_rel in numstat_map:
                i, d = numstat_map[entry_repo_rel]
                ins += i
                dels += d
            if entry_repo_rel in numstat_cached_map:
                i, d = numstat_cached_map[entry_repo_rel]
                ins += i
                dels += d
            if entry["type"] == "dir":
                prefix = entry_repo_rel + "/"
                for p, (i, d) in numstat_map.items():
                    if p.startswith(prefix):
                        ins += i
                        dels += d
                for p, (i, d) in numstat_cached_map.items():
                    if p.startswith(prefix):
                        ins += i
                        dels += d
            entry["insertions"] = ins
            entry["deletions"]  = dels

            entry_abs = os.path.join(resolved, entry["name"])
            entry["is_gitignored"] = entry_abs in ignored_set

        await ctx.send_frame(session, ws, payload={
            "action": proto.FILES_LIST_RESULT,
            "channel_id": channel_id,
            "path": rel_path,
            "entries": raw_entries,
            "truncated": truncated,
        })
    except Exception:
        log.debug("files_list git enrichment failed", exc_info=True)


# ---------------------------------------------------------------------------
# files_changes
# ---------------------------------------------------------------------------


async def _collect_repo_changes(repo: str, cwd: str) -> dict[str, Any] | None:
    """Collect changed files for a single repo (worktree + staged)."""
    results = await asyncio.gather(
        _run_git(repo, ["status", "--porcelain=v1"]),
        _run_git(repo, ["diff", "--numstat"]),
        _run_git(repo, ["diff", "--cached", "--numstat"]),
        git_remote_name(repo),
        git_branch_info(repo),
        return_exceptions=True,
    )

    status_out         = results[0] if not isinstance(results[0], BaseException) else ("", False)
    numstat_out        = results[1] if not isinstance(results[1], BaseException) else ("", False)
    numstat_cached_out = results[2] if not isinstance(results[2], BaseException) else ("", False)
    remote      = results[3] if not isinstance(results[3], BaseException) else None
    branch_info = results[4] if not isinstance(results[4], BaseException) else ("unknown", False, None)

    branch_name, _, _ = branch_info

    status_map         = _parse_porcelain_status(status_out[0])
    numstat_map        = _parse_numstat_per_file(numstat_out[0])
    numstat_cached_map = _parse_numstat_per_file(numstat_cached_out[0])

    all_paths = set(status_map.keys()) | set(numstat_map.keys()) | set(numstat_cached_map.keys())
    if not all_paths:
        return None

    entries: list[dict[str, Any]] = []
    for repo_rel in sorted(all_paths):
        staged, unstaged = status_map.get(repo_rel, (" ", " "))
        ins, dels = 0, 0
        if repo_rel in numstat_map:
            i, d = numstat_map[repo_rel]
            ins += i
            dels += d
        if repo_rel in numstat_cached_map:
            i, d = numstat_cached_map[repo_rel]
            ins += i
            dels += d

        entry = _change_entry(
            repo, cwd, repo_rel,
            unstaged if unstaged.strip() else (staged if staged.strip() else "M"),
            ins, dels,
        )
        entry["staged_status"] = staged if staged.strip() else None
        entry["git_status"]    = unstaged if unstaged.strip() else None
        entries.append(entry)

    return {
        "path": _repo_display_path(repo, cwd),
        "remote": remote,
        "branch": branch_name,
        "entries": entries,
    }


async def _collect_repo_revision_changes(
    repo: str, cwd: str, newer_ref: str, older_ref: str,
) -> dict[str, Any]:
    """Collect changed files for one repo between two revisions."""
    validated_older = await _validate_commit_ref(repo, older_ref)
    if not validated_older:
        meta = await _repo_metadata(repo, cwd, [])
        return {**meta, "newer_ref": newer_ref, "older_ref": older_ref, "error": "Invalid older ref"}

    if newer_ref == "worktree":
        diff_args = [validated_older]
    else:
        newer_valid = await _validate_commit_ref(repo, newer_ref)
        if not newer_valid:
            meta = await _repo_metadata(repo, cwd, [])
            return {**meta, "newer_ref": newer_ref, "older_ref": older_ref, "error": "Invalid newer ref"}
        diff_args = [validated_older, newer_valid]

    name_task = _run_git(repo, ["diff", "--name-status", *diff_args])
    numstat_task = _run_git(repo, ["diff", "--numstat", *diff_args])
    results = await asyncio.gather(name_task, numstat_task, return_exceptions=True)
    name_out    = results[0] if not isinstance(results[0], BaseException) else ("", False)
    numstat_out = results[1] if not isinstance(results[1], BaseException) else ("", False)

    status_map = _parse_name_status(name_out[0])
    numstat_map = _parse_numstat_per_file(numstat_out[0])

    if newer_ref == "worktree":
        untracked_out, _ = await _run_git(repo, ["ls-files", "--others", "--exclude-standard"])
        for path in untracked_out.splitlines():
            if path.strip():
                status_map[path] = "?"
                numstat_map.setdefault(path, (0, 0))

    entries = []
    for repo_rel in sorted(set(status_map) | set(numstat_map)):
        ins, dels = numstat_map.get(repo_rel, (0, 0))
        entries.append(_change_entry(
            repo, cwd, repo_rel, status_map.get(repo_rel, "M"), ins, dels,
        ))

    meta = await _repo_metadata(repo, cwd, entries)
    return {**meta, "newer_ref": newer_ref, "older_ref": older_ref}


async def handle_files_changes(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Return all changed files across all repos in the working directory."""
    channel_id = payload["channel_id"]  # validator: required
    cwd = ctx.get_channel_cwd(channel_id)
    repo_path = payload.get("repo_path")
    newer_ref = payload.get("newer_ref")
    older_ref = payload.get("older_ref")

    if repo_path is not None or newer_ref is not None or older_ref is not None:
        repo = _resolve_repo_path(ctx, cwd, repo_path)
        if not repo:
            await ctx.send_frame(session, ws, payload={
                "action": proto.FILES_CHANGES_RESULT,
                "channel_id": channel_id,
                "repos": [{
                    "path": repo_path or "",
                    "entries": [],
                    "newer_ref": newer_ref,
                    "older_ref": older_ref,
                    "error": "Repository not found",
                }],
            })
            return
        repo_result = await _collect_repo_revision_changes(
            repo, cwd, newer_ref or "worktree", older_ref or "HEAD",
        )
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILES_CHANGES_RESULT,
            "channel_id": channel_id,
            "repos": [repo_result],
            "repo_path": repo_result.get("path"),
            "newer_ref": repo_result.get("newer_ref"),
            "older_ref": repo_result.get("older_ref"),
        })
        return

    repo_paths = _find_all_git_repos(cwd)

    if not repo_paths:
        # Fallback: walk UP from cwd looking for a repo.
        repo = find_git_repo(cwd)
        if repo:
            repo_paths = [repo]

    if not repo_paths:
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILES_CHANGES_RESULT,
            "channel_id": channel_id,
            "repos": [],
        })
        return

    # Collect changes from all repos in parallel.
    repo_results = await asyncio.gather(
        *(_collect_repo_changes(r, cwd) for r in repo_paths),
        return_exceptions=True,
    )

    repos = [r for r in repo_results if isinstance(r, dict)]
    await ctx.send_frame(session, ws, payload={
        "action": proto.FILES_CHANGES_RESULT,
        "channel_id": channel_id,
        "repos": repos,
    })


# ---------------------------------------------------------------------------
# files_commits
# ---------------------------------------------------------------------------


async def handle_files_commits(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Return recent commits for one repo."""
    channel_id = payload["channel_id"]  # validator: required
    repo_path = payload.get("repo_path")
    limit = payload.get("limit", 25)
    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit = 25

    cwd = ctx.get_channel_cwd(channel_id)
    repo = _resolve_repo_path(ctx, cwd, repo_path)
    if not repo:
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILES_COMMITS_RESULT,
            "channel_id": channel_id,
            "repo_path": repo_path or "",
            "commits": [],
            "error": "Repository not found",
        })
        return

    fmt = "%H%x1f%h%x1f%ct%x1f%s"
    out, ok = await _run_git(repo, ["log", f"--max-count={limit}", f"--pretty=format:{fmt}"])
    commits: list[dict[str, Any]] = []
    if ok and out:
        for line in out.splitlines():
            parts = line.split("\x1f", 3)
            if len(parts) != 4:
                continue
            sha, short_sha, ts, subject = parts
            try:
                author_time: int | None = int(ts)
            except ValueError:
                author_time = None
            commits.append({
                "sha": sha,
                "short_sha": short_sha,
                "subject": subject,
                "author_time": author_time,
            })

    await ctx.send_frame(session, ws, payload={
        "action": proto.FILES_COMMITS_RESULT,
        "channel_id": channel_id,
        "repo_path": _repo_display_path(repo, cwd),
        "commits": commits,
        "error": None if ok else "Unable to list commits",
    })


# ---------------------------------------------------------------------------
# file_read
# ---------------------------------------------------------------------------


async def handle_file_read(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Read file contents for the viewer (text / image / generic binary)."""
    # validator: channel_id + path required.
    channel_id = payload["channel_id"]
    rel_path = payload["path"]
    offset = payload.get("offset", 0)
    limit = min(payload.get("limit", _MAX_FILE_READ), _MAX_FILE_READ)

    cwd = ctx.get_channel_cwd(channel_id)
    resolved = ctx.resolve_safe_path(cwd, rel_path)
    if resolved is None:
        await ctx.send_frame(session, ws, payload={
            "action": proto.ERROR,
            "channel_id": channel_id,
            "error": "Path outside working directory",
        })
        return

    try:
        stat = os.stat(resolved)
    except OSError as exc:
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILE_READ_RESULT,
            "channel_id": channel_id,
            "path": rel_path,
            "error": str(exc),
        })
        return

    file_size = stat.st_size

    # Binary detection: check first 8 KB for null bytes.
    try:
        with open(resolved, "rb") as f:
            header = f.read(_BINARY_CHECK_SIZE)
    except OSError as exc:
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILE_READ_RESULT,
            "channel_id": channel_id,
            "path": rel_path,
            "error": str(exc),
        })
        return

    is_binary = b"\x00" in header

    if is_binary:
        ext = os.path.splitext(resolved)[1].lower()
        if ext in _IMAGE_EXTS and file_size <= _MAX_IMAGE_SIZE:
            mime = mimetypes.guess_type(str(resolved))[0] or "image/png"
            try:
                with open(resolved, "rb") as f:
                    raw = f.read()
                b64 = base64.b64encode(raw).decode("ascii")
                data_uri = f"data:{mime};base64,{b64}"

                # Send in chunks to stay within WS frame limits.
                chunk_size = 500_000  # ~500 KB per chunk (base64 text)
                chunks = [data_uri[i:i + chunk_size] for i in range(0, len(data_uri), chunk_size)]
                for idx, chunk in enumerate(chunks):
                    await ctx.send_frame(session, ws, payload={
                        "action": proto.FILE_READ_RESULT,
                        "channel_id": channel_id,
                        "path": rel_path,
                        "content": chunk,
                        "size": file_size,
                        "offset": 0,
                        "truncated": False,
                        "encoding": "base64",
                        "is_binary": True,
                        "is_image": True,
                        "chunk_index": idx,
                        "chunk_total": len(chunks),
                    })
                return
            except OSError:
                pass  # Fall through to generic binary message.

        await ctx.send_frame(session, ws, payload={
            "action": proto.FILE_READ_RESULT,
            "channel_id": channel_id,
            "path": rel_path,
            "content": None,
            "size": file_size,
            "offset": 0,
            "truncated": False,
            "encoding": None,
            "is_binary": True,
        })
        return

    # Text file: read from offset up to limit bytes.
    try:
        with open(resolved, "rb") as f:
            if offset > 0:
                f.seek(offset)
            raw = f.read(limit)
    except OSError as exc:
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILE_READ_RESULT,
            "channel_id": channel_id,
            "path": rel_path,
            "error": str(exc),
        })
        return

    content = raw.decode("utf-8", errors="replace")

    await ctx.send_frame(session, ws, payload={
        "action": proto.FILE_READ_RESULT,
        "channel_id": channel_id,
        "path": rel_path,
        "content": content,
        "size": file_size,
        "offset": offset,
        "truncated": file_size > offset + limit,
        "encoding": "utf-8",
        "is_binary": False,
    })


# ---------------------------------------------------------------------------
# file_diff
# ---------------------------------------------------------------------------


async def handle_file_diff(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Get git diff for a file (worktree, staged, or between refs)."""
    # validator: channel_id + path required.
    channel_id = payload["channel_id"]
    rel_path = payload["path"]
    staged = payload.get("staged", False)
    repo_path = payload.get("repo_path")
    newer_ref = payload.get("newer_ref")
    older_ref = payload.get("older_ref")

    cwd = ctx.get_channel_cwd(channel_id)
    resolved = ctx.resolve_safe_path(cwd, rel_path)
    if resolved is None:
        await ctx.send_frame(session, ws, payload={
            "action": proto.ERROR,
            "channel_id": channel_id,
            "error": "Path outside working directory",
        })
        return

    repo = (
        _resolve_repo_path(ctx, cwd, repo_path)
        if repo_path is not None else find_git_repo(resolved)
    )
    if not repo:
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILE_DIFF_RESULT,
            "channel_id": channel_id,
            "path": rel_path,
            "staged": staged,
            "diff": "",
            "truncated": False,
        })
        return

    # Compute path relative to repo root.
    try:
        file_repo_rel = str(Path(resolved).resolve().relative_to(Path(repo).resolve()))
    except ValueError:
        await ctx.send_frame(session, ws, payload={
            "action": proto.FILE_DIFF_RESULT,
            "channel_id": channel_id,
            "path": rel_path,
            "repo_path": repo_path,
            "newer_ref": newer_ref,
            "older_ref": older_ref,
            "error": "Path outside repository",
            "diff": "",
            "truncated": False,
        })
        return

    if newer_ref is not None or older_ref is not None or repo_path is not None:
        selected_newer = newer_ref or "worktree"
        selected_older = older_ref or "HEAD"
        validated_older = await _validate_commit_ref(repo, selected_older)
        if not validated_older:
            await ctx.send_frame(session, ws, payload={
                "action": proto.FILE_DIFF_RESULT,
                "channel_id": channel_id,
                "path": rel_path,
                "repo_path": _repo_display_path(repo, cwd),
                "newer_ref": selected_newer,
                "older_ref": selected_older,
                "error": "Invalid older ref",
                "diff": "",
                "truncated": False,
            })
            return
        if selected_newer == "worktree":
            diff_out, ok = await _run_git(repo, ["diff", validated_older, "--", file_repo_rel])
            if not diff_out:
                status_out, _ = await _run_git(repo, ["status", "--porcelain=v1", "--", file_repo_rel])
                if status_out.startswith("??"):
                    diff_out, _ = await _run_git(repo, ["diff", "--no-index", "/dev/null", file_repo_rel])
        else:
            validated_newer = await _validate_commit_ref(repo, selected_newer)
            if not validated_newer:
                await ctx.send_frame(session, ws, payload={
                    "action": proto.FILE_DIFF_RESULT,
                    "channel_id": channel_id,
                    "path": rel_path,
                    "repo_path": _repo_display_path(repo, cwd),
                    "newer_ref": selected_newer,
                    "older_ref": selected_older,
                    "error": "Invalid newer ref",
                    "diff": "",
                    "truncated": False,
                })
                return
            diff_out, ok = await _run_git(repo, ["diff", validated_older, validated_newer, "--", file_repo_rel])
    elif staged:
        diff_out, ok = await _run_git(repo, ["diff", "--cached", "--", file_repo_rel])
    else:
        diff_out, ok = await _run_git(repo, ["diff", "--", file_repo_rel])

    # If no diff and the file might be untracked, generate a synthetic diff.
    if not diff_out:
        status_out, _ = await _run_git(repo, ["status", "--porcelain=v1", "--", file_repo_rel])
        if status_out.startswith("??"):
            diff_out, _ = await _run_git(repo, ["diff", "--no-index", "/dev/null", file_repo_rel])

    truncated = len(diff_out) > _MAX_DIFF_SIZE
    if truncated:
        diff_out = diff_out[:_MAX_DIFF_SIZE]

    await ctx.send_frame(session, ws, payload={
        "action": proto.FILE_DIFF_RESULT,
        "channel_id": channel_id,
        "path": rel_path,
        "repo_path": _repo_display_path(repo, cwd),
        "newer_ref": newer_ref,
        "older_ref": older_ref,
        "staged": staged,
        "diff": diff_out,
        "truncated": truncated,
    })
