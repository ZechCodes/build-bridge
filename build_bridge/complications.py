"""Complications — real-time context widgets pushed from client device to web UI.

Complications observe agent tool execution, evaluate context (e.g. git state),
and push updates to the browser over the existing E2EE broadcast pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Awaitable, Protocol

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FileDiffSummary:
    added: int = 0
    modified: int = 0
    deleted: int = 0

    @property
    def total(self) -> int:
        return self.added + self.modified + self.deleted


@dataclass
class GitStatusData:
    repo: str
    branch: str
    upstream: str | None = None
    ahead: int = 0
    behind: int = 0
    staged: FileDiffSummary = field(default_factory=FileDiffSummary)
    unstaged: FileDiffSummary = field(default_factory=FileDiffSummary)
    untracked: int = 0
    conflicts: int = 0
    stash_count: int = 0
    detached: bool = False
    last_fetch: float | None = None
    remote_name: str | None = None  # e.g. "owner/repo"
    insertions: int = 0  # total line insertions (staged + unstaged)
    deletions: int = 0  # total line deletions (staged + unstaged)


@dataclass
class ComplicationOption:
    id: str
    label: str
    enabled: bool = True
    confirm: bool = False
    icon: str | None = None


@dataclass
class ComplicationMessage:
    channel_id: str
    id: str  # stable identifier, e.g. "git:/path/to/repo"
    kind: str  # e.g. "git-status"
    timestamp: float  # unix ms
    data: dict[str, Any]
    options: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

# Tools that mutate files and should trigger complication re-evaluation.
FILE_MUTATION_TOOLS = frozenset({"Edit", "Write", "Bash", "ApplyPatch"})


@lru_cache(maxsize=256)
def find_git_repo(file_path: str) -> str | None:
    """Walk up from file_path to find a .git directory."""
    try:
        expanded = os.path.expandvars(os.path.expanduser(file_path))
        p = Path(expanded).resolve()
        # Check the path itself first, then walk up parents.
        for candidate in [p, *p.parents]:
            if (candidate / ".git").is_dir():
                return str(candidate)
    except (OSError, ValueError):
        pass
    return None


def invalidate_git_repo_cache() -> None:
    """Clear the git repo lookup cache (e.g. after directory changes)."""
    find_git_repo.cache_clear()


async def _run_git(repo: str, args: list[str], timeout: float = 10.0, stdin_data: bytes | None = None) -> tuple[str, bool]:
    """Run a git command in the given repo. Returns (stdout, success)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=repo,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=stdin_data), timeout=timeout)
        return stdout.decode().strip('\n'), proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as exc:
        log.debug("git %s failed in %s: %s", " ".join(args), repo, exc)
        return "", False


async def git_branch_info(repo: str) -> tuple[str, bool, str | None]:
    """Returns (branch_name, is_detached, upstream_or_None)."""
    branch, ok = await _run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    if not ok:
        return "unknown", False, None

    detached = branch == "HEAD"
    if detached:
        # Get short SHA for detached HEAD.
        sha, _ = await _run_git(repo, ["rev-parse", "--short", "HEAD"])
        branch = sha or "detached"

    upstream: str | None = None
    if not detached:
        up, up_ok = await _run_git(repo, ["rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}"])
        if up_ok and up:
            upstream = up

    return branch, detached, upstream


async def git_ahead_behind(repo: str, upstream: str | None) -> tuple[int, int]:
    """Returns (ahead, behind) counts relative to upstream."""
    if not upstream:
        return 0, 0
    out, ok = await _run_git(repo, ["rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
    if not ok or not out:
        return 0, 0
    try:
        parts = out.split()
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


def _parse_numstat(output: str) -> tuple[FileDiffSummary, int, int]:
    """Parse git diff --numstat output into (FileDiffSummary, insertions, deletions)."""
    added = modified = deleted = 0
    total_ins = total_dels = 0
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ins, dels = parts[0], parts[1]
        if ins == "-":  # binary file
            modified += 1
            continue
        try:
            i, d = int(ins), int(dels)
        except ValueError:
            continue
        total_ins += i
        total_dels += d
        if d == 0 and i > 0:
            added += 1
        elif i == 0 and d > 0:
            deleted += 1
        else:
            modified += 1
    return FileDiffSummary(added=added, modified=modified, deleted=deleted), total_ins, total_dels


async def git_diff_summary(repo: str) -> tuple[FileDiffSummary, FileDiffSummary, int, int]:
    """Returns (staged, unstaged, total_insertions, total_deletions)."""
    staged_out, _ = await _run_git(repo, ["diff", "--cached", "--numstat"])
    unstaged_out, _ = await _run_git(repo, ["diff", "--numstat"])
    staged, s_ins, s_dels = _parse_numstat(staged_out)
    unstaged, u_ins, u_dels = _parse_numstat(unstaged_out)
    return staged, unstaged, s_ins + u_ins, s_dels + u_dels


async def git_untracked_count(repo: str) -> int:
    """Count untracked files."""
    out, ok = await _run_git(repo, ["ls-files", "--others", "--exclude-standard"])
    if not ok or not out:
        return 0
    return len(out.splitlines())


async def git_conflict_count(repo: str) -> int:
    """Count files with merge conflicts."""
    out, ok = await _run_git(repo, ["diff", "--name-only", "--diff-filter=U"])
    if not ok or not out:
        return 0
    return len(out.splitlines())


async def git_stash_count(repo: str) -> int:
    """Count stash entries."""
    out, ok = await _run_git(repo, ["rev-list", "--walk-reflogs", "--count", "refs/stash"])
    if not ok or not out:
        return 0
    try:
        return int(out)
    except ValueError:
        return 0


def last_fetch_time(repo: str) -> float | None:
    """Return the mtime of .git/FETCH_HEAD as unix ms, or None."""
    fetch_head = Path(repo) / ".git" / "FETCH_HEAD"
    try:
        return fetch_head.stat().st_mtime * 1000
    except OSError:
        return None


async def git_remote_name(repo: str) -> str | None:
    """Extract a human-readable remote name like 'owner/repo' from the origin URL."""
    url, ok = await _run_git(repo, ["remote", "get-url", "origin"])
    if not ok or not url:
        return None
    # Handle SSH: git@github.com:owner/repo.git
    m = re.search(r"[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


async def evaluate_git_status(repo: str) -> GitStatusData:
    """Evaluate full git status for a repository."""
    branch, detached, upstream = await git_branch_info(repo)
    ahead, behind = await git_ahead_behind(repo, upstream)
    staged, unstaged, insertions, deletions = await git_diff_summary(repo)
    untracked = await git_untracked_count(repo)
    conflicts = await git_conflict_count(repo)
    stashes = await git_stash_count(repo)
    remote = await git_remote_name(repo)

    return GitStatusData(
        repo=repo,
        branch=branch,
        upstream=upstream,
        ahead=ahead,
        behind=behind,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        conflicts=conflicts,
        stash_count=stashes,
        detached=detached,
        last_fetch=last_fetch_time(repo),
        remote_name=remote,
        insertions=insertions,
        deletions=deletions,
    )


def build_git_options(data: GitStatusData) -> list[dict[str, Any]]:
    """Build contextual action options based on git state."""
    options: list[dict[str, Any]] = []

    if data.upstream:
        options.append(asdict(ComplicationOption(
            id="push", label="Push",
            enabled=data.ahead > 0 and data.conflicts == 0,
        )))
        options.append(asdict(ComplicationOption(
            id="pull", label="Pull",
            enabled=data.behind > 0 and data.conflicts == 0,
        )))

    options.append(asdict(ComplicationOption(
        id="fetch", label="Fetch", enabled=True,
    )))

    if data.unstaged.total > 0 or data.untracked > 0:
        options.append(asdict(ComplicationOption(
            id="stash", label="Stash", enabled=True,
        )))

    if data.stash_count > 0:
        options.append(asdict(ComplicationOption(
            id="stash-pop", label="Stash Pop",
            enabled=data.conflicts == 0,
        )))

    return options


def _git_status_to_dict(data: GitStatusData) -> dict[str, Any]:
    """Convert GitStatusData to a JSON-serializable dict."""
    return {
        "repo": data.repo,
        "branch": data.branch,
        "upstream": data.upstream,
        "ahead": data.ahead,
        "behind": data.behind,
        "staged": {"added": data.staged.added, "modified": data.staged.modified, "deleted": data.staged.deleted, "total": data.staged.total},
        "unstaged": {"added": data.unstaged.added, "modified": data.unstaged.modified, "deleted": data.unstaged.deleted, "total": data.unstaged.total},
        "untracked": data.untracked,
        "conflicts": data.conflicts,
        "stash_count": data.stash_count,
        "detached": data.detached,
        "last_fetch": data.last_fetch,
        "remote_name": data.remote_name,
        "insertions": data.insertions,
        "deletions": data.deletions,
    }


# ---------------------------------------------------------------------------
# Git action execution
# ---------------------------------------------------------------------------

async def execute_git_action(repo: str, option_id: str) -> tuple[str, bool]:
    """Execute a git action and return (output, success)."""
    match option_id:
        case "push":
            return await _run_git(repo, ["push"], timeout=30)
        case "force-push":
            return await _run_git(repo, ["push", "--force-with-lease"], timeout=30)
        case "pull":
            return await _run_git(repo, ["pull", "--ff-only"], timeout=30)
        case "fetch":
            return await _run_git(repo, ["fetch", "--all", "--prune"], timeout=30)
        case "stash":
            return await _run_git(repo, ["stash", "push", "-u"], timeout=30)
        case "stash-pop":
            return await _run_git(repo, ["stash", "pop"], timeout=30)
        case _:
            return f"Unknown action: {option_id}", False


# ---------------------------------------------------------------------------
# Complication Registry
# ---------------------------------------------------------------------------

# Type alias for the broadcast callback.
BroadcastFn = Callable[[str, dict[str, Any]], Awaitable[None]]


def extract_file_paths(tool_name: str, tool_input: dict[str, Any]) -> list[str]:
    """Extract affected file paths from a tool's input."""
    if tool_name in ("Edit", "Write", "Read"):
        path = tool_input.get("file_path", "")
        return [path] if path else []
    if tool_name == "Glob":
        path = tool_input.get("path", "")
        return [path] if path else []
    if tool_name == "ApplyPatch":
        paths: list[str] = []
        for change in tool_input.get("changes", []) or []:
            if not isinstance(change, dict):
                continue
            path = change.get("filePath") or change.get("path") or ""
            if path:
                paths.append(path)
        return paths
    # Bash: can't reliably extract paths; caller should use working directory.
    return []


class ComplicationRegistry:
    """Manages complication evaluation, debouncing, and broadcasting."""

    def __init__(self, broadcast: BroadcastFn, agent_store: Any | None = None, debounce_ms: int = 500):
        self._broadcast = broadcast
        self._agent_store = agent_store
        self._debounce_ms = debounce_ms
        self._pending: dict[str, asyncio.Task[None]] = {}
        # Track active repos per channel for background polling.
        self._active_repos: dict[str, set[str]] = {}  # channel_id -> set of repo paths
        self._poll_task: asyncio.Task[None] | None = None
        self._poll_interval = 30.0  # seconds
        # Change detection: only bump ordering timestamp when data actually changes.
        self._last_data: dict[str, str] = {}      # comp_id -> json.dumps(data, sort_keys=True)
        self._changed_at: dict[str, float] = {}   # comp_id -> unix ms when data last changed

    def start_polling(self) -> None:
        """Start the background poll loop."""
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())
            log.info("Complications: background poll started (every %.0fs)", self._poll_interval)

    def stop_polling(self) -> None:
        """Stop the background poll loop."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            self._poll_task = None

    async def _poll_loop(self) -> None:
        """Periodically re-evaluate all active repos."""
        try:
            while True:
                await asyncio.sleep(self._poll_interval)
                await self._poll_active_repos()
        except asyncio.CancelledError:
            pass

    async def _poll_active_repos(self) -> None:
        """Re-evaluate all tracked repos across all channels."""
        await self._remove_stale_channels()
        for channel_id, repos in list(self._active_repos.items()):
            for repo in list(repos):
                try:
                    await self._evaluate_and_broadcast(channel_id, repo)
                except Exception as exc:
                    log.debug("Poll evaluation failed for %s in %s: %s", repo, channel_id, exc)

    async def _remove_stale_channels(self) -> None:
        """Remove individual complications whose last change is >1hr before their channel's latest activity."""
        if not self._agent_store:
            return
        try:
            from datetime import datetime, timezone as tz
            rows = self._agent_store.db.execute(
                "SELECT channel_id, MAX(created_at) as latest "
                "FROM activity_log "
                "WHERE type IN ('tool_use', 'tool_result', 'thinking', 'text') "
                "GROUP BY channel_id"
            ).fetchall()
            if not rows:
                return

            latest_by_channel: dict[str, float] = {}
            for r in rows:
                try:
                    dt = datetime.fromisoformat(r["latest"])
                    latest_by_channel[r["channel_id"]] = dt.timestamp() * 1000  # unix ms
                except (ValueError, TypeError):
                    continue

            if not latest_by_channel:
                return

            one_hour_ms = 3_600_000

            for channel_id in list(self._active_repos.keys()):
                ch_latest = latest_by_channel.get(channel_id)
                if ch_latest is None:
                    continue
                repos_to_remove: list[str] = []
                for repo in list(self._active_repos.get(channel_id, set())):
                    comp_id = f"git:{repo}"
                    changed_at = self._changed_at.get(comp_id)
                    if changed_at is not None and (ch_latest - changed_at) > one_hour_ms:
                        repos_to_remove.append(repo)
                for repo in repos_to_remove:
                    self._remove_single_complication(channel_id, repo)
        except Exception as exc:
            log.debug("Stale channel cleanup failed: %s", exc)

    def _remove_single_complication(self, channel_id: str, repo: str) -> None:
        """Remove a single complication for a channel and broadcast removal."""
        comp_id = f"git:{repo}"
        repos = self._active_repos.get(channel_id)
        if repos:
            repos.discard(repo)
            if not repos:
                self._active_repos.pop(channel_id, None)
        self._last_data.pop(comp_id, None)
        self._changed_at.pop(comp_id, None)
        if self._agent_store:
            try:
                self._agent_store.delete_complication(channel_id, comp_id)
            except Exception:
                pass
        asyncio.create_task(self._broadcast(channel_id, {
            "action": "complication:remove",
            "channel_id": channel_id,
            "id": comp_id,
        }))
        # Cancel pending evaluation for this repo.
        key = f"{channel_id}:{repo}"
        task = self._pending.pop(key, None)
        if task:
            task.cancel()

    def _remove_channel_complications(self, channel_id: str) -> None:
        """Remove all complications for a channel."""
        for repo in list(self._active_repos.get(channel_id, set())):
            self._remove_single_complication(channel_id, repo)

    async def on_tool_event(
        self,
        channel_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        working_directory: str = "",
    ) -> None:
        """Called when an agent tool is used. Triggers debounced evaluation."""
        if tool_name not in FILE_MUTATION_TOOLS:
            return

        # Determine which repos are affected.
        repos: set[str] = set()
        file_paths = extract_file_paths(tool_name, tool_input)
        for fp in file_paths:
            repo = find_git_repo(fp)
            if repo:
                repos.add(repo)

        # For Bash commands, prefer the command cwd when the harness provides it.
        fallback_directory = tool_input.get("cwd") if tool_name == "Bash" else None
        if not isinstance(fallback_directory, str) or not fallback_directory:
            fallback_directory = working_directory
        if not repos and fallback_directory:
            repo = find_git_repo(fallback_directory)
            if repo:
                repos.add(repo)

        # If still no repos found (e.g. Bash with no path info), re-evaluate
        # all already-tracked repos for this channel.
        if not repos:
            repos = self._active_repos.get(channel_id, set()).copy()

        self._schedule_evaluation(channel_id, repos)

    async def on_filesystem_change(
        self,
        channel_id: str,
        working_directory: str = "",
    ) -> None:
        """Called when the agent's workspace watcher reports a filesystem
        change. Re-evaluates git status for the repo containing the
        working directory plus any repos already tracked for this channel.

        Unlike on_tool_event(), this fires for *any* file change — reads,
        manual edits in another window, test runs that touch artifacts,
        etc. — so the git chip stays fresh without waiting for the 30s
        polling fallback."""
        repos: set[str] = set()
        if working_directory:
            repo = find_git_repo(working_directory)
            if repo:
                repos.add(repo)
        # Re-eval anything we're already tracking for this channel.
        repos.update(self._active_repos.get(channel_id, set()))
        if not repos:
            return
        self._schedule_evaluation(channel_id, repos)

    def _schedule_evaluation(self, channel_id: str, repos: set[str]) -> None:
        """Queue a debounced evaluation for each of `repos`. Replaces any
        pending evaluation for the same (channel, repo) pair."""
        for repo in repos:
            self._active_repos.setdefault(channel_id, set()).add(repo)
            self.start_polling()
            key = f"{channel_id}:{repo}"
            if key in self._pending:
                self._pending[key].cancel()
            self._pending[key] = asyncio.create_task(
                self._debounced_evaluate(key, channel_id, repo)
            )

    async def _debounced_evaluate(self, key: str, channel_id: str, repo: str) -> None:
        """Wait for debounce window, then evaluate."""
        try:
            await asyncio.sleep(self._debounce_ms / 1000)
            self._pending.pop(key, None)
            await self._evaluate_and_broadcast(channel_id, repo)
        except asyncio.CancelledError:
            self._pending.pop(key, None)

    def _has_data_changed(self, comp_id: str, data_dict: dict[str, Any]) -> bool:
        """Return True if data differs from last broadcast for this comp_id."""
        serialized = json.dumps(data_dict, sort_keys=True)
        previous = self._last_data.get(comp_id)
        self._last_data[comp_id] = serialized
        return previous is None or previous != serialized

    async def _evaluate_and_broadcast(self, channel_id: str, repo: str) -> None:
        """Evaluate git status and broadcast the complication."""
        try:
            data = await evaluate_git_status(repo)
            options = build_git_options(data)
            data_dict = _git_status_to_dict(data)
            comp_id = f"git:{repo}"
            now_ms = time.time() * 1000

            if self._has_data_changed(comp_id, data_dict):
                self._changed_at[comp_id] = now_ms

            changed_at = self._changed_at.get(comp_id, now_ms)
            stable = (now_ms - changed_at) < 300_000  # 5 minutes

            msg: dict[str, Any] = {
                "action": "complication:update",
                "channel_id": channel_id,
                "id": comp_id,
                "kind": "git-status",
                "timestamp": changed_at,
                "changed_at": changed_at,
                "stable": stable,
                "data": data_dict,
                "options": options,
            }
            await self._broadcast(channel_id, msg)
            # Persist to DB so complications survive restarts.
            if self._agent_store:
                try:
                    self._agent_store.save_complication(
                        channel_id, comp_id, "git-status", data_dict, options,
                        changed_at=changed_at,
                    )
                except Exception as exc:
                    log.debug("Failed to persist complication %s: %s", comp_id, exc)
        except Exception as exc:
            log.error("Complication evaluation failed for %s: %s", repo, exc, exc_info=True)

    async def handle_action(
        self,
        channel_id: str,
        complication_id: str,
        option_id: str,
    ) -> None:
        """Handle a user action on a complication (e.g. git push)."""
        if not complication_id.startswith("git:"):
            log.warning("Unknown complication type: %s", complication_id)
            return

        repo = complication_id[4:]  # Strip "git:" prefix.
        log.info("Executing git action %s in %s for channel %s", option_id, repo, channel_id[:8])

        output, success = await execute_git_action(repo, option_id)
        if not success:
            log.warning("Git action %s failed in %s: %s", option_id, repo, output)

        # Re-evaluate and broadcast updated state.
        await self._evaluate_and_broadcast(channel_id, repo)

    async def get_current_complications(
        self, agent_store: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Return the current complication payloads for all tracked repos.

        Used to send initial state when a browser session connects.
        Loads persisted complications from DB for instant display, and seeds
        ``_active_repos`` so the poll loop keeps them fresh.
        """
        store = agent_store or self._agent_store
        results: list[dict[str, Any]] = []

        # 1. Load persisted complications from DB (instant, no git eval).
        if store is not None:
            try:
                for comp in store.get_all_complications():
                    channel_id = comp["channel_id"]
                    comp_id = comp["id"]
                    now_ms = time.time() * 1000
                    changed_at = comp.get("changed_at") or now_ms
                    stable = (now_ms - changed_at) < 300_000
                    results.append({
                        "action": "complication:update",
                        "channel_id": channel_id,
                        "id": comp_id,
                        "kind": comp["kind"],
                        "timestamp": changed_at,
                        "changed_at": changed_at,
                        "stable": stable,
                        "data": comp["data"],
                        "options": comp["options"],
                    })
                    # Seed change-detection state from persisted data.
                    self._last_data[comp_id] = json.dumps(comp["data"], sort_keys=True)
                    self._changed_at[comp_id] = changed_at
                    # Seed _active_repos from persisted complication ids.
                    if comp_id.startswith("git:"):
                        repo = comp_id[4:]
                        self._active_repos.setdefault(channel_id, set()).add(repo)
            except Exception as exc:
                log.debug("Failed to load persisted complications: %s", exc)

        # 2. Also seed from persisted channel working directories.
        if store is not None:
            try:
                for ch in store.list_resumable_channels():
                    if ch.id in self._active_repos:
                        continue  # already tracked
                    wd = getattr(ch, "working_directory", "")
                    if wd:
                        repo = find_git_repo(wd)
                        if repo:
                            self._active_repos.setdefault(ch.id, set()).add(repo)
            except Exception as exc:
                log.debug("Failed to seed repos from agent store: %s", exc)

        # 3. For channels that were seeded from working dirs but had no
        #    persisted complication, evaluate now so they show immediately.
        seen = {(r["channel_id"], r["id"]) for r in results}
        for channel_id, repos in list(self._active_repos.items()):
            for repo in list(repos):
                comp_id = f"git:{repo}"
                if (channel_id, comp_id) in seen:
                    continue
                try:
                    data = await evaluate_git_status(repo)
                    options = build_git_options(data)
                    data_dict = _git_status_to_dict(data)
                    now_ms = time.time() * 1000
                    # First evaluation — mark as changed now.
                    self._has_data_changed(comp_id, data_dict)
                    self._changed_at[comp_id] = now_ms
                    results.append({
                        "action": "complication:update",
                        "channel_id": channel_id,
                        "id": comp_id,
                        "kind": "git-status",
                        "timestamp": now_ms,
                        "changed_at": now_ms,
                        "stable": True,
                        "data": data_dict,
                        "options": options,
                    })
                    # Persist newly evaluated complication.
                    if store:
                        try:
                            store.save_complication(
                                channel_id, comp_id, "git-status", data_dict, options,
                                changed_at=now_ms,
                            )
                        except Exception:
                            pass
                except Exception as exc:
                    log.debug("Failed to get complication for %s: %s", repo, exc)

        # 4. Start polling if we have active repos.
        if self._active_repos:
            self.start_polling()

        return results

    def remove_channel(self, channel_id: str) -> None:
        """Clean up tracking for a removed channel."""
        repos = self._active_repos.pop(channel_id, set())
        # Clean up change-detection state.
        for repo in repos:
            comp_id = f"git:{repo}"
            self._last_data.pop(comp_id, None)
            self._changed_at.pop(comp_id, None)
        # Cancel any pending evaluations for this channel.
        to_cancel = [k for k in self._pending if k.startswith(f"{channel_id}:")]
        for key in to_cancel:
            task = self._pending.pop(key, None)
            if task:
                task.cancel()
        # Stop polling if no active repos remain.
        if not self._active_repos:
            self.stop_polling()
