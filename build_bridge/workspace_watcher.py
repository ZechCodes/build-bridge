"""Workspace filesystem watcher.

Observes the agent's workspace directory for file changes and invokes an
async callback with the set of relative paths that changed within a debounce
window. Used to push AGENT_FILE_CHANGES events to the browser so the files
view can refresh in realtime as the agent edits files.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable, Iterable

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # pragma: no cover - watchdog is a required dep
    Observer = None  # type: ignore[assignment]
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    FileSystemEvent = object  # type: ignore[assignment,misc]


log = logging.getLogger(__name__)

IGNORE_DIR_NAMES = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".turbo", ".cache",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "target",  # rust
})

DEBOUNCE_SECONDS = 0.25


OnChange = Callable[[list[str]], Awaitable[None]]


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: "WorkspaceWatcher") -> None:
        super().__init__()
        self._watcher = watcher

    def on_any_event(self, event: FileSystemEvent) -> None:
        # Skip pure directory events; file paths are what matter.
        if getattr(event, "is_directory", False):
            return
        src = getattr(event, "src_path", None)
        if src:
            self._watcher._ingest(src)
        dest = getattr(event, "dest_path", None)
        if dest:
            self._watcher._ingest(dest)


class WorkspaceWatcher:
    """Debounced recursive filesystem watcher for a workspace."""

    def __init__(
        self,
        repo_path: str | Path,
        on_change: OnChange,
        loop: asyncio.AbstractEventLoop | None = None,
        debounce: float = DEBOUNCE_SECONDS,
    ) -> None:
        # Expand ~ and env vars; otherwise a channel with a literal ~ in its
        # working_directory resolves against the agent's cwd and produces nonsense.
        expanded = os.path.expandvars(os.path.expanduser(str(repo_path)))
        self.repo_path = Path(expanded).resolve()
        self._on_change = on_change
        self._loop = loop or asyncio.get_event_loop()
        self._debounce = debounce
        self._observer: Observer | None = None  # type: ignore[assignment]
        self._pending: set[str] = set()
        self._flush_handle: asyncio.TimerHandle | None = None
        self._lock_owner_thread: int | None = None

    # ---- Lifecycle ----
    def start(self) -> None:
        if Observer is None:
            log.warning("watchdog not installed; workspace watcher disabled")
            return
        if self._observer is not None:
            return
        if not self.repo_path.exists():
            log.info("workspace watcher: path does not exist: %s", self.repo_path)
            return
        observer = Observer()
        observer.schedule(_Handler(self), str(self.repo_path), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer
        log.info("workspace watcher started for %s", self.repo_path)

    def stop(self) -> None:
        obs = self._observer
        self._observer = None
        if obs is not None:
            try:
                obs.stop()
                obs.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                log.debug("workspace watcher stop failed", exc_info=True)
        if self._flush_handle is not None:
            try:
                self._flush_handle.cancel()
            except Exception:  # noqa: BLE001
                pass
        self._flush_handle = None
        self._pending.clear()

    # ---- Event ingestion (called from observer threads) ----
    def _ingest(self, abs_path: str) -> None:
        rel = self._relativize(abs_path)
        if rel is None:
            return
        # Hand off to the loop thread for debounced flush.
        try:
            self._loop.call_soon_threadsafe(self._enqueue, rel)
        except RuntimeError:
            # Loop is closed.
            pass

    def _relativize(self, abs_path: str) -> str | None:
        try:
            p = Path(abs_path)
            try:
                rel = p.relative_to(self.repo_path)
            except ValueError:
                return None
            parts = rel.parts
            for part in parts:
                if part in IGNORE_DIR_NAMES:
                    return None
            # Skip .lock / .swp / macOS artifacts.
            name = p.name
            if name.endswith("~") or name.endswith(".swp") or name == ".DS_Store":
                return None
            return str(rel).replace(os.sep, "/")
        except Exception:  # noqa: BLE001
            return None

    def _enqueue(self, rel_path: str) -> None:
        self._pending.add(rel_path)
        if self._flush_handle is not None:
            self._flush_handle.cancel()
        self._flush_handle = self._loop.call_later(self._debounce, self._flush)

    def _flush(self) -> None:
        self._flush_handle = None
        if not self._pending:
            return
        paths = sorted(self._pending)
        self._pending.clear()
        asyncio.ensure_future(self._dispatch(paths), loop=self._loop)

    async def _dispatch(self, paths: list[str]) -> None:
        try:
            await self._on_change(paths)
        except Exception:  # noqa: BLE001
            log.debug("workspace watcher on_change raised", exc_info=True)


__all__ = ["WorkspaceWatcher", "IGNORE_DIR_NAMES"]
