"""Local SQLite storage for E2EE channels and messages."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path.home() / ".config" / "build" / "messages.db"


@dataclass
class Channel:
    id: str
    name: str
    created_at: float


@dataclass
class Message:
    id: str
    channel_id: str
    session_id: str
    sender: str  # "client" | agent name (e.g., "Claude Code", "Codex CLI")
    content: str
    created_at: float
    delivered_at: float | None = None
    read_at: float | None = None
    attachments: list[dict[str, Any]] | None = None  # [{file_id, filename, size, mime_type, path}]


@dataclass
class Project:
    id: str
    name: str
    root_path: str
    repo: str
    default_branch: str
    color: str
    created_at: float
    updated_at: float


@dataclass
class Worktree:
    id: str
    project_id: str
    name: str
    path: str
    branch: str
    status: str
    created_at: float
    updated_at: float
    channel_id: str | None = None
    base_ref: str = ""
    head_ref: str = ""


@dataclass
class Plan:
    id: str
    project_id: str
    title: str
    status: str
    created_at: float
    updated_at: float
    worktree_id: str | None = None
    channel_id: str | None = None
    body: str = ""
    step_count: int = 1
    done_step_count: int = 0
    model: str = ""


class MessageStore:
    """SQLite-backed local message store for the device."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS channels (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL REFERENCES channels(id),
                session_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                delivered_at REAL,
                read_at REAL,
                attachments TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_messages_channel
                ON messages(channel_id, created_at);

            CREATE TABLE IF NOT EXISTS projects (
                id             TEXT PRIMARY KEY,
                name           TEXT NOT NULL,
                root_path      TEXT NOT NULL DEFAULT '',
                repo           TEXT NOT NULL DEFAULT '',
                default_branch TEXT NOT NULL DEFAULT 'main',
                color          TEXT NOT NULL DEFAULT '',
                created_at     REAL NOT NULL,
                updated_at     REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_projects_updated
                ON projects(updated_at);

            CREATE TABLE IF NOT EXISTS worktrees (
                id         TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id),
                channel_id TEXT UNIQUE,
                name       TEXT NOT NULL,
                path       TEXT NOT NULL DEFAULT '',
                branch     TEXT NOT NULL DEFAULT '',
                base_ref   TEXT NOT NULL DEFAULT '',
                head_ref   TEXT NOT NULL DEFAULT '',
                status     TEXT NOT NULL DEFAULT 'idle',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_worktrees_project
                ON worktrees(project_id, updated_at);

            CREATE TABLE IF NOT EXISTS plans (
                id              TEXT PRIMARY KEY,
                project_id      TEXT NOT NULL REFERENCES projects(id),
                worktree_id     TEXT REFERENCES worktrees(id),
                channel_id      TEXT,
                title           TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'draft',
                body            TEXT NOT NULL DEFAULT '',
                step_count      INTEGER NOT NULL DEFAULT 1,
                done_step_count INTEGER NOT NULL DEFAULT 0,
                model           TEXT NOT NULL DEFAULT '',
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_plans_project
                ON plans(project_id, updated_at);

            CREATE INDEX IF NOT EXISTS idx_plans_worktree
                ON plans(worktree_id, updated_at);
        """)
        # Migration: add attachments column if missing (existing databases).
        try:
            self.db.execute("SELECT attachments FROM messages LIMIT 0")
        except sqlite3.OperationalError:
            self.db.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")

    # ----- Projects / worktrees -----

    def upsert_project(
        self,
        project_id: str,
        name: str,
        *,
        root_path: str = "",
        repo: str = "",
        default_branch: str = "main",
        color: str = "",
    ) -> Project:
        """Create or update a project primitive."""
        now = time.time()
        existing = self.get_project(project_id)
        created_at = existing.created_at if existing else now
        changed = not existing or (
            existing.name != name
            or existing.root_path != root_path
            or existing.repo != repo
            or existing.default_branch != default_branch
            or existing.color != color
        )
        updated_at = now if changed else existing.updated_at
        self.db.execute(
            """INSERT INTO projects
               (id, name, root_path, repo, default_branch, color, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   name = excluded.name,
                   root_path = excluded.root_path,
                   repo = excluded.repo,
                   default_branch = excluded.default_branch,
                   color = excluded.color,
                   updated_at = excluded.updated_at""",
            (project_id, name, root_path, repo, default_branch, color, created_at, updated_at),
        )
        self.db.commit()
        return Project(
            id=project_id,
            name=name,
            root_path=root_path,
            repo=repo,
            default_branch=default_branch,
            color=color,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_project(self, project_id: str) -> Project | None:
        row = self.db.execute(
            "SELECT * FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self) -> list[Project]:
        rows = self.db.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_project(row) for row in rows]

    def clear_projects(self) -> dict[str, int]:
        """Delete project primitives without deleting channels or messages."""
        counts = {
            "plans": int(self.db.execute("SELECT COUNT(*) FROM plans").fetchone()[0]),
            "worktrees": int(self.db.execute("SELECT COUNT(*) FROM worktrees").fetchone()[0]),
            "projects": int(self.db.execute("SELECT COUNT(*) FROM projects").fetchone()[0]),
        }
        self.db.execute("DELETE FROM plans")
        self.db.execute("DELETE FROM worktrees")
        self.db.execute("DELETE FROM projects")
        self.db.commit()
        return counts

    def upsert_worktree(
        self,
        worktree_id: str,
        project_id: str,
        name: str,
        *,
        path: str = "",
        branch: str = "",
        status: str = "idle",
        channel_id: str | None = None,
        base_ref: str = "",
        head_ref: str = "",
    ) -> Worktree:
        """Create or update a worktree primitive."""
        now = time.time()
        existing = self.get_worktree(worktree_id)
        if existing is None and channel_id:
            existing = self.get_worktree_by_channel(channel_id)
            if existing:
                worktree_id = existing.id
        created_at = existing.created_at if existing else now
        changed = not existing or (
            existing.project_id != project_id
            or existing.channel_id != channel_id
            or existing.name != name
            or existing.path != path
            or existing.branch != branch
            or existing.base_ref != base_ref
            or existing.head_ref != head_ref
            or existing.status != status
        )
        updated_at = now if changed else existing.updated_at
        self.db.execute(
            """INSERT INTO worktrees
               (id, project_id, channel_id, name, path, branch, base_ref, head_ref, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   project_id = excluded.project_id,
                   channel_id = excluded.channel_id,
                   name = excluded.name,
                   path = excluded.path,
                   branch = excluded.branch,
                   base_ref = excluded.base_ref,
                   head_ref = excluded.head_ref,
                   status = excluded.status,
                   updated_at = excluded.updated_at""",
            (
                worktree_id,
                project_id,
                channel_id,
                name,
                path,
                branch,
                base_ref,
                head_ref,
                status,
                created_at,
                updated_at,
            ),
        )
        self.db.commit()
        return Worktree(
            id=worktree_id,
            project_id=project_id,
            channel_id=channel_id,
            name=name,
            path=path,
            branch=branch,
            base_ref=base_ref,
            head_ref=head_ref,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_worktree(self, worktree_id: str) -> Worktree | None:
        row = self.db.execute(
            "SELECT * FROM worktrees WHERE id = ?",
            (worktree_id,),
        ).fetchone()
        return self._row_to_worktree(row) if row else None

    def get_worktree_by_channel(self, channel_id: str) -> Worktree | None:
        row = self.db.execute(
            "SELECT * FROM worktrees WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
        return self._row_to_worktree(row) if row else None

    def list_worktrees(self, project_id: str | None = None) -> list[Worktree]:
        if project_id:
            rows = self.db.execute(
                "SELECT * FROM worktrees WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM worktrees ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_worktree(row) for row in rows]

    def clear_worktree_channel(self, channel_id: str) -> None:
        """Detach a deleted channel from its worktree without deleting the project."""
        self.db.execute(
            "UPDATE worktrees SET channel_id = NULL, status = 'idle', updated_at = ? WHERE channel_id = ?",
            (time.time(), channel_id),
        )
        self.db.execute(
            "UPDATE plans SET channel_id = NULL, status = 'draft', updated_at = ? WHERE channel_id = ?",
            (time.time(), channel_id),
        )
        self.db.commit()

    def upsert_plan(
        self,
        plan_id: str,
        project_id: str,
        title: str,
        *,
        worktree_id: str | None = None,
        channel_id: str | None = None,
        status: str = "draft",
        body: str = "",
        step_count: int = 1,
        done_step_count: int = 0,
        model: str = "",
    ) -> Plan:
        """Create or update a plan primitive."""
        now = time.time()
        existing = self.get_plan(plan_id)
        created_at = existing.created_at if existing else now
        changed = not existing or (
            existing.project_id != project_id
            or existing.worktree_id != worktree_id
            or existing.channel_id != channel_id
            or existing.title != title
            or existing.status != status
            or existing.body != body
            or existing.step_count != step_count
            or existing.done_step_count != done_step_count
            or existing.model != model
        )
        updated_at = now if changed else existing.updated_at
        self.db.execute(
            """INSERT INTO plans
               (id, project_id, worktree_id, channel_id, title, status, body, step_count, done_step_count, model, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   project_id = excluded.project_id,
                   worktree_id = excluded.worktree_id,
                   channel_id = excluded.channel_id,
                   title = excluded.title,
                   status = excluded.status,
                   body = excluded.body,
                   step_count = excluded.step_count,
                   done_step_count = excluded.done_step_count,
                   model = excluded.model,
                   updated_at = excluded.updated_at""",
            (
                plan_id,
                project_id,
                worktree_id,
                channel_id,
                title,
                status,
                body,
                step_count,
                done_step_count,
                model,
                created_at,
                updated_at,
            ),
        )
        self.db.commit()
        return Plan(
            id=plan_id,
            project_id=project_id,
            worktree_id=worktree_id,
            channel_id=channel_id,
            title=title,
            status=status,
            body=body,
            step_count=step_count,
            done_step_count=done_step_count,
            model=model,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_plan(self, plan_id: str) -> Plan | None:
        row = self.db.execute(
            "SELECT * FROM plans WHERE id = ?",
            (plan_id,),
        ).fetchone()
        return self._row_to_plan(row) if row else None

    def list_plans(
        self,
        project_id: str | None = None,
        worktree_id: str | None = None,
    ) -> list[Plan]:
        if worktree_id:
            rows = self.db.execute(
                "SELECT * FROM plans WHERE worktree_id = ? ORDER BY updated_at DESC",
                (worktree_id,),
            ).fetchall()
        elif project_id:
            rows = self.db.execute(
                "SELECT * FROM plans WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM plans ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_plan(row) for row in rows]

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            root_path=row["root_path"],
            repo=row["repo"],
            default_branch=row["default_branch"],
            color=row["color"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_plan(row: sqlite3.Row) -> Plan:
        return Plan(
            id=row["id"],
            project_id=row["project_id"],
            worktree_id=row["worktree_id"],
            channel_id=row["channel_id"],
            title=row["title"],
            status=row["status"],
            body=row["body"],
            step_count=row["step_count"],
            done_step_count=row["done_step_count"],
            model=row["model"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_worktree(row: sqlite3.Row) -> Worktree:
        return Worktree(
            id=row["id"],
            project_id=row["project_id"],
            channel_id=row["channel_id"],
            name=row["name"],
            path=row["path"],
            branch=row["branch"],
            base_ref=row["base_ref"],
            head_ref=row["head_ref"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_channel(self, channel_id: str, name: str) -> Channel:
        """Create a new channel."""
        now = time.time()
        self.db.execute(
            "INSERT INTO channels (id, name, created_at) VALUES (?, ?, ?)",
            (channel_id, name, now),
        )
        self.db.commit()
        return Channel(id=channel_id, name=name, created_at=now)

    def get_channel(self, channel_id: str) -> Channel | None:
        """Get a channel by ID."""
        row = self.db.execute(
            "SELECT id, name, created_at FROM channels WHERE id = ?",
            (channel_id,),
        ).fetchone()
        if not row:
            return None
        return Channel(id=row["id"], name=row["name"], created_at=row["created_at"])

    def list_channels(self) -> list[Channel]:
        """List all channels ordered by creation time."""
        rows = self.db.execute(
            "SELECT id, name, created_at FROM channels ORDER BY created_at DESC"
        ).fetchall()
        return [
            Channel(id=r["id"], name=r["name"], created_at=r["created_at"])
            for r in rows
        ]

    def rename_channel(self, channel_id: str, name: str) -> None:
        """Rename a channel."""
        self.db.execute(
            "UPDATE channels SET name = ? WHERE id = ?", (name, channel_id)
        )
        self.db.commit()

    def delete_channel(self, channel_id: str) -> None:
        """Delete a channel and all its messages."""
        self.db.execute(
            "UPDATE worktrees SET channel_id = NULL, status = 'idle', updated_at = ? WHERE channel_id = ?",
            (time.time(), channel_id),
        )
        self.db.execute(
            "UPDATE plans SET channel_id = NULL, status = 'draft', updated_at = ? WHERE channel_id = ?",
            (time.time(), channel_id),
        )
        self.db.execute(
            "DELETE FROM messages WHERE channel_id = ?", (channel_id,)
        )
        self.db.execute(
            "DELETE FROM channels WHERE id = ?", (channel_id,)
        )
        self.db.commit()

    def store_message(
        self,
        message_id: str,
        channel_id: str,
        session_id: str,
        sender: str,
        content: str,
        created_at: float | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> Message:
        """Store a message."""
        now = created_at or time.time()
        attachments_json = json.dumps(attachments) if attachments else None
        self.db.execute(
            """INSERT OR REPLACE INTO messages
               (id, channel_id, session_id, sender, content, created_at, attachments)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (message_id, channel_id, session_id, sender, content, now, attachments_json),
        )
        self.db.commit()
        return Message(
            id=message_id,
            channel_id=channel_id,
            session_id=session_id,
            sender=sender,
            content=content,
            created_at=now,
            attachments=attachments,
        )

    def mark_delivered(self, message_id: str) -> None:
        """Mark a message as delivered."""
        self.db.execute(
            "UPDATE messages SET delivered_at = ? WHERE id = ?",
            (time.time(), message_id),
        )
        self.db.commit()

    def mark_read(self, message_id: str) -> None:
        """Mark a message as read."""
        now = time.time()
        self.db.execute(
            "UPDATE messages SET read_at = ?, delivered_at = COALESCE(delivered_at, ?) WHERE id = ?",
            (now, now, message_id),
        )
        self.db.commit()

    def get_message(self, message_id: str) -> Message | None:
        """Get a single message by ID."""
        row = self.db.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if not row:
            return None
        return Message(
            id=row["id"],
            channel_id=row["channel_id"],
            session_id=row["session_id"],
            sender=row["sender"],
            content=row["content"],
            created_at=row["created_at"],
            delivered_at=row["delivered_at"],
            read_at=row["read_at"],
            attachments=json.loads(row["attachments"]) if row["attachments"] else None,
        )

    def get_messages(
        self,
        channel_id: str,
        limit: int = 50,
        before: float | None = None,
    ) -> list[Message]:
        """Get messages for a channel, most recent first."""
        if before:
            rows = self.db.execute(
                """SELECT * FROM messages
                   WHERE channel_id = ? AND created_at < ?
                   ORDER BY created_at DESC LIMIT ?""",
                (channel_id, before, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT * FROM messages
                   WHERE channel_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (channel_id, limit),
            ).fetchall()

        return [
            Message(
                id=r["id"],
                channel_id=r["channel_id"],
                session_id=r["session_id"],
                sender=r["sender"],
                content=r["content"],
                created_at=r["created_at"],
                delivered_at=r["delivered_at"],
                read_at=r["read_at"],
                attachments=json.loads(r["attachments"]) if r["attachments"] else None,
            )
            for r in reversed(rows)  # Return in chronological order
        ]

    def get_unread_messages(self, channel_id: str) -> list[Message]:
        """Get unread client messages on a channel (read_at IS NULL, sender='client')."""
        rows = self.db.execute(
            "SELECT * FROM messages WHERE channel_id = ? AND sender = 'client' "
            "AND read_at IS NULL ORDER BY created_at ASC",
            (channel_id,),
        ).fetchall()
        return [
            Message(
                id=r["id"],
                channel_id=r["channel_id"],
                session_id=r["session_id"],
                sender=r["sender"],
                content=r["content"],
                created_at=r["created_at"],
                delivered_at=r["delivered_at"],
                read_at=r["read_at"],
                attachments=json.loads(r["attachments"]) if r["attachments"] else None,
            )
            for r in rows
        ]

    def close(self) -> None:
        self.db.close()
