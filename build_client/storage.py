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
        """)
        # Migration: add attachments column if missing (existing databases).
        try:
            self.db.execute("SELECT attachments FROM messages LIMIT 0")
        except sqlite3.OperationalError:
            self.db.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")

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
