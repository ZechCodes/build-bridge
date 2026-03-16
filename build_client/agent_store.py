"""SQLite storage for Build Agent Protocol data (§9).

Tables: agent_channels, chat_messages, activity_log, tool_uses.
Designed to share the same DB file as the E2EE MessageStore.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_client.agent_protocol import now_iso


DEFAULT_DB_PATH = Path.home() / ".config" / "build" / "messages.db"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentChannel:
    id: str
    agent_id: str
    harness: str
    model: str
    system_prompt: str
    status: str  # active | idle | closed | error
    created_at: str
    updated_at: str
    working_directory: str = ""


@dataclass
class ChatMessage:
    id: str
    channel_id: str
    role: str  # user | assistant
    content: str  # text or JSON-encoded content block array
    created_at: str


@dataclass
class ActivityEntry:
    id: str
    channel_id: str
    type: str  # text | thinking | tool_use | tool_result
    data: str  # JSON-encoded
    created_at: str


@dataclass
class ToolUseRecord:
    id: str  # tool_use_id
    channel_id: str
    name: str
    input: str  # JSON
    output: str | None  # JSON, NULL until result
    is_error: bool | None  # NULL until result
    created_at: str
    completed_at: str | None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AgentStore:
    """SQLite-backed storage for Build Agent Protocol data."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS agent_channels (
                id          TEXT PRIMARY KEY,
                agent_id    TEXT NOT NULL,
                harness     TEXT NOT NULL,
                model       TEXT NOT NULL,
                system_prompt TEXT NOT NULL DEFAULT '',
                working_directory TEXT NOT NULL DEFAULT '',
                status      TEXT NOT NULL DEFAULT 'active',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_agent_channels_agent_id
                ON agent_channels(agent_id);

            CREATE TABLE IF NOT EXISTS chat_messages (
                id          TEXT PRIMARY KEY,
                channel_id  TEXT NOT NULL REFERENCES agent_channels(id),
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chat_messages_channel
                ON chat_messages(channel_id, created_at);

            CREATE TABLE IF NOT EXISTS activity_log (
                id          TEXT PRIMARY KEY,
                channel_id  TEXT NOT NULL REFERENCES agent_channels(id),
                type        TEXT NOT NULL,
                data        TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_activity_log_channel
                ON activity_log(channel_id, created_at);

            CREATE TABLE IF NOT EXISTS tool_uses (
                id           TEXT PRIMARY KEY,
                channel_id   TEXT NOT NULL REFERENCES agent_channels(id),
                name         TEXT NOT NULL,
                input        TEXT NOT NULL,
                output       TEXT,
                is_error     BOOLEAN,
                created_at   TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_tool_uses_channel
                ON tool_uses(channel_id, created_at);
        """)

    # ----- Channels -----

    def create_channel(
        self,
        channel_id: str,
        agent_id: str,
        harness: str,
        model: str,
        system_prompt: str = "",
        working_directory: str = "",
    ) -> AgentChannel:
        """Create a new agent channel."""
        now = now_iso()
        self.db.execute(
            """INSERT INTO agent_channels
               (id, agent_id, harness, model, system_prompt, working_directory, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
            (channel_id, agent_id, harness, model, system_prompt, working_directory, now, now),
        )
        self.db.commit()
        return AgentChannel(
            id=channel_id, agent_id=agent_id, harness=harness,
            model=model, system_prompt=system_prompt,
            status="active", created_at=now, updated_at=now,
            working_directory=working_directory,
        )

    def get_channel_by_agent_id(self, agent_id: str) -> AgentChannel | None:
        """Find a channel by its agent_id (for reconnection)."""
        row = self.db.execute(
            "SELECT * FROM agent_channels WHERE agent_id = ? ORDER BY created_at DESC LIMIT 1",
            (agent_id,),
        ).fetchone()
        return self._row_to_channel(row) if row else None

    def get_channel(self, channel_id: str) -> AgentChannel | None:
        """Get a channel by ID."""
        row = self.db.execute(
            "SELECT * FROM agent_channels WHERE id = ?", (channel_id,),
        ).fetchone()
        return self._row_to_channel(row) if row else None

    def update_channel_status(self, channel_id: str, status: str) -> None:
        """Update a channel's status and updated_at timestamp."""
        self.db.execute(
            "UPDATE agent_channels SET status = ?, updated_at = ? WHERE id = ?",
            (status, now_iso(), channel_id),
        )
        self.db.commit()

    def touch_channel(self, channel_id: str) -> None:
        """Update a channel's updated_at timestamp."""
        self.db.execute(
            "UPDATE agent_channels SET updated_at = ? WHERE id = ?",
            (now_iso(), channel_id),
        )
        self.db.commit()

    def list_active_channels(self) -> list[AgentChannel]:
        """List all channels with status 'active'."""
        rows = self.db.execute(
            "SELECT * FROM agent_channels WHERE status = 'active' ORDER BY updated_at DESC",
        ).fetchall()
        return [self._row_to_channel(r) for r in rows]

    # ----- Chat Messages -----

    def store_chat_message(
        self,
        msg_id: str,
        channel_id: str,
        role: str,
        content: str,
    ) -> ChatMessage:
        """Store a chat message."""
        now = now_iso()
        self.db.execute(
            "INSERT OR REPLACE INTO chat_messages (id, channel_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (msg_id, channel_id, role, content, now),
        )
        self.db.commit()
        self.touch_channel(channel_id)
        return ChatMessage(id=msg_id, channel_id=channel_id, role=role, content=content, created_at=now)

    def get_chat_history(self, channel_id: str) -> list[ChatMessage]:
        """Get full chat history for a channel, chronological order."""
        rows = self.db.execute(
            "SELECT * FROM chat_messages WHERE channel_id = ? ORDER BY created_at ASC",
            (channel_id,),
        ).fetchall()
        return [
            ChatMessage(
                id=r["id"], channel_id=r["channel_id"],
                role=r["role"], content=r["content"], created_at=r["created_at"],
            )
            for r in rows
        ]

    # ----- Activity Log -----

    def store_activity(
        self,
        channel_id: str,
        entry_type: str,
        data: dict[str, Any],
    ) -> ActivityEntry:
        """Store an activity log entry."""
        entry_id = f"act_{uuid.uuid4().hex[:16]}"
        now = now_iso()
        data_json = json.dumps(data)
        self.db.execute(
            "INSERT INTO activity_log (id, channel_id, type, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (entry_id, channel_id, entry_type, data_json, now),
        )
        self.db.commit()
        self.touch_channel(channel_id)
        return ActivityEntry(
            id=entry_id, channel_id=channel_id,
            type=entry_type, data=data_json, created_at=now,
        )

    def get_activity_history(self, channel_id: str) -> list[ActivityEntry]:
        """Get full activity history for a channel, chronological order."""
        rows = self.db.execute(
            "SELECT * FROM activity_log WHERE channel_id = ? ORDER BY created_at ASC",
            (channel_id,),
        ).fetchall()
        return [
            ActivityEntry(
                id=r["id"], channel_id=r["channel_id"],
                type=r["type"], data=r["data"], created_at=r["created_at"],
            )
            for r in rows
        ]

    # ----- Tool Uses -----

    def store_tool_use(
        self,
        tool_use_id: str,
        channel_id: str,
        name: str,
        tool_input: dict[str, Any],
    ) -> ToolUseRecord:
        """Store a tool use event."""
        now = now_iso()
        input_json = json.dumps(tool_input)
        self.db.execute(
            """INSERT OR REPLACE INTO tool_uses
               (id, channel_id, name, input, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (tool_use_id, channel_id, name, input_json, now),
        )
        self.db.commit()
        self.touch_channel(channel_id)
        return ToolUseRecord(
            id=tool_use_id, channel_id=channel_id, name=name,
            input=input_json, output=None, is_error=None,
            created_at=now, completed_at=None,
        )

    def store_tool_result(
        self,
        tool_use_id: str,
        content: Any,
        is_error: bool,
    ) -> None:
        """Update a tool use record with its result."""
        now = now_iso()
        output_json = json.dumps(content) if not isinstance(content, str) else content
        self.db.execute(
            "UPDATE tool_uses SET output = ?, is_error = ?, completed_at = ? WHERE id = ?",
            (output_json, is_error, now, tool_use_id),
        )
        self.db.commit()

    def get_tool_uses(self, channel_id: str) -> list[ToolUseRecord]:
        """Get all tool uses for a channel, chronological order."""
        rows = self.db.execute(
            "SELECT * FROM tool_uses WHERE channel_id = ? ORDER BY created_at ASC",
            (channel_id,),
        ).fetchall()
        return [
            ToolUseRecord(
                id=r["id"], channel_id=r["channel_id"], name=r["name"],
                input=r["input"], output=r["output"],
                is_error=bool(r["is_error"]) if r["is_error"] is not None else None,
                created_at=r["created_at"], completed_at=r["completed_at"],
            )
            for r in rows
        ]

    # ----- Helpers -----

    @staticmethod
    def _row_to_channel(row: sqlite3.Row) -> AgentChannel:
        return AgentChannel(
            id=row["id"], agent_id=row["agent_id"],
            harness=row["harness"], model=row["model"],
            system_prompt=row["system_prompt"], status=row["status"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            working_directory=row["working_directory"] if "working_directory" in row.keys() else "",
        )

    def close(self) -> None:
        self.db.close()
