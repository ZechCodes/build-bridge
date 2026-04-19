"""SQLite storage for Build Agent Protocol data (§9).

Tables: agent_channels, chat_messages, activity_log, tool_uses, complications.
Designed to share the same DB file as the E2EE MessageStore.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_bridge.agent_protocol import now_iso


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
    plan_mode: bool = False
    session_start_at: str | None = None
    last_seen_at: str | None = None
    effort: str = ""
    resume_cursor: str = ""  # JSON: session_id (Claude) or thread_id (Codex)
    auto_approve_tools: bool = False


@dataclass
class ChatMessage:
    id: str
    channel_id: str
    role: str  # user | assistant
    content: str  # text or JSON-encoded content block array
    created_at: str
    metadata: str | None = None  # JSON interaction data, NULL for regular messages
    suggested_actions: str | None = None  # JSON array of action labels, NULL for regular messages


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

            CREATE TABLE IF NOT EXISTS complications (
                id          TEXT NOT NULL,
                channel_id  TEXT NOT NULL,
                kind        TEXT NOT NULL,
                data        TEXT NOT NULL,
                options     TEXT NOT NULL DEFAULT '[]',
                updated_at  TEXT NOT NULL,
                PRIMARY KEY (channel_id, id)
            );
        """)

        # Migrations for existing databases.
        for stmt in (
            "ALTER TABLE chat_messages ADD COLUMN metadata TEXT",
            "ALTER TABLE agent_channels ADD COLUMN plan_mode INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE agent_channels ADD COLUMN session_start_at TEXT",
            "ALTER TABLE agent_channels ADD COLUMN last_seen_at TEXT",
            "ALTER TABLE complications ADD COLUMN changed_at REAL",
            "ALTER TABLE chat_messages ADD COLUMN suggested_actions TEXT",
            "ALTER TABLE agent_channels ADD COLUMN effort TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE agent_channels ADD COLUMN resume_cursor TEXT NOT NULL DEFAULT ''",
            # Existing channels migrate with auto_approve_tools=1 so behavior
            # matches what users experienced before the feature landed. New
            # channels default to 0 via the explicit INSERT in create_channel().
            "ALTER TABLE agent_channels ADD COLUMN auto_approve_tools INTEGER NOT NULL DEFAULT 1",
        ):
            try:
                self.db.execute(stmt)
                self.db.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists.

    # ----- Channels -----

    def create_channel(
        self,
        channel_id: str,
        agent_id: str,
        harness: str,
        model: str,
        system_prompt: str = "",
        working_directory: str = "",
        auto_approve_tools: bool = False,
    ) -> AgentChannel:
        """Create a new agent channel."""
        now = now_iso()
        self.db.execute(
            """INSERT INTO agent_channels
               (id, agent_id, harness, model, system_prompt, working_directory, status, created_at, updated_at, auto_approve_tools)
               VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
            (channel_id, agent_id, harness, model, system_prompt, working_directory, now, now, int(auto_approve_tools)),
        )
        self.db.commit()
        return AgentChannel(
            id=channel_id, agent_id=agent_id, harness=harness,
            model=model, system_prompt=system_prompt,
            status="active", created_at=now, updated_at=now,
            working_directory=working_directory,
            auto_approve_tools=auto_approve_tools,
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

    def update_channel_agent(
        self,
        channel_id: str,
        agent_id: str,
        harness: str,
        model: str,
        system_prompt: str = "",
        working_directory: str = "",
    ) -> None:
        """Update a channel with a new agent assignment and reactivate it."""
        self.db.execute(
            "UPDATE agent_channels SET agent_id = ?, harness = ?, model = ?, "
            "system_prompt = ?, working_directory = ?, status = 'active', updated_at = ? WHERE id = ?",
            (agent_id, harness, model, system_prompt, working_directory, now_iso(), channel_id),
        )
        self.db.commit()

    def update_working_directory(self, channel_id: str, working_directory: str) -> None:
        """Update a channel's working directory."""
        self.db.execute(
            "UPDATE agent_channels SET working_directory = ?, updated_at = ? WHERE id = ?",
            (working_directory, now_iso(), channel_id),
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

    def list_resumable_channels(self) -> list[AgentChannel]:
        """List channels that should have agents re-spawned on restart.

        Includes 'active' (agent was working) and 'idle' (agent was waiting
        for input) channels.
        """
        rows = self.db.execute(
            "SELECT * FROM agent_channels WHERE status IN ('active', 'idle') ORDER BY updated_at DESC",
        ).fetchall()
        return [self._row_to_channel(r) for r in rows]

    def delete_channel(self, channel_id: str) -> None:
        """Delete a channel and all related records."""
        self.db.execute("DELETE FROM complications WHERE channel_id = ?", (channel_id,))
        self.db.execute("DELETE FROM tool_uses WHERE channel_id = ?", (channel_id,))
        self.db.execute("DELETE FROM activity_log WHERE channel_id = ?", (channel_id,))
        self.db.execute("DELETE FROM chat_messages WHERE channel_id = ?", (channel_id,))
        self.db.execute("DELETE FROM agent_channels WHERE id = ?", (channel_id,))
        self.db.commit()

    # ----- Chat Messages -----

    def store_chat_message(
        self,
        msg_id: str,
        channel_id: str,
        role: str,
        content: str,
        suggested_actions: list[str] | None = None,
    ) -> ChatMessage:
        """Store a chat message."""
        now = now_iso()
        sa_json = json.dumps(suggested_actions) if suggested_actions else None
        self.db.execute(
            "INSERT OR REPLACE INTO chat_messages (id, channel_id, role, content, created_at, suggested_actions) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, channel_id, role, content, now, sa_json),
        )
        self.db.commit()
        self.touch_channel(channel_id)
        return ChatMessage(id=msg_id, channel_id=channel_id, role=role, content=content, created_at=now, suggested_actions=sa_json)

    def get_chat_history(self, channel_id: str, since: str | None = None) -> list[ChatMessage]:
        """Get chat history for a channel, optionally filtered to messages after `since`."""
        if since:
            rows = self.db.execute(
                "SELECT * FROM chat_messages WHERE channel_id = ? AND created_at > ? ORDER BY created_at ASC",
                (channel_id, since),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM chat_messages WHERE channel_id = ? ORDER BY created_at ASC",
                (channel_id,),
            ).fetchall()
        return [self._row_to_chat_message(r) for r in rows]

    @staticmethod
    def _row_to_chat_message(row: sqlite3.Row) -> ChatMessage:
        keys = row.keys()
        return ChatMessage(
            id=row["id"], channel_id=row["channel_id"],
            role=row["role"], content=row["content"], created_at=row["created_at"],
            metadata=row["metadata"] if "metadata" in keys else None,
            suggested_actions=row["suggested_actions"] if "suggested_actions" in keys else None,
        )

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

    def get_activity_history(self, channel_id: str, since: str | None = None) -> list[ActivityEntry]:
        """Get activity history for a channel, optionally filtered to entries after `since`."""
        if since:
            rows = self.db.execute(
                "SELECT * FROM activity_log WHERE channel_id = ? AND created_at > ? ORDER BY created_at ASC",
                (channel_id, since),
            ).fetchall()
        else:
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
    ) -> str:
        """Update a tool use record with its result. Returns completed_at."""
        now = now_iso()
        output_json = json.dumps(content) if not isinstance(content, str) else content
        self.db.execute(
            "UPDATE tool_uses SET output = ?, is_error = ?, completed_at = ? WHERE id = ?",
            (output_json, is_error, now, tool_use_id),
        )
        self.db.commit()
        return now

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
        keys = row.keys()
        return AgentChannel(
            id=row["id"], agent_id=row["agent_id"],
            harness=row["harness"], model=row["model"],
            system_prompt=row["system_prompt"], status=row["status"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            working_directory=row["working_directory"] if "working_directory" in keys else "",
            plan_mode=bool(row["plan_mode"]) if "plan_mode" in keys else False,
            session_start_at=row["session_start_at"] if "session_start_at" in keys else None,
            last_seen_at=row["last_seen_at"] if "last_seen_at" in keys else None,
            effort=row["effort"] if "effort" in keys else "",
            resume_cursor=row["resume_cursor"] if "resume_cursor" in keys else "",
            auto_approve_tools=bool(row["auto_approve_tools"]) if "auto_approve_tools" in keys else False,
        )

    # ----- Interactions -----

    def store_interaction(
        self,
        interaction_id: str,
        channel_id: str,
        question: str,
        kind: str,
        options: list[dict[str, Any]],
        allow_freeform: bool,
        plan: str | None = None,
        multiselect: bool = False,
    ) -> ChatMessage:
        """Store an interaction request as an assistant chat message with metadata."""
        now = now_iso()
        metadata = json.dumps({
            "interaction_id": interaction_id,
            "kind": kind,
            "options": options,
            "allow_freeform": allow_freeform,
            "plan": plan,
            "multiselect": multiselect,
        })
        self.db.execute(
            "INSERT OR REPLACE INTO chat_messages (id, channel_id, role, content, created_at, metadata) "
            "VALUES (?, ?, 'assistant', ?, ?, ?)",
            (interaction_id, channel_id, question, now, metadata),
        )
        self.db.commit()
        self.touch_channel(channel_id)
        return ChatMessage(
            id=interaction_id, channel_id=channel_id,
            role="assistant", content=question, created_at=now, metadata=metadata,
        )

    def resolve_interaction(
        self,
        interaction_id: str,
        channel_id: str,
        selected_option: str | None,
        freeform_response: str | None,
        selected_options: list[str] | None = None,
    ) -> None:
        """Mark an interaction as resolved and store the user's response."""
        now = now_iso()
        # Update original message metadata with resolution.
        row = self.db.execute(
            "SELECT metadata FROM chat_messages WHERE id = ?", (interaction_id,),
        ).fetchone()
        if row and row["metadata"]:
            meta = json.loads(row["metadata"])
            meta["selected_option"] = selected_option
            meta["selected_options"] = selected_options
            meta["freeform_response"] = freeform_response
            meta["resolved_at"] = now
            self.db.execute(
                "UPDATE chat_messages SET metadata = ? WHERE id = ?",
                (json.dumps(meta), interaction_id),
            )
        # Store user response as a message.
        if selected_options:
            response_text = freeform_response or ", ".join(selected_options)
        else:
            response_text = freeform_response or selected_option or ""
        response_id = f"{interaction_id}_resp"
        self.db.execute(
            "INSERT OR REPLACE INTO chat_messages (id, channel_id, role, content, created_at) "
            "VALUES (?, ?, 'user', ?, ?)",
            (response_id, channel_id, response_text, now),
        )
        self.db.commit()
        self.touch_channel(channel_id)

    def update_plan_mode(self, channel_id: str, enabled: bool) -> None:
        """Update a channel's plan_mode state."""
        self.db.execute(
            "UPDATE agent_channels SET plan_mode = ?, updated_at = ? WHERE id = ?",
            (int(enabled), now_iso(), channel_id),
        )
        self.db.commit()

    def update_auto_approve_tools(self, channel_id: str, enabled: bool) -> None:
        """Update a channel's auto_approve_tools flag."""
        self.db.execute(
            "UPDATE agent_channels SET auto_approve_tools = ?, updated_at = ? WHERE id = ?",
            (int(enabled), now_iso(), channel_id),
        )
        self.db.commit()

    def reset_session(self, channel_id: str) -> str:
        """Mark a new session boundary and clear resume cursor. Returns the timestamp."""
        now = now_iso()
        self.db.execute(
            "UPDATE agent_channels SET session_start_at = ?, resume_cursor = '', updated_at = ? WHERE id = ?",
            (now, now, channel_id),
        )
        self.db.commit()
        return now

    def update_resume_cursor(self, channel_id: str, cursor: str) -> None:
        """Store the provider session resume cursor (session_id or thread_id)."""
        self.db.execute(
            "UPDATE agent_channels SET resume_cursor = ?, updated_at = ? WHERE id = ?",
            (cursor, now_iso(), channel_id),
        )
        self.db.commit()

    def update_effort(self, channel_id: str, effort: str) -> None:
        """Update a channel's effort level."""
        self.db.execute(
            "UPDATE agent_channels SET effort = ?, updated_at = ? WHERE id = ?",
            (effort, now_iso(), channel_id),
        )
        self.db.commit()

    def update_model(self, channel_id: str, model: str) -> None:
        """Update a channel's model."""
        self.db.execute(
            "UPDATE agent_channels SET model = ?, updated_at = ? WHERE id = ?",
            (model, now_iso(), channel_id),
        )
        self.db.commit()

    def mark_channel_seen(self, channel_id: str) -> str:
        """Mark a channel as seen by the user. Returns the timestamp."""
        now = now_iso()
        self.db.execute(
            "UPDATE agent_channels SET last_seen_at = ? WHERE id = ?",
            (now, channel_id),
        )
        self.db.commit()
        return now

    # ----- Complications -----

    def save_complication(
        self,
        channel_id: str,
        comp_id: str,
        kind: str,
        data: dict[str, Any],
        options: list[dict[str, Any]],
        changed_at: float | None = None,
    ) -> None:
        """Upsert a complication payload."""
        self.db.execute(
            "INSERT OR REPLACE INTO complications (id, channel_id, kind, data, options, updated_at, changed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (comp_id, channel_id, kind, json.dumps(data), json.dumps(options), now_iso(), changed_at),
        )
        self.db.commit()

    def get_complications(self, channel_id: str) -> list[dict[str, Any]]:
        """Return all persisted complications for a channel."""
        rows = self.db.execute(
            "SELECT * FROM complications WHERE channel_id = ?", (channel_id,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "channel_id": r["channel_id"],
                "kind": r["kind"],
                "data": json.loads(r["data"]),
                "options": json.loads(r["options"]),
                "updated_at": r["updated_at"],
                "changed_at": r["changed_at"],
            }
            for r in rows
        ]

    def get_all_complications(self) -> list[dict[str, Any]]:
        """Return all persisted complications across all channels."""
        rows = self.db.execute("SELECT * FROM complications").fetchall()
        return [
            {
                "id": r["id"],
                "channel_id": r["channel_id"],
                "kind": r["kind"],
                "data": json.loads(r["data"]),
                "options": json.loads(r["options"]),
                "updated_at": r["updated_at"],
                "changed_at": r["changed_at"],
            }
            for r in rows
        ]

    def delete_complication(self, channel_id: str, comp_id: str) -> None:
        """Remove a persisted complication."""
        self.db.execute(
            "DELETE FROM complications WHERE channel_id = ? AND id = ?",
            (channel_id, comp_id),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
