"""SQLite persistence for conversations, messages, state, and structured artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from uuid import uuid4

from backend.config import settings
from backend.memory import AgentMemory, ChatTurn, ConversationState


def _now() -> str:
    # Microseconds keep rapid consecutive turns and artifact refinements ordered.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


class ConversationStore:
    """Small repository boundary that can later be replaced by PostgreSQL."""

    def __init__(self, db_path: str | Path | None = None, *, user_id: str | None = None) -> None:
        self.db_path = Path(db_path or settings.conversation_db_path)
        self.user_id = (user_id or settings.conversation_user_id).strip() or "local_user"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                    ON conversations(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    intent TEXT,
                    rewritten_query TEXT,
                    result_json TEXT,
                    paper_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                    ON messages(conversation_id, id);

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    parent_artifact_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_conversation_type
                    ON artifacts(conversation_id, artifact_type, updated_at DESC);

                CREATE TABLE IF NOT EXISTS conversation_states (
                    conversation_id TEXT PRIMARY KEY,
                    current_paper_id TEXT,
                    selected_paper_ids_json TEXT NOT NULL DEFAULT '[]',
                    generated_cards_json TEXT NOT NULL DEFAULT '{}',
                    last_answer TEXT NOT NULL DEFAULT '',
                    last_sources_json TEXT NOT NULL DEFAULT '[]',
                    last_export_path TEXT NOT NULL DEFAULT '',
                    last_intent TEXT,
                    last_user_query TEXT NOT NULL DEFAULT '',
                    rewritten_query TEXT,
                    active_document_id TEXT,
                    active_section TEXT,
                    active_artifact_id TEXT,
                    active_artifact_type TEXT,
                    selected_category INTEGER,
                    filters_json TEXT NOT NULL DEFAULT '{}',
                    last_answer_summary TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """
            )

    def create_conversation(
        self,
        title: str = "新会话",
        *,
        conversation_id: str | None = None,
    ) -> str:
        identifier = conversation_id or uuid4().hex
        timestamp = _now()
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversations(id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (identifier, self.user_id, title[:100] or "新会话", timestamp, timestamp),
            )
            connection.execute(
                "INSERT OR IGNORE INTO conversation_states(conversation_id, updated_at) VALUES (?, ?)",
                (identifier, timestamp),
            )
        return identifier

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count "
                "FROM conversations c LEFT JOIN messages m ON m.conversation_id=c.id "
                "WHERE c.user_id=? GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?",
                (self.user_id, max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def conversation_exists(self, conversation_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversations WHERE id=? AND user_id=?",
                (conversation_id, self.user_id),
            ).fetchone()
        return row is not None

    def load_messages(self, conversation_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        bounded_limit = settings.conversation_message_limit if limit is None else limit
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM (SELECT id, role, content, intent, rewritten_query, result_json, "
                "paper_id, created_at FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?) "
                "ORDER BY id ASC",
                (conversation_id, max(1, int(bounded_limit))),
            ).fetchall()
        messages = []
        for row in rows:
            item = {
                "role": row["role"],
                "content": row["content"],
                "intent": row["intent"],
                "rewritten_query": row["rewritten_query"],
                "created_at": row["created_at"],
            }
            result = _loads(row["result_json"], None)
            if result is not None:
                item["result"] = result
            if row["paper_id"]:
                item["paper_id"] = row["paper_id"]
            messages.append(item)
        return messages

    def load_memory(self, conversation_id: str) -> AgentMemory:
        self.create_conversation(conversation_id=conversation_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_states WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            artifact = None
            if row and row["active_artifact_id"]:
                artifact_row = connection.execute(
                    "SELECT payload_json FROM artifacts WHERE id=? AND conversation_id=?",
                    (row["active_artifact_id"], conversation_id),
                ).fetchone()
                artifact = _loads(artifact_row["payload_json"], {}) if artifact_row else {}
            history_rows = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        if row is None:
            return AgentMemory()
        state = ConversationState(
            last_intent=row["last_intent"],
            last_user_query=row["last_user_query"] or "",
            rewritten_query=row["rewritten_query"],
            active_document_id=row["active_document_id"],
            active_section=row["active_section"],
            active_artifact_id=row["active_artifact_id"],
            active_artifact_type=row["active_artifact_type"],
            active_artifact=artifact or {},
            selected_category=row["selected_category"],
            filters=_loads(row["filters_json"], {}),
            last_answer_summary=row["last_answer_summary"] or "",
        )
        return AgentMemory(
            current_paper_id=row["current_paper_id"],
            selected_paper_ids=_loads(row["selected_paper_ids_json"], []),
            generated_cards=_loads(row["generated_cards_json"], {}),
            last_answer=row["last_answer"] or "",
            last_sources=_loads(row["last_sources_json"], []),
            last_export_path=row["last_export_path"] or "",
            state=state,
            history=[ChatTurn(role=item["role"], content=item["content"]) for item in history_rows],
        )

    def save_exchange(
        self,
        conversation_id: str,
        *,
        user_content: str,
        assistant_content: str,
        memory: AgentMemory,
        intent: str,
        rewritten_query: str | None = None,
        assistant_result: dict[str, Any] | None = None,
        paper_id: str | None = None,
    ) -> None:
        self.create_conversation(conversation_id=conversation_id)
        timestamp = _now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, intent, rewritten_query, created_at) "
                "VALUES (?, 'user', ?, ?, ?, ?)",
                (conversation_id, user_content, intent, rewritten_query, timestamp),
            )
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, intent, rewritten_query, result_json, paper_id, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    assistant_content,
                    intent,
                    rewritten_query,
                    _json(assistant_result) if assistant_result is not None else None,
                    paper_id,
                    timestamp,
                ),
            )
            self._save_memory(connection, conversation_id, memory, timestamp)
            title = " ".join(user_content.split())[:36] or "新会话"
            connection.execute(
                "UPDATE conversations SET title=CASE WHEN title='新会话' THEN ? ELSE title END, updated_at=? "
                "WHERE id=? AND user_id=?",
                (title, timestamp, conversation_id, self.user_id),
            )

    def save_memory(self, conversation_id: str, memory: AgentMemory) -> None:
        self.create_conversation(conversation_id=conversation_id)
        timestamp = _now()
        with self._connection() as connection:
            self._save_memory(connection, conversation_id, memory, timestamp)
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE id=? AND user_id=?",
                (timestamp, conversation_id, self.user_id),
            )

    def _save_memory(
        self,
        connection: sqlite3.Connection,
        conversation_id: str,
        memory: AgentMemory,
        timestamp: str,
    ) -> None:
        state = memory.state
        artifact_id = state.active_artifact_id
        if state.active_artifact and state.active_artifact_type:
            artifact_id = artifact_id or str(
                state.active_artifact.get("artifact_id") or uuid4().hex
            )
            state.active_artifact_id = artifact_id
            parent_id = state.active_artifact.get("refined_from_artifact_id")
            if not parent_id:
                parent_id = (state.active_artifact.get("parent_category") or {}).get(
                    "source_artifact_id"
                )
            connection.execute(
                "INSERT INTO artifacts(id, conversation_id, artifact_type, payload_json, parent_artifact_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "payload_json=excluded.payload_json, parent_artifact_id=excluded.parent_artifact_id, "
                "updated_at=excluded.updated_at",
                (
                    artifact_id,
                    conversation_id,
                    state.active_artifact_type,
                    _json(state.active_artifact),
                    parent_id,
                    timestamp,
                    timestamp,
                ),
            )
        values = (
            memory.current_paper_id,
            _json(memory.selected_paper_ids),
            _json(memory.generated_cards),
            memory.last_answer,
            _json(memory.last_sources),
            memory.last_export_path,
            state.last_intent,
            state.last_user_query,
            state.rewritten_query,
            state.active_document_id,
            state.active_section,
            artifact_id,
            state.active_artifact_type,
            state.selected_category,
            _json(state.filters),
            state.last_answer_summary,
            timestamp,
            conversation_id,
        )
        connection.execute(
            """
            UPDATE conversation_states SET
                current_paper_id=?, selected_paper_ids_json=?, generated_cards_json=?,
                last_answer=?, last_sources_json=?, last_export_path=?, last_intent=?,
                last_user_query=?, rewritten_query=?, active_document_id=?, active_section=?,
                active_artifact_id=?, active_artifact_type=?, selected_category=?, filters_json=?,
                last_answer_summary=?, updated_at=?
            WHERE conversation_id=?
            """,
            values,
        )

    def list_artifacts(self, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, artifact_type, payload_json, parent_artifact_id, created_at, updated_at "
                "FROM artifacts WHERE conversation_id=? ORDER BY updated_at DESC LIMIT ?",
                (conversation_id, max(1, int(limit))),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "artifact_type": row["artifact_type"],
                "payload": _loads(row["payload_json"], {}),
                "parent_artifact_id": row["parent_artifact_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def clear_conversation(self, conversation_id: str) -> None:
        timestamp = _now()
        with self._connection() as connection:
            connection.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            connection.execute("DELETE FROM artifacts WHERE conversation_id=?", (conversation_id,))
            connection.execute("DELETE FROM conversation_states WHERE conversation_id=?", (conversation_id,))
            connection.execute(
                "INSERT INTO conversation_states(conversation_id, updated_at) VALUES (?, ?)",
                (conversation_id, timestamp),
            )
            connection.execute(
                "UPDATE conversations SET title='新会话', updated_at=? WHERE id=? AND user_id=?",
                (timestamp, conversation_id, self.user_id),
            )

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id=? AND user_id=?",
                (conversation_id, self.user_id),
            )
        return cursor.rowcount > 0
