"""
database.py - SQLite Async Persistence Layer for Chainlit Multi-Session History
================================================================================

Provides production SQLite persistence for Chainlit chat sessions using `aiosqlite`:
- Stores users, threads, steps (messages/tools), elements, and feedback.
- Enables the Chainlit left-sidebar Chat History with multi-session switching.
- Allows resuming prior threads with state restored into in-memory multi-turn context.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite
import chainlit as cl
import chainlit.data as cl_data
from chainlit.types import PaginatedResponse, Pagination, ThreadDict, ThreadFilter

logger = logging.getLogger("sqlite_data_layer")

DB_PATH = os.getenv("CHAT_DB_PATH", "chat_history.db")


class SQLiteDataLayer(cl_data.BaseDataLayer):
    """
    Asynchronous SQLite Data Layer implementing Chainlit's BaseDataLayer.
    Stores thread conversations and enables UI multi-session restoration.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    async def _get_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        return db

    async def setup(self) -> None:
        """Initialize database tables on application startup."""
        async with await self._get_db() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    identifier TEXT PRIMARY KEY,
                    createdAt TEXT,
                    metadata TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    createdAt TEXT,
                    name TEXT,
                    userId TEXT,
                    userIdentifier TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY,
                    threadId TEXT,
                    name TEXT,
                    type TEXT,
                    subtype TEXT,
                    payload TEXT,
                    createdAt TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS elements (
                    id TEXT PRIMARY KEY,
                    threadId TEXT,
                    type TEXT,
                    name TEXT,
                    url TEXT,
                    chainlitKey TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    threadId TEXT,
                    value INTEGER,
                    comment TEXT
                )
            """)
            await db.commit()
            logger.info("SQLite Chat Database initialized at: %s", self.db_path)

    # --------------------------------------------------------------------------
    # User Management
    # --------------------------------------------------------------------------
    async def get_user(self, identifier: str) -> Optional[cl.PersistedUser]:
        async with await self._get_db() as db:
            cursor = await db.execute("SELECT * FROM users WHERE identifier = ?", (identifier,))
            row = await cursor.fetchone()
            if row:
                return cl.PersistedUser(
                    id=row["identifier"],
                    identifier=row["identifier"],
                    createdAt=row["createdAt"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
        return None

    async def create_user(self, user: cl.User) -> cl.PersistedUser:
        now = datetime.now(timezone.utc).isoformat()
        async with await self._get_db() as db:
            await db.execute(
                "INSERT OR REPLACE INTO users (identifier, createdAt, metadata) VALUES (?, ?, ?)",
                (user.identifier, now, json.dumps(user.metadata or {})),
            )
            await db.commit()
        return cl.PersistedUser(
            id=user.identifier,
            identifier=user.identifier,
            createdAt=now,
            metadata=user.metadata or {},
        )

    # --------------------------------------------------------------------------
    # Thread (Conversation Session) Management
    # --------------------------------------------------------------------------
    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        async with await self._get_db() as db:
            cursor = await db.execute("SELECT * FROM threads WHERE id = ?", (thread_id,))
            thread_row = await cursor.fetchone()
            if not thread_row:
                return None

            steps_cursor = await db.execute(
                "SELECT payload FROM steps WHERE threadId = ? ORDER BY createdAt ASC",
                (thread_id,),
            )
            step_rows = await steps_cursor.fetchall()
            steps = [json.loads(row["payload"]) for row in step_rows]

            return {
                "id": thread_row["id"],
                "createdAt": thread_row["createdAt"],
                "name": thread_row["name"] or "Chat Session",
                "userId": thread_row["userId"],
                "userIdentifier": thread_row["userIdentifier"],
                "tags": json.loads(thread_row["tags"] or "[]"),
                "metadata": json.loads(thread_row["metadata"] or "{}"),
                "steps": steps,
            }

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        async with await self._get_db() as db:
            cursor = await db.execute("SELECT id FROM threads WHERE id = ?", (thread_id,))
            exists = await cursor.fetchone()
            now = datetime.now(timezone.utc).isoformat()

            if not exists:
                await db.execute(
                    """INSERT INTO threads (id, createdAt, name, userId, userIdentifier, tags, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        thread_id,
                        now,
                        name or "New Conversation",
                        user_id or "default",
                        user_id or "default",
                        json.dumps(tags or []),
                        json.dumps(metadata or {}),
                    ),
                )
            else:
                await db.execute(
                    """UPDATE threads SET
                       name = COALESCE(?, name),
                       userId = COALESCE(?, userId),
                       userIdentifier = COALESCE(?, userIdentifier),
                       tags = COALESCE(?, tags),
                       metadata = COALESCE(?, metadata)
                       WHERE id = ?""",
                    (
                        name,
                        user_id,
                        user_id,
                        json.dumps(tags) if tags is not None else None,
                        json.dumps(metadata) if metadata is not None else None,
                        thread_id,
                    ),
                )
            await db.commit()

    async def delete_thread(self, thread_id: str) -> None:
        async with await self._get_db() as db:
            await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            await db.execute("DELETE FROM steps WHERE threadId = ?", (thread_id,))
            await db.execute("DELETE FROM feedback WHERE threadId = ?", (thread_id,))
            await db.execute("DELETE FROM elements WHERE threadId = ?", (thread_id,))
            await db.commit()

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        limit = pagination.first or 40
        async with await self._get_db() as db:
            query = "SELECT * FROM threads"
            params: List[Any] = []
            if filters and filters.userId:
                query += " WHERE userId = ? OR userIdentifier = ?"
                params.extend([filters.userId, filters.userId])
            query += " ORDER BY createdAt DESC LIMIT ?"
            params.append(limit)

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            threads_data: List[ThreadDict] = []
            for row in rows:
                threads_data.append({
                    "id": row["id"],
                    "createdAt": row["createdAt"],
                    "name": row["name"] or "Chat Session",
                    "userId": row["userId"],
                    "userIdentifier": row["userIdentifier"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "metadata": json.loads(row["metadata"] or "{}"),
                })

            return PaginatedResponse(
                data=threads_data,
                pageInfo={
                    "hasNextPage": False,
                    "startCursor": None,
                    "endCursor": None,
                },
            )

    # --------------------------------------------------------------------------
    # Step & Message Tracking
    # --------------------------------------------------------------------------
    async def create_step(self, step_dict: dict) -> None:
        async with await self._get_db() as db:
            created_at = step_dict.get("createdAt") or datetime.now(timezone.utc).isoformat()
            await db.execute(
                """INSERT OR REPLACE INTO steps (id, threadId, name, type, subtype, payload, createdAt)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    step_dict.get("id"),
                    step_dict.get("threadId"),
                    step_dict.get("name"),
                    step_dict.get("type"),
                    step_dict.get("subtype"),
                    json.dumps(step_dict),
                    created_at,
                ),
            )
            await db.commit()

    async def update_step(self, step_dict: dict) -> None:
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str) -> None:
        async with await self._get_db() as db:
            await db.execute("DELETE FROM steps WHERE id = ?", (step_id,))
            await db.commit()

    # --------------------------------------------------------------------------
    # Elements & Feedback
    # --------------------------------------------------------------------------
    async def get_element(self, thread_id: str, element_id: str) -> Optional[dict]:
        async with await self._get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM elements WHERE threadId = ? AND id = ?",
                (thread_id, element_id),
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "threadId": row["threadId"],
                    "type": row["type"],
                    "name": row["name"],
                    "url": row["url"],
                    "chainlitKey": row["chainlitKey"],
                }
        return None

    async def create_element(self, element: cl.Element) -> None:
        async with await self._get_db() as db:
            await db.execute(
                """INSERT OR REPLACE INTO elements (id, threadId, type, name, url, chainlitKey)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    getattr(element, "id", None) or getattr(element, "name", ""),
                    getattr(element, "thread_id", ""),
                    getattr(element, "type", "text"),
                    getattr(element, "name", ""),
                    getattr(element, "url", ""),
                    getattr(element, "chainlit_key", ""),
                ),
            )
            await db.commit()

    async def delete_element(self, element_id: str) -> None:
        async with await self._get_db() as db:
            await db.execute("DELETE FROM elements WHERE id = ?", (element_id,))
            await db.commit()

    async def upsert_feedback(self, feedback: cl_data.Feedback) -> str:
        async with await self._get_db() as db:
            await db.execute(
                "INSERT OR REPLACE INTO feedback (id, threadId, value, comment) VALUES (?, ?, ?, ?)",
                (feedback.id, feedback.threadId, feedback.value, feedback.comment),
            )
            await db.commit()
        return feedback.id

    async def delete_feedback(self, feedback_id: str) -> bool:
        async with await self._get_db() as db:
            await db.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
            await db.commit()
        return True

    async def close(self) -> None:
        pass


# Global data layer instance
sqlite_data_layer = SQLiteDataLayer()
