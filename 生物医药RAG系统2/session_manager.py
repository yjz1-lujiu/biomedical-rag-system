# session_manager.py
"""
会话管理模块
- 支持 SQLite 和 PostgreSQL 存储会话元数据
- 使用 ChromaDB 存储对话历史（支持向量检索）
- 提供上下文窗口压缩（滑动窗口 + 摘要）
- 自动标题生成（轻量级 LLM）
- 可选 Redis 缓存会话列表（连接失败自动降级）
"""

import os
import uuid
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

import chromadb
from chromadb.config import Settings

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory

# ====================== 配置 ======================
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/rag_sessions")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

DEFAULT_SQLITE_PATH = "./sessions.db"
DEFAULT_CHROMA_DIR = "./chat_histories_chroma"


class SessionStore:
    def __init__(self, db_path: str = DEFAULT_SQLITE_PATH):
        self.use_postgres = USE_POSTGRES and PSYCOPG2_AVAILABLE
        if self.use_postgres:
            self.db_url = DATABASE_URL
            self._init_pg()
        else:
            self.db_path = db_path
            self._init_sqlite()

    def _init_sqlite(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

    def _init_pg(self):
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        title TEXT,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP
                    )
                """)
                conn.commit()

    def _get_conn(self):
        if self.use_postgres:
            return psycopg2.connect(self.db_url)
        else:
            return sqlite3.connect(self.db_path)

    def create_session(self, session_id: Optional[str] = None, title: Optional[str] = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        title = title or f"新对话 {datetime.now().strftime('%m-%d %H:%M')}"
        now = datetime.now()
        with self._get_conn() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s)",
                        (session_id, title, now, now)
                    )
                    conn.commit()
            else:
                conn.execute(
                    "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, title, now, now)
                )
        return session_id

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if self.use_postgres:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
                    )
                    rows = cur.fetchall()
                return [dict(r) for r in rows]
            else:
                cursor = conn.execute(
                    "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
                )
                rows = cursor.fetchall()
                return [
                    {"session_id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
                    for r in rows
                ]

    def update_session_title(self, session_id: str, title: str):
        with self._get_conn() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE sessions SET title = %s, updated_at = %s WHERE session_id = %s",
                        (title, datetime.now(), session_id)
                    )
                    conn.commit()
            else:
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title, datetime.now(), session_id)
                )

    def update_session_timestamp(self, session_id: str):
        with self._get_conn() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE sessions SET updated_at = %s WHERE session_id = %s",
                        (datetime.now(), session_id)
                    )
                    conn.commit()
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (datetime.now(), session_id)
                )

    def delete_session(self, session_id: str):
        with self._get_conn() as conn:
            if self.use_postgres:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
                    conn.commit()
            else:
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


class ChromaChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, persist_dir: str = DEFAULT_CHROMA_DIR):
        self.session_id = session_id
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = f"session_{session_id}"
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(self.collection_name)

    @property
    def messages(self) -> List[BaseMessage]:
        try:
            results = self.collection.get()
            if not results['ids']:
                return []
            messages = []
            items = sorted(
                zip(results['ids'], results['documents'], results['metadatas']),
                key=lambda x: x[2].get('timestamp', 0) if x[2] else 0
            )
            for msg_id, content, meta in items:
                if meta and meta.get('type') == 'human':
                    messages.append(HumanMessage(content=content))
                elif meta and meta.get('type') == 'ai':
                    messages.append(AIMessage(content=content))
            return messages
        except:
            return []

    def add_messages(self, messages: List[BaseMessage]):
        for msg in messages:
            msg_type = 'human' if isinstance(msg, HumanMessage) else 'ai'
            self.collection.add(
                documents=[msg.content],
                metadatas=[{"type": msg_type, "timestamp": datetime.now().timestamp()}],
                ids=[str(uuid.uuid4())]
            )

    def clear(self):
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)
        except:
            pass

    def get_messages_for_llm(self, llm, max_token_limit: int = 2000) -> str:
        all_messages = self.messages
        if not all_messages:
            return ""

        total_chars = sum(len(m.content) for m in all_messages)
        if total_chars < max_token_limit * 2:
            return "\n".join([
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                for m in all_messages
            ])

        recent = all_messages[-12:] if len(all_messages) >= 12 else all_messages
        older = all_messages[:-12] if len(all_messages) > 12 else []

        context = ""
        if older:
            older_text = "\n".join([
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                for m in older
            ])
            summary_prompt = f"请将以下对话历史总结为一段简洁的摘要，保留关键信息：\n{older_text}"
            try:
                summary = llm.invoke(summary_prompt)
            except:
                summary = "（对话历史摘要生成失败）"
            context = f"【对话历史摘要】\n{summary}\n\n【最近对话】\n"

        context += "\n".join([
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
            for m in recent
        ])
        return context


class SessionManager:
    def __init__(
            self,
            db_path: str = DEFAULT_SQLITE_PATH,
            chroma_dir: str = DEFAULT_CHROMA_DIR
    ):
        self.store = SessionStore(db_path)
        self.chroma_dir = chroma_dir
        self._current_session_id: Optional[str] = None
        self._current_history: Optional[ChromaChatMessageHistory] = None

        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                r = redis.from_url(REDIS_URL)
                r.ping()
                self.redis_client = r
            except Exception:
                pass

    @property
    def current_session_id(self) -> Optional[str]:
        return self._current_session_id

    def _invalidate_cache(self):
        if self.redis_client:
            try:
                self.redis_client.delete("sessions:list")
            except Exception:
                self.redis_client = None

    def create_session(self, title: Optional[str] = None) -> str:
        session_id = self.store.create_session(title=title)
        self._invalidate_cache()
        return session_id

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        if self.redis_client:
            try:
                cached = self.redis_client.get("sessions:list")
                if cached:
                    return json.loads(cached)
            except Exception:
                self.redis_client = None

        sessions = self.store.get_all_sessions()

        if self.redis_client:
            try:
                self.redis_client.setex("sessions:list", 300, json.dumps(sessions, default=str))
            except Exception:
                self.redis_client = None
        return sessions

    def load_session(self, session_id: str) -> ChromaChatMessageHistory:
        self._current_session_id = session_id
        self._current_history = ChromaChatMessageHistory(session_id, self.chroma_dir)
        return self._current_history

    def get_current_history(self) -> Optional[ChromaChatMessageHistory]:
        return self._current_history

    def switch_session(self, session_id: str) -> ChromaChatMessageHistory:
        return self.load_session(session_id)

    def delete_session(self, session_id: str):
        try:
            client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            client.delete_collection(f"session_{session_id}")
        except:
            pass
        self.store.delete_session(session_id)
        if self._current_session_id == session_id:
            self._current_session_id = None
            self._current_history = None
        self._invalidate_cache()

    def auto_generate_title(self, session_id: str, first_message: str, llm=None):
        if llm is None:
            title = first_message[:20] + ("..." if len(first_message) > 20 else "")
        else:
            prompt = f"""请根据以下用户的第一条消息，生成一个极其简洁的会话标题（不超过10个字）。
消息内容：{first_message}

标题："""
            try:
                title = llm.invoke(prompt).strip()
                if len(title) > 20:
                    title = title[:20] + "..."
            except:
                title = first_message[:20] + ("..." if len(first_message) > 20 else "")
        self.store.update_session_title(session_id, title)
        self._invalidate_cache()

    def update_timestamp(self, session_id: str):
        self.store.update_session_timestamp(session_id)
        self._invalidate_cache()