"""
生物医药RAG智能系统 · 多模态增强版 + DeepSeek风格会话管理
- 多模态文档解析（文本/图像/表格/公式）
- 知识图谱 + 跨模态检索
- 会话管理：列表、新建、删除、自动标题、上下文窗口压缩
- 存储：SQLite/PostgreSQL + ChromaDB，可选 Redis 缓存
"""

import os
import re
import json
import base64
import hashlib
import uuid
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from session_manager import SessionManager, ChromaChatMessageHistory

# LangChain 核心
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import Tongyi
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory

# 多智能体系统
from multi_agent_system import run_agent_workflow
from audit_logger import AuditLogger

# 外部增强
from Bio import Entrez

# 多模态相关
import networkx as nx

# 尝试导入 MinerU（若未安装则降级）
try:
    from mineru import MinerU

    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False

# 尝试导入 PostgreSQL 和 Redis（可选）
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

# 导入mem0
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

# ====================== 环境配置 ======================
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "请替换为你的通义千问APIKey")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "your_email@example.com")
Entrez.email = PUBMED_EMAIL

# 数据库配置（可通过环境变量切换）
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/rag_sessions")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=DASHSCOPE_API_KEY
)


# ====================== 存储层：会话元数据 ======================
class SessionStore:
    """会话元数据存储，支持 SQLite 和 PostgreSQL"""

    def __init__(self, db_path: str = "./sessions.db"):
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


# ====================== 对话历史存储（ChromaDB） ======================
class ChromaChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, persist_dir: str = "./chat_histories_chroma"):
        self.session_id = session_id
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        import chromadb
        from chromadb.config import Settings
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
        """返回压缩后的上下文字符串（滑动窗口 + 摘要）"""
        all_messages = self.messages
        if not all_messages:
            return ""

        # 粗略估算：1 token ≈ 2 字符
        total_chars = sum(len(m.content) for m in all_messages)
        if total_chars < max_token_limit * 2:
            return "\n".join(
                [f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}" for m in all_messages])

        # 保留最后 6 轮，其余生成摘要
        recent = all_messages[-6:]
        older = all_messages[:-6]
        if older:
            older_text = "\n".join([f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}" for m in older])
            summary_prompt = f"请将以下对话历史总结为一段简洁的摘要，保留关键信息：\n{older_text}"
            try:
                summary = llm.invoke(summary_prompt)
            except:
                summary = "（对话历史摘要生成失败）"
            context = f"【对话历史摘要】\n{summary}\n\n【最近对话】\n"
        else:
            context = ""

        context += "\n".join([f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}" for m in recent])
        return context


# ====================== 会话管理器 ======================
class SessionManager:
    def __init__(self, db_path: str = "./sessions.db", chroma_dir: str = "./chat_histories_chroma"):
        self.store = SessionStore(db_path)
        self.chroma_dir = chroma_dir
        self._current_session_id: Optional[str] = None
        self._current_history: Optional[ChromaChatMessageHistory] = None

        # Redis 缓存（可选）
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(REDIS_URL)
            except:
                pass

        # mem0 记忆管理（可选）
        self.memory = None
        if MEM0_AVAILABLE and MEM0_API_KEY:
            try:
                self.memory = Memory(api_key=MEM0_API_KEY)
            except:
                pass

    @property
    def current_session_id(self) -> Optional[str]:
        return self._current_session_id

    def _invalidate_cache(self):
        if self.redis_client:
            self.redis_client.delete("sessions:list")

    def create_session(self, title: Optional[str] = None) -> str:
        session_id = self.store.create_session(title=title)
        self._invalidate_cache()
        return session_id

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        if self.redis_client:
            cached = self.redis_client.get("sessions:list")
            if cached:
                return json.loads(cached)
        sessions = self.store.get_all_sessions()
        if self.redis_client:
            self.redis_client.setex("sessions:list", 300, json.dumps(sessions, default=str))
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
        # 删除 ChromaDB 数据
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            client.delete_collection(f"session_{session_id}")
        except:
            pass
        # 删除元数据
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


# ====================== 多模态解析（保留原有） ======================
@dataclass
class ContentBlock:
    block_id: str
    block_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str]] = field(default_factory=list)


class MultimodalDocumentParser:
    def __init__(self, use_mineru: bool = True):
        self.use_mineru = use_mineru and MINERU_AVAILABLE
        if self.use_mineru:
            self.parser = MinerU()

    def parse_pdf(self, pdf_path: str) -> List[ContentBlock]:
        file_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:6]
        if self.use_mineru:
            return self._parse_with_mineru(pdf_path, file_hash)
        else:
            return self._parse_with_pypdf(pdf_path, file_hash)

    def _parse_with_mineru(self, pdf_path: str, file_hash: str) -> List[ContentBlock]:
        blocks = []
        result = self.parser.parse(pdf_path)
        source_name = Path(pdf_path).name

        for idx, text_block in enumerate(result.get("text_blocks", [])):
            block_id = f"{file_hash}_text_{idx}"
            blocks.append(ContentBlock(
                block_id=block_id, block_type="text",
                content=text_block.get("content", ""),
                metadata={"page": text_block.get("page", 0), "source_file": source_name}
            ))

        for idx, img_block in enumerate(result.get("image_blocks", [])):
            block_id = f"{file_hash}_image_{idx}"
            img_data = img_block.get("image_data")
            img_base64 = base64.b64encode(img_data).decode("utf-8") if img_data else ""
            blocks.append(ContentBlock(
                block_id=block_id, block_type="image", content=img_base64,
                metadata={"page": img_block.get("page", 0), "caption": img_block.get("caption", ""),
                          "source_file": source_name}
            ))

        for idx, table_block in enumerate(result.get("table_blocks", [])):
            block_id = f"{file_hash}_table_{idx}"
            blocks.append(ContentBlock(
                block_id=block_id, block_type="table",
                content=json.dumps(table_block.get("table_data", {}), ensure_ascii=False),
                metadata={"page": table_block.get("page", 0), "caption": table_block.get("caption", ""),
                          "source_file": source_name}
            ))

        for idx, eq_block in enumerate(result.get("equation_blocks", [])):
            block_id = f"{file_hash}_equation_{idx}"
            blocks.append(ContentBlock(
                block_id=block_id, block_type="equation",
                content=eq_block.get("latex", ""),
                metadata={"page": eq_block.get("page", 0), "source_file": source_name}
            ))

        blocks = self._build_cross_modal_relations(blocks, result, file_hash)
        return blocks

    def _parse_with_pypdf(self, pdf_path: str, file_hash: str) -> List[ContentBlock]:
        from pypdf import PdfReader
        blocks = []
        reader = PdfReader(pdf_path)
        source_name = Path(pdf_path).name
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                for idx, para in enumerate(text.split("\n\n")):
                    if para.strip():
                        block_id = f"{file_hash}_text_{page_num}_{idx}"
                        blocks.append(ContentBlock(
                            block_id=block_id, block_type="text", content=para.strip(),
                            metadata={"page": page_num, "source_file": source_name}
                        ))
        return blocks

    def _build_cross_modal_relations(self, blocks: List[ContentBlock], raw_result: Dict, file_hash: str) -> List[
        ContentBlock]:
        block_index = {b.block_id: b for b in blocks}
        patterns = [
            (r"如图\s*(\d+)", "image"), (r"Figure\s*(\d+)", "image"),
            (r"表\s*(\d+)", "table"), (r"Table\s*(\d+)", "table"),
            (r"公式\s*[\(（]?\s*(\d+)\s*[\)）]?", "equation"), (r"Equation\s*[\(]?\s*(\d+)\s*[\)]?", "equation"),
        ]
        for block in blocks:
            if block.block_type != "text":
                continue
            for pattern, target_type in patterns:
                for match in re.findall(pattern, block.content, re.IGNORECASE):
                    target_id = f"{file_hash}_{target_type}_{match}"
                    if target_id in block_index:
                        block.relations.append(("REFERENCES", target_id))
                        block_index[target_id].relations.append(("REFERENCED_BY", block.block_id))
        return blocks


# ====================== 多模态知识图谱 ======================
class MultimodalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.block_index: Dict[str, ContentBlock] = {}

    def build_from_blocks(self, blocks: List[ContentBlock]) -> "MultimodalKnowledgeGraph":
        self.graph.clear()
        self.block_index.clear()
        for block in blocks:
            self.graph.add_node(block.block_id, block_type=block.block_type, metadata=block.metadata)
            self.block_index[block.block_id] = block
        for block in blocks:
            for rel_type, target_id in block.relations:
                if target_id in self.block_index:
                    self.graph.add_edge(block.block_id, target_id, relation=rel_type)
        return self

    def get_related_blocks(self, block_id: str, relation_types: List[str] = None, max_depth: int = 1) -> List[
        ContentBlock]:
        if block_id not in self.graph:
            return []
        related_ids = set()
        current = {block_id}
        for _ in range(max_depth):
            nxt = set()
            for node in current:
                for neighbor in self.graph.neighbors(node):
                    edge = self.graph.get_edge_data(node, neighbor)
                    rel = edge.get("relation", "")
                    if relation_types is None or rel in relation_types:
                        if neighbor not in related_ids:
                            related_ids.add(neighbor)
                            nxt.add(neighbor)
            current = nxt
        return [self.block_index[rid] for rid in related_ids if rid in self.block_index]

    def export_graph_data(self) -> Dict:
        nodes = [{"id": nid, "type": data.get("block_type"), "metadata": data.get("metadata", {})}
                 for nid, data in self.graph.nodes(data=True)]
        edges = [{"source": u, "target": v, "relation": d.get("relation", "")}
                 for u, v, d in self.graph.edges(data=True)]
        return {"nodes": nodes, "edges": edges}


# ====================== 多模态向量存储 ======================
class MultimodalVectorStore:
    def __init__(self, persist_dir: str = "multimodal_rag_db"):
        self.persist_dir = persist_dir
        self.embedding = embedding
        self.collections = {"text": None, "image": None, "table": None, "equation": None}
        self.graph: Optional[MultimodalKnowledgeGraph] = None

    def initialize(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        for name in self.collections:
            coll_path = os.path.join(self.persist_dir, name)
            try:
                self.collections[name] = Chroma(embedding_function=self.embedding, persist_directory=coll_path)
            except Exception as e:
                # 捕获ChromaDB的错误
                error_str = str(e)
                if "'RustBindingsAPI' object has no attribute 'bindings'" in error_str or "KeyError" in error_str:
                    # 忽略这些错误，继续运行
                    pass
                else:
                    # 其他错误仍然抛出
                    raise

    def index_blocks(self, blocks: List[ContentBlock], graph: MultimodalKnowledgeGraph):
        self.graph = graph
        texts, metas, ids = [], [], []
        MAX_LEN = 3000
        for block in blocks:
            text_to_embed = None
            if block.block_type == "text":
                text_to_embed = block.content
            elif block.block_type == "image":
                text_to_embed = block.metadata.get("caption", "")
            elif block.block_type == "table":
                text_to_embed = block.metadata.get("caption", "")
                if not text_to_embed and block.content:
                    try:
                        table_data = json.loads(block.content)
                        rows = table_data.get("rows", [])[:3]
                        text_to_embed = "表格内容：" + str(rows)
                    except:
                        text_to_embed = ""
            elif block.block_type == "equation":
                text_to_embed = block.content

            if text_to_embed and text_to_embed.strip():
                if len(text_to_embed) > MAX_LEN:
                    text_to_embed = text_to_embed[:MAX_LEN]
                texts.append(text_to_embed)
                meta = block.metadata.copy()
                meta["block_id"] = block.block_id
                meta["block_type"] = block.block_type
                metas.append(meta)
                ids.append(block.block_id)

        if texts:
            self.collections["text"].add_texts(texts=texts, metadatas=metas, ids=ids)
        self._save_graph()

    def _save_graph(self):
        if self.graph:
            path = os.path.join(self.persist_dir, "knowledge_graph.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.graph.export_graph_data(), f, ensure_ascii=False, indent=2)


# ====================== 跨模态检索器 ======================
class CrossModalRetriever:
    def __init__(self, vector_store: MultimodalVectorStore):
        self.vector_store = vector_store
        self.graph = vector_store.graph

    def retrieve(self, query: str, top_k: int = 5) -> List[ContentBlock]:
        all_docs = []
        for coll in self.vector_store.collections.values():
            if coll:
                all_docs.extend(coll.similarity_search(query, k=top_k))
        all_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)

        retrieved = []
        seen = set()
        
        # 检查graph是否存在
        if self.graph and hasattr(self.graph, 'block_index'):
            for doc in all_docs[:top_k]:
                bid = doc.metadata.get("block_id")
                if not bid:
                    bid = doc.page_content[:20]
                if bid in self.graph.block_index:
                    block = self.graph.block_index[bid]
                    retrieved.append(block)
                    seen.add(bid)

            expanded = []
            for block in retrieved:
                related = self.graph.get_related_blocks(block.block_id, ["REFERENCES", "RELATED_TO"], max_depth=1)
                for rel in related:
                    if rel.block_id not in seen:
                        expanded.append(rel)
                        seen.add(rel.block_id)
            return retrieved + expanded
        else:
            # 如果graph不存在，返回空列表
            return []

    def format_context(self, blocks: List[ContentBlock]) -> str:
        parts = []
        for b in blocks:
            if b.block_type == "text":
                parts.append(f"[文本] {b.content}")
            elif b.block_type == "image":
                parts.append(f"[图片] 图注：{b.metadata.get('caption', '')}")
            elif b.block_type == "table":
                parts.append(f"[表格] 表注：{b.metadata.get('caption', '')}\n数据：{b.content[:300]}")
            elif b.block_type == "equation":
                parts.append(f"[公式] {b.content}")
        return "\n\n---\n\n".join(parts)


# ====================== 大模型 ======================
def get_llm(temperature=0.3, max_tokens=1500):
    return Tongyi(
        model_name="qwen-max",
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens
    )


def get_title_llm():
    return Tongyi(
        model_name="qwen-turbo",
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=0,
        max_tokens=20
    )


# ====================== PubMed 检索 ======================
def search_pubmed_abstracts(query, max_results=3):
    # ... 保持原有 ...
    pass


# ====================== 知识库构建 ======================
def build_multimodal_knowledge_base(pdf_files):
    vector_store = MultimodalVectorStore()
    vector_store.initialize()
    
    parser = MultimodalDocumentParser()
    all_blocks = []
    
    for pdf_file in pdf_files:
        # 保存上传的文件到临时目录
        temp_pdf_path = f"./temp_{uuid.uuid4()}.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # 解析PDF
        blocks = parser.parse_pdf(temp_pdf_path)
        all_blocks.extend(blocks)
        
        # 删除临时文件
        os.remove(temp_pdf_path)
    
    # 构建知识图谱
    graph = MultimodalKnowledgeGraph().build_from_blocks(all_blocks)
    
    # 索引到向量存储
    vector_store.index_blocks(all_blocks, graph)
    
    return vector_store


# ====================== RAG 回答生成（融合记忆） ======================
def generate_response_with_memory(user_query: str, vector_store, llm, history: ChromaChatMessageHistory, session_manager=None, use_multi_agent=False):
    if use_multi_agent:
        # 使用多智能体系统
        session_id = history.session_id if hasattr(history, 'session_id') else str(uuid.uuid4())
        result = run_agent_workflow(user_query, session_id, vector_store)
        
        # 构建流式回答
        def stream_answer():
            full_response = result.get("fact_checked_answer", {}).get("final_answer", "")
            # 模拟流式输出
            for i in range(0, len(full_response), 50):
                yield full_response[i:i+50]
            
            # 存储回答到mem0（如果可用）
            if session_manager and session_manager.memory:
                try:
                    session_manager.memory.add(full_response)
                except:
                    pass
        
        return stream_answer, []
    else:
        # 传统RAG流程
        # 1. 获取压缩后的历史上下文
        history_context = history.get_messages_for_llm(llm) if history else ""

        # 2. 多模态检索
        retriever = CrossModalRetriever(vector_store)
        blocks = retriever.retrieve(user_query, top_k=5)
        knowledge_context = retriever.format_context(blocks)

        # 3. 使用mem0获取相关记忆（如果可用）
        memory_context = ""
        if session_manager and session_manager.memory:
            try:
                # 检索与当前查询相关的记忆
                memories = session_manager.memory.get(user_query)
                if memories:
                    memory_context = "\n".join([f"- {m['content']}" for m in memories])
            except:
                pass

        # 4. 构建提示词
        prompt = f"""
你是一个生物医药专家助手。
【历史对话上下文】
{history_context}

【文献知识上下文】
{knowledge_context}

【相关记忆】
{memory_context if memory_context else "无"}

【当前问题】
{user_query}

请结合历史对话的上下文（如果有关联）、文献知识和相关记忆，给出专业、准确的回答。"""

        # 5. 流式生成回答
        def stream_answer():
            full_response = ""
            for chunk in llm.stream(prompt):
                full_response += chunk
                yield chunk
            
            # 6. 存储回答到mem0（如果可用）
            if session_manager and session_manager.memory:
                try:
                    session_manager.memory.add(full_response)
                except:
                    pass
        
        return stream_answer, blocks


# ====================== Streamlit 主界面 ======================
def main():
    st.set_page_config(page_title="生物医药系统", page_icon="�", layout="wide")
    st.title("智能生物医药系统")

    # 初始化会话状态
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages_ui" not in st.session_state:
        st.session_state.messages_ui = []
    if "vector_store" not in st.session_state:
        # 检查是否存在已构建的知识库
        if os.path.exists("multimodal_rag_db") and os.path.isdir("multimodal_rag_db"):
            # 加载已存在的知识库
            vector_store = MultimodalVectorStore()
            vector_store.initialize()
            st.session_state.vector_store = vector_store
        else:
            st.session_state.vector_store = None
    if "legacy_db" not in st.session_state:
        st.session_state.legacy_db = None

    # 侧边栏
    with st.sidebar:
        st.header("会话管理")
        if st.button("新建对话", use_container_width=True):
            new_id = st.session_state.session_manager.create_session()
            st.session_state.current_session_id = new_id
            st.session_state.messages_ui = []
            st.rerun()

        st.divider()
        sessions = st.session_state.session_manager.get_all_sessions()
        for sess in sessions:
            sid = sess['session_id']
            title = sess['title']
            updated = sess['updated_at'][:16] if sess['updated_at'] else ""
            col1, col2 = st.columns([4, 1])
            with col1:
                btn_type = "primary" if sid == st.session_state.current_session_id else "secondary"
                if st.button(f"{title}\n{updated}", key=f"btn_{sid}", use_container_width=True, type=btn_type):
                    st.session_state.current_session_id = sid
                    history = st.session_state.session_manager.load_session(sid)
                    st.session_state.messages_ui = [
                        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                        for m in history.messages
                    ]
                    st.rerun()
            with col2:
                if st.button("删除", key=f"del_{sid}"):
                    st.session_state.session_manager.delete_session(sid)
                    if st.session_state.current_session_id == sid:
                        st.session_state.current_session_id = None
                        st.session_state.messages_ui = []
                    st.rerun()

        st.divider()
        st.header("系统配置")
        use_multi_agent = st.checkbox("启用多智能体系统", value=False)
        st.session_state.use_multi_agent = use_multi_agent

        st.divider()
        st.header("上传文献")
        pdf_files = st.file_uploader("选择PDF文件", type="pdf", accept_multiple_files=True)
        if st.button("构建知识库", disabled=not pdf_files):
            with st.spinner("解析中..."):
                # 调用知识库构建函数
                st.session_state.vector_store = build_multimodal_knowledge_base(pdf_files)
                st.success("知识库构建完成")

    # 主界面：历史消息渲染
    for msg in st.session_state.messages_ui:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    user_query = st.chat_input("输入你的问题...")
    if user_query:
        # 确保有当前会话
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = st.session_state.session_manager.create_session()

        # 获取当前历史
        history = st.session_state.session_manager.load_session(st.session_state.current_session_id)

        # 自动生成标题（第一条消息）
        if len(history.messages) == 0:
            title_llm = get_title_llm()
            st.session_state.session_manager.auto_generate_title(
                st.session_state.current_session_id, user_query, llm=title_llm
            )

        # 显示用户消息
        st.session_state.messages_ui.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                llm = get_llm()
                if st.session_state.vector_store is not None:
                    use_multi_agent = st.session_state.get("use_multi_agent", False)
                    stream_func, _ = generate_response_with_memory(
                        user_query, st.session_state.vector_store, llm, history, 
                        st.session_state.session_manager, use_multi_agent=use_multi_agent
                    )
                    # 使用流式输出
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in stream_func():
                        full_response += chunk
                        response_placeholder.markdown(full_response)
                else:
                    full_response = "请先在侧边栏上传PDF并构建知识库。"
                    st.markdown(full_response)

        # 保存交互
        history.add_messages([HumanMessage(content=user_query), AIMessage(content=full_response)])
        st.session_state.messages_ui.append({"role": "assistant", "content": full_response})
        st.session_state.session_manager.update_timestamp(st.session_state.current_session_id)
        st.rerun()

    if st.session_state.vector_store is None:
        st.info("👆 请先在侧边栏上传PDF并构建知识库。")


if __name__ == "__main__":
    main()