"""
生物医药RAG智能系统 · 多模态增强版
整合 RAG-Anything 设计理念：多模态解析 + 知识图谱 + 跨模态检索
修复：ID 唯一性、空字符串过滤、文本截断
"""

import os
import re
import json
import base64
import hashlib
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from io import BytesIO

# LangChain 核心
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi

# 外部增强
from Bio import Entrez

# 多模态相关
import networkx as nx
from PIL import Image

# 尝试导入 MinerU（若未安装则降级）
try:
    from mineru import MinerU
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False

# ====================== 环境配置 ======================
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "请替换为你的通义千问APIKey")
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "your_email@example.com")
Entrez.email = PUBMED_EMAIL

embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=DASHSCOPE_API_KEY
)


# ====================== 多模态数据结构 ======================
@dataclass
class ContentBlock:
    """统一的多模态内容块"""
    block_id: str
    block_type: str  # text, image, table, equation
    content: str      # 文本内容 / 图片base64 / 表格JSON / LaTeX
    metadata: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str]] = field(default_factory=list)


# ====================== 多模态文档解析器 ======================
class MultimodalDocumentParser:
    """多模态文档解析器 - 支持 MinerU 和 PyPDF 降级"""

    def __init__(self, use_mineru: bool = True):
        self.use_mineru = use_mineru and MINERU_AVAILABLE
        if self.use_mineru:
            self.parser = MinerU()

    def parse_pdf(self, pdf_path: str) -> List[ContentBlock]:
        # 生成文件唯一标识（短哈希）
        file_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:6]
        if self.use_mineru:
            return self._parse_with_mineru(pdf_path, file_hash)
        else:
            return self._parse_with_pypdf(pdf_path, file_hash)

    def _parse_with_mineru(self, pdf_path: str, file_hash: str) -> List[ContentBlock]:
        blocks = []
        result = self.parser.parse(pdf_path)
        source_name = Path(pdf_path).name

        # 文本块
        for idx, text_block in enumerate(result.get("text_blocks", [])):
            block_id = f"{file_hash}_text_{idx}"
            blocks.append(ContentBlock(
                block_id=block_id,
                block_type="text",
                content=text_block.get("content", ""),
                metadata={
                    "page": text_block.get("page", 0),
                    "source_file": source_name
                }
            ))

        # 图像块
        for idx, img_block in enumerate(result.get("image_blocks", [])):
            block_id = f"{file_hash}_image_{idx}"
            img_data = img_block.get("image_data")
            img_base64 = base64.b64encode(img_data).decode("utf-8") if img_data else ""
            blocks.append(ContentBlock(
                block_id=block_id,
                block_type="image",
                content=img_base64,
                metadata={
                    "page": img_block.get("page", 0),
                    "caption": img_block.get("caption", ""),
                    "source_file": source_name
                }
            ))

        # 表格块
        for idx, table_block in enumerate(result.get("table_blocks", [])):
            block_id = f"{file_hash}_table_{idx}"
            blocks.append(ContentBlock(
                block_id=block_id,
                block_type="table",
                content=json.dumps(table_block.get("table_data", {}), ensure_ascii=False),
                metadata={
                    "page": table_block.get("page", 0),
                    "caption": table_block.get("caption", ""),
                    "source_file": source_name
                }
            ))

        # 公式块
        for idx, eq_block in enumerate(result.get("equation_blocks", [])):
            block_id = f"{file_hash}_equation_{idx}"
            blocks.append(ContentBlock(
                block_id=block_id,
                block_type="equation",
                content=eq_block.get("latex", ""),
                metadata={
                    "page": eq_block.get("page", 0),
                    "source_file": source_name
                }
            ))

        # 建立跨模态关联（需要根据新的 ID 规则调整匹配）
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
                            block_id=block_id,
                            block_type="text",
                            content=para.strip(),
                            metadata={"page": page_num, "source_file": source_name}
                        ))
        return blocks

    def _build_cross_modal_relations(self, blocks: List[ContentBlock], raw_result: Dict, file_hash: str) -> List[ContentBlock]:
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
                    # 使用带前缀的 ID 匹配
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

    def get_related_blocks(self, block_id: str, relation_types: List[str] = None, max_depth: int = 1) -> List[ContentBlock]:
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
            self.collections[name] = Chroma(embedding_function=self.embedding, persist_directory=coll_path)

    def index_blocks(self, blocks: List[ContentBlock], graph: MultimodalKnowledgeGraph):
        self.graph = graph
        texts, metas, ids = [], [], []
        MAX_LEN = 3000  # 约2048 tokens

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
                ids.append(block.block_id)  # 现在 ID 已保证唯一

        if texts:
            # Chroma 的 add_texts 要求 IDs 唯一，我们已经通过前缀保证
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
        for doc in all_docs[:top_k]:
            bid = doc.metadata.get("block_id")
            if not bid:
                bid = doc.page_content[:20]  # 兼容旧数据
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
def get_llm():
    return Tongyi(model_name="qwen-max", dashscope_api_key=DASHSCOPE_API_KEY, temperature=0.3, max_tokens=1500)


# ====================== PubMed 检索 ======================
def search_pubmed_abstracts(query, max_results=3):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        ids = record["IdList"]
        if not ids:
            return ""
        fetch_handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        return fetch_handle.read()
    except Exception as e:
        st.warning(f"PubMed检索不可用: {e}")
        return ""


# ====================== 知识库构建（多模态） ======================
def build_multimodal_knowledge_base(pdf_files):
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    saved_paths = []
    for file in pdf_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(file_path)

    parser = MultimodalDocumentParser()
    vector_store = MultimodalVectorStore()
    vector_store.initialize()
    all_blocks = []
    progress_bar = st.progress(0)
    for i, path in enumerate(saved_paths):
        blocks = parser.parse_pdf(path)
        all_blocks.extend(blocks)
        progress_bar.progress((i + 1) / len(saved_paths))
    graph = MultimodalKnowledgeGraph().build_from_blocks(all_blocks)
    vector_store.index_blocks(all_blocks, graph)
    progress_bar.empty()

    for path in saved_paths:
        os.remove(path)
    os.rmdir(temp_dir)
    return vector_store, graph


# ====================== 场景处理函数（多模态版） ======================
def run_experiment_method_multimodal(query, vector_store, llm):
    retriever = CrossModalRetriever(vector_store)
    blocks = retriever.retrieve(query, top_k=6)
    context = retriever.format_context(blocks)
    prompt = f"""你是生物医药实验专家。基于以下文献内容（含图表信息），针对问题"{query}"给出分步骤实验方法总结。
要求：提取关键步骤、参数、试剂、注意事项；若涉及定量分析给出公式；标注来源。

文献内容：
{context}

请回答："""
    answer = llm.invoke(prompt)
    return answer, blocks


def run_review_outline_multimodal(query, vector_store, llm):
    retriever = CrossModalRetriever(vector_store)
    decomp_prompt = f"将综述主题'{query}'分解为5-6个子主题，每行一个。"
    sub_topics_str = llm.invoke(decomp_prompt)
    sub_topics = [line.strip() for line in sub_topics_str.split('\n') if line.strip()][:6]

    sub_summaries = {}
    all_sources = []
    progress_bar = st.progress(0)
    for i, sub in enumerate(sub_topics):
        blocks = retriever.retrieve(sub, top_k=3)
        all_sources.extend(blocks)
        context = retriever.format_context(blocks)
        summary_prompt = f"基于以下文献片段，总结子主题'{sub}'的研究现状和代表性方法。\n{context}\n总结："
        sub_summaries[sub] = llm.invoke(summary_prompt)
        progress_bar.progress((i + 1) / len(sub_topics))

    synthesis_prompt = f"""你是学术编辑。基于以下子主题总结生成文献综述框架（含引言、正文章节、结论与展望）。
子主题总结：{sub_summaries}
请生成框架："""
    framework = llm.invoke(synthesis_prompt)
    progress_bar.empty()
    unique_files = {b.metadata.get('source_file', 'unknown') for b in all_sources}
    return framework, list(unique_files)[:10]


def run_cross_domain_multimodal(query, vector_store, llm):
    retriever = CrossModalRetriever(vector_store)
    local_blocks = retriever.retrieve("药物递送 纳米载体 响应释放 材料", top_k=5)
    local_context = retriever.format_context(local_blocks)
    external = search_pubmed_abstracts("(materials science OR biomaterials) AND drug delivery", 3)
    prompt = f"""你是跨学科创新专家。基于以下两部分知识，提出至少3个将材料科学方法（如MOF、自修复水凝胶等）应用于药物递送的具体创新方向，并说明可行性与优势。

【本地文献（含图表）】
{local_context[:2000]}

【PubMed前沿】
{external[:1500]}

请回答："""
    answer = llm.invoke(prompt)
    return answer, local_blocks


def run_general_qa_multimodal(query, vector_store, llm):
    retriever = CrossModalRetriever(vector_store)
    blocks = retriever.retrieve(query, top_k=5)
    context = retriever.format_context(blocks)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是生物医药专家助手。基于以下上下文回答问题，若信息不足请说明。\n\n上下文：{context}"),
        ("human", "{input}")
    ])
    full_prompt = prompt.format(context=context, input=query)
    answer = llm.invoke(full_prompt)
    return answer, blocks


# ====================== 纯文本降级模式 ======================
def classify_section(text):
    keywords = ["制备", "合成", "测定", "步骤", "protocol", "离心", "孵育"]
    return "method" if any(kw in text for kw in keywords) else "other"


def build_knowledge_base_legacy(pdf_files):
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    saved_paths = []
    for file in pdf_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(file_path)
    loader = DirectoryLoader(temp_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    for doc in texts:
        doc.metadata["section_type"] = classify_section(doc.page_content)
    db = Chroma.from_documents(texts, embedding, persist_directory="biomedical_rag_db")
    db.persist()
    for path in saved_paths:
        os.remove(path)
    os.rmdir(temp_dir)
    return db


# ====================== Streamlit 主界面 ======================
def main():
    st.set_page_config(page_title="生物医药RAG·多模态版", page_icon="🔬", layout="wide")
    st.title("🔬 生物医药RAG智能系统 · 多模态增强版")
    st.subheader("⚡实验方法 | 📝综述框架 | 🌉跨领域发现 | 🖼️图表解析")
    st.divider()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "legacy_db" not in st.session_state:
        st.session_state.legacy_db = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    with st.sidebar:
        st.header("📤 上传生物医药文献PDF")
        use_multimodal = st.checkbox("启用多模态解析（推荐）", value=True)
        pdf_files = st.file_uploader("支持多PDF", type="pdf", accept_multiple_files=True)

        if st.button("✅ 构建知识库", disabled=not pdf_files):
            with st.spinner("解析中...（多模态约2-8分钟）"):
                try:
                    if use_multimodal:
                        vs, kg = build_multimodal_knowledge_base(pdf_files)
                        st.session_state.vector_store = vs
                        st.session_state.graph = kg
                        st.session_state.legacy_db = None
                        st.success("✅ 多模态知识库构建完成！")
                    else:
                        db = build_knowledge_base_legacy(pdf_files)
                        st.session_state.legacy_db = db
                        st.session_state.vector_store = None
                        st.success("✅ 文本知识库构建完成！")
                except Exception as e:
                    st.error(f"构建失败：{e}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        exp_btn = st.button("🔍 实验方法定位")
    with col2:
        review_btn = st.button("📝 综述框架")
    with col3:
        cross_btn = st.button("🌉 跨领域发现")
    with col4:
        chart_btn = st.button("📊 图表问答")

    user_query = st.text_input("✏️ 输入问题", value=st.session_state.last_query)

    manual_trigger = st.button("🚀 开始分析", type="primary")

    query_to_run = None
    scenario = None

    if exp_btn:
        query_to_run = "请给出脂质体包封率的测定方法（步骤、参数、注意事项）"
        scenario = "exp"
    elif review_btn:
        query_to_run = "刺激响应型纳米药物递送系统"
        scenario = "review"
    elif cross_btn:
        query_to_run = "将材料科学方法创新应用于药物递送系统"
        scenario = "cross"
    elif chart_btn:
        query_to_run = "请描述文献中与药物释放相关的图表内容"
        scenario = "chart"
    elif manual_trigger and user_query.strip():
        query_to_run = user_query.strip()
        if any(kw in query_to_run for kw in ["实验方法", "步骤", "测定"]):
            scenario = "exp"
        elif any(kw in query_to_run for kw in ["综述", "框架"]):
            scenario = "review"
        elif any(kw in query_to_run for kw in ["跨领域", "材料科学"]):
            scenario = "cross"
        elif any(kw in query_to_run for kw in ["图", "表", "图表"]):
            scenario = "chart"
        else:
            scenario = "general"

    if query_to_run and scenario:
        if st.session_state.vector_store is None and st.session_state.legacy_db is None:
            st.error("请先构建知识库！")
        else:
            st.session_state.last_query = query_to_run
            llm = get_llm()
            use_mm = st.session_state.vector_store is not None

            if use_mm:
                vs = st.session_state.vector_store
                if scenario == "exp":
                    with st.spinner("多模态检索实验方法..."):
                        ans, src = run_experiment_method_multimodal(query_to_run, vs, llm)
                        st.markdown("### 📋 实验方法总结")
                        st.write(ans)
                        st.markdown("### 🔗 参考来源")
                        for b in src[:5]:
                            st.caption(f"- {b.metadata.get('source_file', 'unknown')} ({b.block_type})")
                elif scenario == "review":
                    with st.spinner("生成综述框架..."):
                        fw, files = run_review_outline_multimodal(query_to_run, vs, llm)
                        st.markdown("### 📑 综述框架")
                        st.markdown(fw)
                        st.markdown("### 📚 参考文献")
                        for f in files:
                            st.caption(f"- {f}")
                elif scenario in ["cross", "chart"]:
                    with st.spinner("跨领域/图表分析..."):
                        if scenario == "cross":
                            ans, src = run_cross_domain_multimodal(query_to_run, vs, llm)
                        else:
                            ans, src = run_general_qa_multimodal(query_to_run, vs, llm)
                        st.markdown("### 💡 分析结果")
                        st.write(ans)
                        st.markdown("### 📖 来源")
                        for b in src[:5]:
                            st.caption(f"- {b.metadata.get('source_file', 'unknown')} ({b.block_type})")
                else:
                    with st.spinner("通用问答..."):
                        ans, src = run_general_qa_multimodal(query_to_run, vs, llm)
                        st.markdown("### 🤖 回答")
                        st.write(ans)
            else:
                st.warning("当前为纯文本模式，建议启用多模态解析以获得图表分析能力。")

    if st.session_state.vector_store is None and st.session_state.legacy_db is None:
        st.info("👆 请上传PDF并点击「构建知识库」开始。")


if __name__ == "__main__":
    main()