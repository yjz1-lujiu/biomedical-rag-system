# main.py
import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
from Bio import Entrez

# ====================== 环境配置 ======================
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "请替换为你的通义千问APIKey")
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "your_email@example.com")
Entrez.email = PUBMED_EMAIL

embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# ====================== 辅助函数 ======================
def classify_section(text):
    method_keywords = [
        "制备", "合成", "测定", "检测", "步骤", "protocol", "包封率", "粒径", "表征",
        "色谱条件", "离心", "孵育", "方法", "实验", "操作", "加入", "混合", "反应",
        "溶液", "浓度", "温度", "时间", "pH", "透析", "超声", "过滤", "纯化"
    ]
    if any(kw in text for kw in method_keywords):
        return "method"
    elif re.search(r"图\d+|Figure|结果|讨论|结论|引言|背景", text):
        return "other"
    else:
        return "other"

def build_knowledge_base(pdf_files):
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["。", "！", "？", "\n", "；", "."]
    )
    texts = text_splitter.split_documents(documents)
    for doc in texts:
        doc.metadata["section_type"] = classify_section(doc.page_content)

    db = Chroma.from_documents(texts, embedding, persist_directory="biomedical_rag_db")
    db.persist()

    for path in saved_paths:
        os.remove(path)
    os.rmdir(temp_dir)
    return db

def get_llm():
    return Tongyi(
        model_name="qwen-max",
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=0.3,
        max_tokens=1500
    )

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
        st.warning(f"PubMed检索暂时不可用: {e}")
        return ""

# ====================== 核心场景处理 ======================
def run_experiment_method(query, db, llm):
    retriever = db.as_retriever(search_kwargs={"k": 5, "filter": {"section_type": "method"}})
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""你是一位生物医药实验专家。请基于以下文献中的实验方法片段，针对问题"{query}"给出清晰、分步骤的实验方法总结。
要求：
1. 提取关键步骤、参数、试剂和注意事项。
2. 如果涉及定量分析，说明计算公式或标准。
3. 注明信息来源（文献文件名）。

文献内容：
{context}

请回答："""
    answer = llm.invoke(prompt)   # ✅ 修复：使用 invoke
    return answer, docs

def run_review_outline(query, db, llm):
    decomp_prompt = f"""你是生物医药领域专家。请将综述主题"{query}"分解为5-6个关键子主题或章节，用于构建文献综述框架。
只输出子主题列表，每行一个，不要编号。"""
    sub_topics_str = llm.invoke(decomp_prompt)   # ✅ 修复
    sub_topics = [line.strip() for line in sub_topics_str.split('\n') if line.strip()][:6]

    retriever = db.as_retriever(search_kwargs={"k": 3})
    sub_summaries = {}
    all_sources = []
    progress_bar = st.progress(0)
    for i, sub in enumerate(sub_topics):
        docs = retriever.invoke(sub)
        all_sources.extend(docs)
        context = "\n\n".join([d.page_content for d in docs])
        summary_prompt = f"""基于以下文献片段，用一段话总结子主题"{sub}"的研究现状、主要发现和代表性方法。
文献内容：
{context}

总结："""
        summary = llm.invoke(summary_prompt)   # ✅ 修复
        sub_summaries[sub] = summary
        progress_bar.progress((i + 1) / len(sub_topics))

    synthesis_prompt = f"""你是一位资深学术编辑。请根据以下各子主题的总结，生成一个完整的文献综述框架。
框架应包含：
I. 引言（研究背景与意义）
II. 正文各章节（请为每个子主题拟一个专业标题，并列出2-3个核心要点）
III. 结论与未来展望

子主题总结：
{sub_summaries}

请生成综述框架："""
    framework = llm.invoke(synthesis_prompt)   # ✅ 修复
    progress_bar.empty()
    unique_sources = {doc.metadata.get('source', 'unknown') for doc in all_sources}
    return framework, list(unique_sources)[:10]

def run_cross_domain(query, db, llm):
    retriever_local = db.as_retriever(search_kwargs={"k": 4})
    local_docs = retriever_local.invoke("药物递送 纳米载体 响应释放 材料")
    pubmed_query = "(materials science OR biomaterials) AND drug delivery AND (innovation OR novel approach)"
    external_knowledge = search_pubmed_abstracts(pubmed_query, max_results=3)
    local_context = "\n\n".join([d.page_content for d in local_docs])
    prompt = f"""你是一位跨学科创新专家，擅长将材料科学方法应用于生物医药领域。
请基于以下两部分知识，提出至少3个具体的创新方向，将材料科学中的新方法（如MOF、自修复水凝胶、仿生材料、智能响应材料等）应用于药物递送系统。
要求：
1. 每个方向说明：材料科学方法名称、在药物递送中的具体应用设想、潜在优势。
2. 给出简要可行性分析。

【本地文献内容（药物递送）】
{local_context[:2000]}

【外部材料科学前沿摘要】
{external_knowledge[:1500]}

请回答："""
    answer = llm.invoke(prompt)   # ✅ 修复
    sources = [doc.metadata.get('source', 'unknown') for doc in local_docs]
    return answer, sources

def run_general_qa(query, db, llm):
    retriever = db.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个生物医药专家助手。请根据以下上下文回答问题，如果上下文没有相关信息，请如实告知。\n\n上下文：{context}"),
        ("human", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    result = rag_chain.invoke({"input": query})
    return result["answer"], result["context"]

# ====================== 界面展示 ======================
def display_experiment_result(answer, sources):
    st.success("实验方法定位完成")
    st.markdown("### 📋 实验方法总结")
    st.write(answer)
    st.markdown("### 🔗 参考来源片段")
    for i, doc in enumerate(sources[:5], 1):
        src = doc.metadata.get('source', 'unknown').split('/')[-1]
        st.caption(f"{i}. {src}")
        st.text(doc.page_content[:300] + "...")

def display_review_result(framework, source_files):
    st.success("综述框架生成完毕")
    st.markdown("### 📑 文献综述框架")
    st.markdown(framework)
    st.markdown("### 📚 参考来源文献")
    for f in source_files:
        st.caption(f"- {f.split('/')[-1]}")

def display_cross_result(answer, sources):
    st.success("跨领域创新方向生成完毕")
    st.markdown("### 💡 跨领域创新建议")
    st.write(answer)
    st.markdown("### 📖 参考来源")
    for s in set(sources):
        st.caption(f"- {s.split('/')[-1]}")

def display_general_result(answer, sources):
    st.markdown("### 🤖 回答")
    st.write(answer)
    st.markdown("### 🔗 来源片段")
    for i, doc in enumerate(sources[:5], 1):
        src = doc.metadata.get('source', 'unknown').split('/')[-1]
        st.caption(f"{i}. {src}")
        st.text(doc.page_content[:300] + "...")

# ====================== 主程序 ======================
def main():
    st.set_page_config(page_title="生物医药RAG智能系统", page_icon="🔬", layout="wide")
    st.title("🔬 生物医药RAG智能系统")
    st.subheader("⚡实验方法定位 | 📝综述框架生成 | 🌉跨领域创新发现")
    st.divider()

    if "db" not in st.session_state:
        st.session_state.db = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    with st.sidebar:
        st.header("📤 上传生物医药文献PDF")
        pdf_files = st.file_uploader(
            "支持多PDF（实验方法、药物递送、材料科学相关）",
            type="pdf",
            accept_multiple_files=True
        )
        if st.button("✅ 构建知识库", disabled=not pdf_files):
            with st.spinner("正在解析PDF并构建向量库...（约1-5分钟）"):
                try:
                    st.session_state.db = build_knowledge_base(pdf_files)
                    st.success("✅ 知识库构建完成！可开始提问。")
                except Exception as e:
                    st.error(f"构建失败：{e}")

    col1, col2, col3 = st.columns(3)
    with col1:
        exp_btn = st.button("🔍 实验方法快速定位")
    with col2:
        review_btn = st.button("📝 自动生成综述框架")
    with col3:
        cross_btn = st.button("🌉 跨领域关联发现")

    user_query = st.text_input(
        "✏️ 输入你的具体问题",
        value=st.session_state.last_query,
        placeholder="例如：Western Blot 实验步骤；CRISPR-Cas9 递送载体综述框架；MOF材料在药物缓释中的应用..."
    )

    manual_trigger = st.button("🚀 开始分析", type="primary")

    query_to_run = None
    scenario = None

    if exp_btn:
        query_to_run = "请根据知识库文献，给出脂质体包封率的测定方法（步骤、参数、注意事项）"
        scenario = "exp"
    elif review_btn:
        query_to_run = "刺激响应型纳米药物递送系统"
        scenario = "review"
    elif cross_btn:
        query_to_run = "将材料科学方法创新应用于药物递送系统"
        scenario = "cross"
    elif manual_trigger and user_query.strip():
        query_to_run = user_query.strip()
        if any(kw in query_to_run for kw in ["实验方法", "步骤", "测定", "protocol"]):
            scenario = "exp"
        elif any(kw in query_to_run for kw in ["综述", "框架", "进展"]):
            scenario = "review"
        elif any(kw in query_to_run for kw in ["跨领域", "材料科学", "创新"]):
            scenario = "cross"
        else:
            scenario = "general"

    if query_to_run and scenario:
        if st.session_state.db is None:
            st.error("请先上传PDF并构建知识库！")
        else:
            st.session_state.last_query = query_to_run
            llm = get_llm()

            if scenario == "exp":
                with st.spinner("正在精准定位实验方法段落..."):
                    answer, sources = run_experiment_method(query_to_run, st.session_state.db, llm)
                    display_experiment_result(answer, sources)

            elif scenario == "review":
                with st.spinner("正在迭代生成综述框架，请稍候..."):
                    framework, source_files = run_review_outline(query_to_run, st.session_state.db, llm)
                    display_review_result(framework, source_files)

            elif scenario == "cross":
                with st.spinner("正在整合本地文献与PubMed前沿，寻找跨领域创新点..."):
                    answer, sources = run_cross_domain(query_to_run, st.session_state.db, llm)
                    display_cross_result(answer, sources)

            else:
                with st.spinner("正在检索并生成答案..."):
                    answer, sources = run_general_qa(query_to_run, st.session_state.db, llm)
                    display_general_result(answer, sources)

    if st.session_state.db is None:
        st.info("👆 请从左侧上传PDF文献并点击「构建知识库」开始使用。")

if __name__ == "__main__":
    main()