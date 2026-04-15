#!/usr/bin/env python3
"""
智能生物医药系统 · 临床决策支持系统（CDSS）
- 多模态文档解析（文本/图像/表格/公式）
- 知识图谱 + 跨模态检索
- 临床决策支持：患者信息抽取、诊断推理、文献支持
- 会话管理：列表、新建、删除、自动标题、上下文窗口压缩
- 存储：SQLite/PostgreSQL + ChromaDB，可选 Redis 缓存
"""

import os
import json
import uuid
from datetime import datetime
import streamlit as st
from cdss_integrator import CDSSIntegrator
from main import SessionManager, MultimodalVectorStore, build_multimodal_knowledge_base, get_llm, get_title_llm, generate_response_with_memory
from drug_repositioning_model import DrugRepositioningModel
from hypothesis_validation import HypothesisValidator
from langchain_core.messages import HumanMessage, AIMessage
from multi_agent_system import run_agent_workflow

# 设置页面配置
st.set_page_config(
    page_title="智能生物医药系统",
    page_icon="🧬",
    layout="wide"
)

# 页面标题
st.title("🧬 智能生物医药系统")
st.subheader("临床决策支持 + 多模态知识管理")

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
        try:
            vector_store = MultimodalVectorStore()
            vector_store.initialize()
            st.session_state.vector_store = vector_store
        except Exception as e:
            print(f"加载知识库失败: {e}")
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# 初始化CDSS集成器
@st.cache_resource
def init_cdss():
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "Yjz61925!"
    return CDSSIntegrator(neo4j_uri, neo4j_user, neo4j_password)

cdss = init_cdss()

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
    st.header("上传文献")
    pdf_files = st.file_uploader("选择PDF文件", type="pdf", accept_multiple_files=True)
    if st.button("构建知识库", disabled=not pdf_files):
        with st.spinner("解析中..."):
            # 调用知识库构建函数
            st.session_state.vector_store = build_multimodal_knowledge_base(pdf_files)
            st.success("知识库构建完成")
    
    st.divider()
    st.header("系统配置")
    use_multi_agent = st.checkbox("启用多智能体系统", value=False)
    st.session_state.use_multi_agent = use_multi_agent
    st.info("本系统基于知识图谱和RAG技术，为临床诊断提供支持。")
    st.divider()
    st.header("使用说明")
    st.markdown("1. 上传PDF文献并构建知识库")
    st.markdown("2. 输入临床病历文本")
    st.markdown("3. 点击'分析病历'按钮")
    st.markdown("4. 查看诊断结果和证据")

# 主界面：历史消息渲染
for msg in st.session_state.messages_ui:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 病历输入
clinical_text = st.text_area(
    "请输入临床病历文本",
    placeholder="例如：患者，男，65岁，因\"发热、咳嗽、咳黄痰3天\"入院。查体：T 38.5℃，双肺可闻及湿啰音。血常规：WBC 14.2×10^9/L，中性粒细胞百分比85%。胸部CT示右下肺斑片状高密度影。",
    height=200
)

# 分析按钮
if st.button("分析病历", use_container_width=True):
    if not clinical_text:
        st.error("请输入临床病历文本")
    else:
        # 确保有当前会话
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = st.session_state.session_manager.create_session()

        # 获取当前历史
        history = st.session_state.session_manager.load_session(st.session_state.current_session_id)

        # 自动生成标题（第一条消息）
        if len(history.messages) == 0:
            title_llm = get_title_llm()
            st.session_state.session_manager.auto_generate_title(
                st.session_state.current_session_id, clinical_text, llm=title_llm
            )

        # 显示用户消息
        st.session_state.messages_ui.append({"role": "user", "content": clinical_text})
        with st.chat_message("user"):
            st.markdown(clinical_text)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                # 处理临床文本
                result = cdss.process_clinical_text(clinical_text, st.session_state.vector_store)
                
                # 检查Neo4j是否可用
                if not result.get('neo4j_available', True):
                    st.warning("⚠️ Neo4j数据库未连接，诊断推理功能不可用。请启动Neo4j数据库后再尝试。")
                    st.info("""启动Neo4j的步骤：
1. 打开Neo4j Desktop
2. 创建或启动一个数据库
3. 确保数据库在bolt://localhost:7687上运行
4. 用户名: neo4j
5. 密码: password""")
                
                # 显示结果
                st.success("分析完成！")
                
                # 提取的实体
                st.subheader("提取的临床实体")
                entities = result['extraction_result']['entities']
                if entities:
                    for entity in entities:
                        col1, col2, col3 = st.columns([3, 2, 2])
                        col1.write(f"**{entity['text']}**")
                        col2.write(f"类型: {entity['type']}")
                        col3.write(f"概念ID: {entity['concept_id']}")
                else:
                    st.info("未提取到实体")
                
                # 诊断结果
                st.subheader("鉴别诊断")
                diagnoses = result['diagnosis_result']['diagnoses']
                if diagnoses:
                    for i, diagnosis in enumerate(diagnoses):
                        with st.expander(f"{i+1}. {diagnosis['disease_name']} (置信度: {diagnosis['score']:.2f})"):
                            st.write(f"**诊断方法:** {diagnosis['method']}")
                            st.write(f"**证据:** {diagnosis['evidence']} (等级: {diagnosis['evidence_level']})")
                            
                            # 证据路径
                            st.write("**证据路径:**")
                            for item in diagnosis['evidence_path']:
                                st.write(f"- {item['type']}: {item['name']} (证据: {item['evidence']}, 等级: {item['evidence_level']})")
                else:
                    st.info("未生成诊断结果")
                
                # 文献证据
                st.subheader("文献支持")
                literature = result['literature_evidence']
                if literature:
                    for i, item in enumerate(literature):
                        with st.expander(f"文献 {i+1}"):
                            st.write(item['content'])
                            st.write(f"**来源:** {item['source']}")
                else:
                    st.info("未检索到相关文献")
                
                # 最终诊断建议
                st.subheader("诊断建议")
                st.write(result['final_advice']['advice'])
                
                # 保存交互
                history.add_messages([HumanMessage(content=clinical_text), AIMessage(content=result['final_advice']['advice'])])
                st.session_state.messages_ui.append({"role": "assistant", "content": result['final_advice']['advice']})
                st.session_state.session_manager.update_timestamp(st.session_state.current_session_id)
                st.rerun()

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

# 药物重定位分析
st.header("💊 药物重定位分析")
drug_name = st.text_input("输入药物名称", value="Metformin")
model_name = st.selectbox("知识图谱嵌入模型", ["RotatE", "ComplEx"], index=0)
top_k = st.slider("返回结果数量", min_value=1, max_value=10, value=5)

if st.button("分析药物重定位", use_container_width=True):
    if not drug_name:
        st.error("请输入药物名称")
    else:
        # 确保有当前会话
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = st.session_state.session_manager.create_session()

        # 获取当前历史
        history = st.session_state.session_manager.load_session(st.session_state.current_session_id)

        # 自动生成标题（第一条消息）
        if len(history.messages) == 0:
            title_llm = get_title_llm()
            st.session_state.session_manager.auto_generate_title(
                st.session_state.current_session_id, f"药物重定位分析: {drug_name}", llm=title_llm
            )

        # 显示用户消息
        st.session_state.messages_ui.append({"role": "user", "content": f"药物重定位分析: {drug_name}"})
        with st.chat_message("user"):
            st.markdown(f"药物重定位分析: {drug_name}")

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                # 初始化模型
                model = DrugRepositioningModel()
                validator = HypothesisValidator()
                
                try:
                    # 训练模型
                    model.train_model(model_name=model_name, epochs=50, embedding_dim=50)
                    
                    # 生成假设
                    hypotheses = model.generate_hypotheses(drug_name, top_k=top_k)
                    
                    # 验证并排序假设
                    validated_hypotheses = validator.validate_and_rank_hypotheses(hypotheses, vector_store=st.session_state.vector_store)
                    
                    # 显示结果
                    st.success("分析完成！")
                    
                    # 结果概览
                    st.subheader("🎯 分析概览")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("生成假设数量", len(hypotheses))
                    col2.metric("验证通过假设", len(validated_hypotheses))
                    col3.metric("分析时间", f"{datetime.now().strftime('%H:%M:%S')}")
                    
                    # 详细结果
                    st.subheader("📊 详细结果")
                    
                    for i, hypothesis in enumerate(validated_hypotheses, 1):
                        with st.expander(f"排名{i}: {hypothesis['drug']} → {hypothesis['disease']} (综合得分: {hypothesis['validation']['total_score']:.4f})"):
                            # 得分详情
                            st.write("### 得分详情")
                            score_cols = st.columns(3)
                            score_cols[0].metric("文献证据得分", f"{hypothesis['validation']['literature_score']:.4f}")
                            score_cols[1].metric("结合亲和力得分", f"{hypothesis['validation']['binding_score']:.4f}")
                            score_cols[2].metric("通路合理性得分", f"{hypothesis['validation']['pathway_score']:.4f}")
                            
                            # 证据路径
                            st.write("### 证据路径")
                            if hypothesis['evidence_paths']:
                                for path in hypothesis['evidence_paths']:
                                    st.write(f"**{path['type']}路径**:")
                                    for step in path['steps']:
                                        st.write(f"- {step}")
                            else:
                                st.write("无证据路径")
                            
                            # 支持文献
                            st.write("### 支持文献")
                            if hypothesis['validation']['evidence']:
                                for j, item in enumerate(hypothesis['validation']['evidence'], 1):
                                    st.write(f"**{j}. {item['title']}** (PMID: {item['pmid']})")
                                    if item.get('abstract'):
                                        st.write(f"摘要: {item['abstract'][:200]}...")
                            else:
                                st.write("无支持文献")
                    
                    # 保存交互
                    history.add_messages([HumanMessage(content=f"药物重定位分析: {drug_name}"), AIMessage(content=f"药物重定位分析完成，发现 {len(validated_hypotheses)} 个潜在新适应症")])
                    st.session_state.messages_ui.append({"role": "assistant", "content": f"药物重定位分析完成，发现 {len(validated_hypotheses)} 个潜在新适应症"})
                    st.session_state.session_manager.update_timestamp(st.session_state.current_session_id)
                    # 移除st.rerun()，避免页面重新加载导致详细结果不显示
                    
                except Exception as e:
                    st.error(f"分析出错: {e}")
                finally:
                    model.close()
                    validator.close()

# 示例病历
st.header("示例病历")
with st.expander("查看示例病历"):
    st.code("""患者，男，65岁，因\"发热、咳嗽、咳黄痰3天\"入院。查体：T 38.5℃，双肺可闻及湿啰音。血常规：WBC 14.2×10^9/L，中性粒细胞百分比85%。胸部CT示右下肺斑片状高密度影。""")

# 页脚
st.divider()
st.markdown("© 2026 智能生物医药系统")
