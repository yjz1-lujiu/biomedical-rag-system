#!/usr/bin/env python3
"""
药物重定位发现工具演示页面
- 用户输入老药名称，系统展示潜在新适应症、证据链路和相关文献摘要
"""

import os
import json
import streamlit as st
from datetime import datetime

# 导入药物重定位模块
from drug_repositioning_model import DrugRepositioningModel
from hypothesis_validation import HypothesisValidator

# 设置页面配置
st.set_page_config(
    page_title="药物重定位发现工具",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #757575;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .input-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .expander-header {
        font-weight: bold;
        color: #2c3e50;
    }
    .evidence-path {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #2196f3;
    }
    .literature-card {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
    }
    .submit-button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.75rem;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    .submit-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #9e9e9e;
        font-size: 0.9rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.markdown('<h1 class="main-header">💊 药物重定位发现工具</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">通过挖掘"药物-靶点-疾病"的隐含关系，发现老药新用的潜力</p>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown('<h2 class="sidebar-header">分析配置</h2>', unsafe_allow_html=True)
    
    # 药物输入
    drug_name = st.text_input("药物名称", value="Metformin", help="输入要分析的药物名称")
    
    # 模型配置
    model_name = st.selectbox("知识图谱嵌入模型", ["RotatE", "ComplEx"], index=0, help="选择用于知识图谱嵌入的模型")
    top_k = st.slider("返回结果数量", min_value=1, max_value=10, value=5, help="设置要返回的候选适应症数量")
    
    # 运行按钮
    run_button = st.button("开始分析", use_container_width=True)

# 主内容
if run_button:
    with st.spinner("正在分析..."):
        # 初始化模型
        model = DrugRepositioningModel()
        validator = HypothesisValidator()
        
        try:
            # 训练模型
            model.train_model(model_name=model_name, epochs=50, embedding_dim=50)
            
            # 生成假设
            hypotheses = model.generate_hypotheses(drug_name, top_k=top_k)
            
            # 验证并排序假设
            validated_hypotheses = validator.validate_and_rank_hypotheses(hypotheses)
            
            # 显示结果
            st.markdown(f"<h2 style='color: #2c3e50; margin-top: 2rem;'>{drug_name}的药物重定位分析结果</h2>", unsafe_allow_html=True)
            
            # 结果概览
            st.markdown("<h3 style='color: #424242; margin-top: 1.5rem;'>🎯 分析概览</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("生成假设数量", len(hypotheses))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("验证通过假设", len(validated_hypotheses))
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("分析时间", f"{datetime.now().strftime('%H:%M:%S')}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 详细结果
            st.markdown("<h3 style='color: #424242; margin-top: 2rem;'>📊 详细结果</h3>", unsafe_allow_html=True)
            
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
                            st.markdown('<div class="evidence-path">', unsafe_allow_html=True)
                            st.write(f"**{path['type']}路径**:")
                            for step in path['steps']:
                                st.write(f"- {step}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.write("无证据路径")
                    
                    # 支持文献
                    st.write("### 支持文献")
                    if hypothesis['validation']['evidence']:
                        for j, item in enumerate(hypothesis['validation']['evidence'], 1):
                            st.markdown('<div class="literature-card">', unsafe_allow_html=True)
                            st.write(f"**{j}. {item['title']}** (PMID: {item['pmid']})")
                            if item.get('abstract'):
                                st.write(f"摘要: {item['abstract'][:200]}...")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.write("无支持文献")
            
            # 可视化
            st.markdown("<h3 style='color: #424242; margin-top: 2rem;'>🔍 可视化分析</h3>", unsafe_allow_html=True)
            st.write("### 药物-靶点-疾病关系网络")
            st.info("关系网络可视化功能正在开发中...")
            
            # 保存结果
            result_path = os.path.join("./drug_repositioning_data", f"{drug_name}_repositioning_results.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(validated_hypotheses, f, ensure_ascii=False, indent=2)
            
            st.success(f"分析结果已保存到: {result_path}")
            
        except Exception as e:
            st.error(f"分析出错: {e}")
        finally:
            model.close()
            validator.close()
else:
    # 示例输入
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.info("请在侧边栏输入药物名称并点击'开始分析'按钮")
    
    # 示例数据
    st.subheader("示例输入")
    st.code("Metformin")
    
    st.subheader("示例输出")
    st.write("#### 候选新适应症:")
    st.write("- 结直肠癌（关联得分0.5211）")
    st.write("- 乳腺癌（关联得分0.4779）")
    
    st.write("#### 证据路径:")
    st.write("Metformin → AMPK → 结直肠癌")
    st.write("Metformin → mTOR → 结直肠癌")
    
    st.write("#### 文献支持:")
    st.write("[PMID:30405532] 相关研究表明...")
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("药物重定位发现工具 - 基于知识图谱和深度学习")
st.markdown('</div>', unsafe_allow_html=True)
