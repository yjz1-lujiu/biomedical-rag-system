#!/usr/bin/env python3
"""
生物医药多智能体系统前端
"""

import streamlit as st
import requests
import json

# 设置页面配置
st.set_page_config(
    page_title="生物医药多智能体系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .input-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .status-container {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #2196f3;
    }
    .answer-container {
        background-color: #f1f8e9;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #4caf50;
    }
    .error-container {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #f44336;
    }
    .submit-button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .submit-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .placeholder-text {
        color: #9e9e9e;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.markdown('<h1 class="main-header">🧬 生物医药多智能体系统</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #757575; margin-bottom: 2rem;'>基于LangGraph的智能医疗助手</p>", unsafe_allow_html=True)

# 用户输入区域
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    user_query = st.text_area(
        "临床问题",
        placeholder="例如：对于eGFR<30的2型糖尿病患者，是否可以使用SGLT-2抑制剂？",
        height=100,
        help="请输入您的临床问题，系统将通过多智能体协作为您提供专业的医疗建议"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_button = st.button("提交", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 处理提交
if submit_button:
    if not user_query:
        st.markdown('<div class="error-container">请输入临床问题</div>', unsafe_allow_html=True)
    else:
        # 显示状态
        status_container = st.empty()
        answer_container = st.empty()
        
        # 调用FastAPI接口
        url = "http://localhost:8000/v1/chat"
        data = {"query": user_query}
        
        try:
            response = requests.post(url, json=data, stream=True)
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_part = line[6:]
                        try:
                            json_data = json.loads(data_part)
                            if json_data.get("type") == "status":
                                status_container.markdown(f'<div class="status-container">{json_data.get("message")}</div>', unsafe_allow_html=True)
                            elif json_data.get("type") == "answer":
                                answer_container.markdown(f'<div class="answer-container">{json_data.get("message")}</div>', unsafe_allow_html=True)
                            elif json_data.get("type") == "error":
                                status_container.markdown(f'<div class="error-container">{json_data.get("message")}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f'<div class="error-container">解析响应时出错: {str(e)}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-container">调用API失败: {str(e)}</div>', unsafe_allow_html=True)

# 页脚
st.markdown("""
<div style='text-align: center; margin-top: 3rem; color: #9e9e9e; font-size: 0.9rem;'>
    © 2026 生物医药多智能体系统 | 基于LangGraph和Streamlit构建
</div>
""", unsafe_allow_html=True)
