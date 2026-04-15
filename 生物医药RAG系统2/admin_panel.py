#!/usr/bin/env python3
"""
智能生物医药系统 · 管理员面板
- 审计日志查看
- 系统配置管理
- 性能监控
"""

import os
import json
import streamlit as st
from datetime import datetime, timedelta
from audit_logger import AuditLogger
from main import SessionManager

# 设置页面配置
st.set_page_config(
    page_title="智能生物医药系统 - 管理员面板",
    page_icon="🛠️",
    layout="wide"
)

# 页面标题
st.title("🛠️ 智能生物医药系统 - 管理员面板")

# 初始化审计日志管理器
@st.cache_resource
def init_audit_logger():
    return AuditLogger()

audit_logger = init_audit_logger()

# 初始化会话管理器
@st.cache_resource
def init_session_manager():
    return SessionManager()

session_manager = init_session_manager()

# 侧边栏
with st.sidebar:
    st.header("导航")
    selected_page = st.radio(
        "选择页面",
        ["审计日志", "系统配置", "性能监控"]
    )
    
    st.divider()
    st.header("系统信息")
    st.write(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"审计日志目录: ./audit_logs")

# 审计日志页面
if selected_page == "审计日志":
    st.subheader("📋 审计日志")
    
    # 过滤选项
    col1, col2, col3 = st.columns(3)
    with col1:
        time_range = st.selectbox(
            "时间范围",
            ["全部", "今天", "昨天", "最近7天", "最近30天"]
        )
    with col2:
        session_id_filter = st.text_input("会话ID")
    with col3:
        search_query = st.text_input("搜索关键词")
    
    # 日志摘要
    st.subheader("📊 日志摘要")
    logs = audit_logger.get_logs(session_id_filter)
    
    # 计算统计信息
    total_logs = len(logs)
    total_sessions = len(set(log["session_id"] for log in logs))
    avg_confidence = sum(log.get("overall_confidence", 0) for log in logs) / total_logs if total_logs > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("总日志数", total_logs)
    col2.metric("总会话数", total_sessions)
    col3.metric("平均置信度", f"{avg_confidence:.2f}")
    
    # 详细日志列表
    st.subheader("📝 详细日志")
    
    if logs:
        # 分页
        page_size = 10
        total_pages = (total_logs + page_size - 1) // page_size
        page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_logs = logs[start_idx:end_idx]
        
        # 显示日志
        for log in paginated_logs:
            with st.expander(f"日志ID: {log['log_id']} | 会话: {log['session_id']} | 时间: {log['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**用户查询:**")
                    st.write(log['user_query'])
                    st.write("**最终回答:**")
                    st.write(log['final_answer'])
                with col2:
                    st.write("**置信度:**")
                    st.write(f"{log['overall_confidence']:.2f}")
                    st.write("**引用证据:**")
                    for evidence in log['cited_evidence']:
                        st.write(f"- {evidence['source']} (等级: {evidence['grade']})")
                
                st.write("**中间步骤:**")
                st.json(log['intermediate_steps'])
    else:
        st.info("没有找到日志")
    
    # 导出功能
    st.subheader("💾 导出日志")
    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox("导出格式", ["JSON", "CSV"])
    with col2:
        export_session_id = st.text_input("导出特定会话", placeholder="留空导出所有")
    
    if st.button("导出日志"):
        export_file = audit_logger.export_logs(export_session_id, format=export_format.lower())
        if export_file:
            st.success(f"日志已导出到: {export_file}")
            with open(export_file, "rb") as f:
                st.download_button(
                    label="下载导出文件",
                    data=f,
                    file_name=os.path.basename(export_file),
                    mime="application/json" if export_format == "JSON" else "text/csv"
                )
        else:
            st.error("导出失败")

# 系统配置页面
elif selected_page == "系统配置":
    st.subheader("⚙️ 系统配置")
    
    # 环境变量
    st.subheader("环境变量")
    env_vars = {
        "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY", "未设置"),
        "DATABASE_URL": os.getenv("DATABASE_URL", "未设置"),
        "REDIS_URL": os.getenv("REDIS_URL", "未设置"),
        "USE_POSTGRES": os.getenv("USE_POSTGRES", "false"),
    }
    
    for key, value in env_vars.items():
        if key == "DASHSCOPE_API_KEY" and value and len(value) > 10:
            value = value[:10] + "***"
        st.write(f"**{key}:** {value}")
    
    # 会话管理
    st.subheader("会话管理")
    sessions = session_manager.get_all_sessions()
    st.write(f"**总会话数:** {len(sessions)}")
    
    if sessions:
        with st.expander("查看所有会话"):
            for sess in sessions:
                st.write(f"- ID: {sess['session_id']} | 标题: {sess['title']} | 更新时间: {sess['updated_at']}")
    
    # 知识库状态
    st.subheader("知识库状态")
    kb_path = "multimodal_rag_db"
    if os.path.exists(kb_path) and os.path.isdir(kb_path):
        st.success("知识库已构建")
        # 统计知识库文件数
        kb_files = 0
        for root, dirs, files in os.walk(kb_path):
            kb_files += len(files)
        st.write(f"**知识库文件数:** {kb_files}")
    else:
        st.warning("知识库未构建")

# 性能监控页面
elif selected_page == "性能监控":
    st.subheader("📈 性能监控")
    
    # 系统资源
    st.subheader("系统资源")
    try:
        import psutil
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("CPU使用率", f"{cpu_usage}%")
        col2.metric("内存使用率", f"{memory.percent}%")
        col3.metric("磁盘使用率", f"{disk.percent}%")
    except ImportError:
        st.info("psutil未安装，无法显示系统资源使用情况")
    
    # 日志统计
    st.subheader("日志统计")
    logs = audit_logger.get_logs()
    
    # 按日期统计
    if logs:
        date_counts = {}
        for log in logs:
            date = log['timestamp'][:10]
            date_counts[date] = date_counts.get(date, 0) + 1
        
        st.bar_chart(date_counts)
    else:
        st.info("没有日志数据")

# 页脚
st.divider()
st.markdown("© 2026 智能生物医药系统 - 管理员面板")
