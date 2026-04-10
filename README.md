# 生物医药RAG智能系统

## 项目简介
本项目是一个面向生物医药领域的检索增强生成（RAG）智能系统...（你可以从我们总结的报告中复制摘要部分）

## 功能特点
- **实验方法快速定位**：平均耗时从45分钟压缩至3分钟。
- **文献综述框架生成**：自动输出结构化的综述大纲。
- **跨领域关联发现**：融合本地与PubMed知识，提出创新方向。

## 快速开始
1. 克隆项目：
   `git clone https://github.com/你的用户名/你的仓库名.git`
2. 安装依赖：
   `pip install -r requirements.txt`
3. 配置环境变量：
   复制 `.env.example` 文件为 `.env`，并填入你的 `DASHSCOPE_API_KEY` 和 `PUBMED_EMAIL`。
4. 运行应用：
   `streamlit run main.py`

## 主要技术栈
- Python, Streamlit, LangChain, ChromaDB, DashScope, PubMed API
