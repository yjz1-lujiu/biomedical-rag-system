# 🔬 生物医药RAG智能系统 · 多模态增强版

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个面向生物医药科研场景的**多模态检索增强生成（RAG）**智能助手，深度融合 **RAG-Anything** 设计理念，不仅能理解文献文字，还能解析图表、表格和公式，帮助研究人员：

- ⚡ **3 分钟定位实验方法**
- 📝 **自动生成文献综述框架**
- 🌉 **发现跨领域创新关联**（如材料科学 → 药物递送）
- 🖼️ **图表智能问答**（基于文献中图像、表格的内容回答）

---

## ✨ 新特性（多模态增强版）

| 能力 | 纯文本版 | 多模态增强版 |
| :--- | :--- | :--- |
| 文献解析 | 仅提取纯文本 | **MinerU 2.0** 解析文本、图像、表格、公式 |
| 内容关联 | 无 | **知识图谱** 建立“文字→图表”引用关系 |
| 检索方式 | 向量相似度 | 向量检索 + **图谱扩展** 跨模态混合检索 |
| 图表问答 | 不支持 | 可描述图表内容，结合图注回答 |
| 实验方法定位 | 基于关键词过滤 | 同时检索相关实验步骤文本及**示意图/流程图** |

---

## 🛠️ 技术栈

| 组件 | 技术选型 |
| :--- | :--- |
| **前端界面** | Streamlit |
| **RAG 编排** | LangChain (新版 `create_retrieval_chain` API) |
| **大语言模型** | 通义千问 (Qwen-Max) |
| **文本向量化** | DashScope `text-embedding-v2` |
| **多模态解析** | MinerU 2.0（降级兼容 PyPDF） |
| **知识图谱** | NetworkX |
| **向量数据库** | Chroma（多 Collection） |
| **外部知识增强** | PubMed (Bio.Entrez) |


biomedical-rag-system/
├── main.py # 主程序入口（多模态版）
├── requirements.txt # Python 依赖列表
├── .env.example # 环境变量示例
├── .gitignore # Git 忽略规则
├── README.md # 项目说明文档
├── multimodal_rag_db/ # 多模态向量库持久化目录（自动生成）
└── temp_pdfs/ # 临时 PDF 存储目录（自动清理）


🚀 快速开始
 1. 克隆项目
git clone https://github.com/yjz1-lujiu/biomedical-rag-system.git
cd biomedical-rag-system

2. 安装依赖
推荐使用 Python 3.10 及以上版本
pip install -r requirements.txt
若 mineru 安装失败（需特定系统依赖），系统会自动降级为 PyPDF 纯文本模式，图表解析功能将受限。

3. 配置环境变量
复制示例文件并填入你的密钥：
cp .env.example .env
编辑 .env 文件：
DASHSCOPE_API_KEY=你的通义千问APIKey
PUBMED_EMAIL=你的邮箱@example.com

4. 运行应用
streamlit run main.py
应用将在浏览器中自动打开（默认地址 http://localhost:8501）。

📖 使用指南
第一步：上传 PDF 文献
在左侧边栏点击 “Browse files”，上传生物医药相关 PDF（支持批量）。
勾选 “启用多模态解析（推荐）”。
点击 “✅ 构建知识库”，系统将解析文档并建立多模态索引（约需 2~8 分钟）。

第二步：选择功能或输入问题
快捷按钮：
🔍 实验方法定位 → 默认查询脂质体包封率测定方法。

📝 综述框架 → 以“刺激响应型纳米药物递送系统”为例。

🌉 跨领域发现 → 探索材料科学在药物递送中的创新应用。

📊 图表问答 → 描述文献中与药物释放相关的图表内容。

手动输入：在输入框中自由提问，点击 “🚀 开始分析”。

第三步：查看结果
回答内容展示在主界面，下方附带 来源内容块类型（文本/图片/表格）及文件名。

🧪 示例效果
输入：
请描述文献中与纳米药物相关的图表内容

输出：
<img width="1386" height="642" alt="image" src="https://github.com/user-attachments/assets/1880e5c5-114b-4cef-a891-d2567617cd63" />

❗ 常见问题
Q1：构建知识库时提示 ModuleNotFoundError？
A：请确保已正确安装所有依赖：pip install -r requirements.txt

Q2：运行时出现 DashScope API Key 无效 或 Arrearage？
A：检查 .env 中的 DASHSCOPE_API_KEY 是否正确，且阿里云账户余额充足。

Q3：多模态解析失败，提示 MinerU not available？
A：MinerU 需要特定系统库支持，若安装失败，取消勾选“启用多模态解析”即可使用纯文本降级模式。

Q4：如何更新已上传的文献？
A：重新上传 PDF 并点击“构建知识库”，系统会覆盖旧的向量库。

Q5：PubMed 检索失败？
A：请确保 .env 中填入了有效的邮箱地址，且网络可访问 NCBI 服务。

🤝 贡献指南
欢迎提交 Issue 或 Pull Request！如果你想为项目添加新功能（例如支持更多文件格式、VLM 图表直接问答等），请遵循以下流程：

Fork 本仓库

创建你的特性分支 (git checkout -b feature/AmazingFeature)

提交你的改动 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

打开一个 Pull Request

📄 许可证
本项目采用 MIT 许可证。

📧 联系方式
如有问题或建议，欢迎通过以下方式联系：

GitHub Issues：提交问题

邮箱：1814016448@qq.com

🌟 致谢
RAG-Anything - 多模态 RAG 设计灵感

MinerU - 强大的开源文档解析模型

LangChain - LLM 应用开发框架

Streamlit - 快速构建数据应用

DashScope - 稳定高效的中文 Embedding 与 LLM 服务

PubMed - 权威的生物医学文献库


