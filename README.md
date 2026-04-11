# 🔬 生物医药RAG智能系统

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个面向生物医药科研场景的 **检索增强生成（RAG）** 智能助手，能够帮助研究人员：
- ⚡ **3 分钟定位实验方法**（传统方式需 45 分钟）
- 📝 **自动生成文献综述框架**
- 🌉 **发现跨领域创新关联**（如材料科学 → 药物递送）

---

## ✨ 功能特点

### 1. 实验方法快速定位
- 上传文献 PDF，系统自动识别“实验方法”段落。
- 输入问题（如“脂质体包封率测定方法”），精准返回步骤、参数、试剂和注意事项。
- 所有回答均可溯源至原始文献片段。

### 2. 文献综述框架生成
- 将综述主题自动分解为 5~6 个子主题。
- 迭代检索相关文献并总结核心观点。
- 输出包含“引言—正文—结论”的完整结构化大纲。

### 3. 跨领域创新发现
- 融合本地文献与 **PubMed** 最新摘要。
- 针对“材料科学 × 药物递送”等交叉领域，提出具体创新方向及可行性分析。

### 4. 通用文献问答
- 支持任意生物医药相关问题的自由问答。
- 返回答案及参考来源片段，确保科研严谨性。

---

## 🛠️ 技术栈

| 组件               | 技术选型                                         |
| :----------------- | :----------------------------------------------- |
| **前端界面**       | Streamlit                                        |
| **RAG 编排**       | LangChain (新版 `create_retrieval_chain` API)    |
| **大语言模型**     | 通义千问 (Qwen-Max)                              |
| **文本向量化**     | DashScope `text-embedding-v2`                    |
| **向量数据库**     | Chroma                                           |
| **文档解析**       | PyPDFLoader                                      |
| **外部知识增强**   | PubMed (Bio.Entrez)                              |
| **段落分类**       | 基于关键词规则的自定义分类器                     |

---

## 📁 项目结构
<img width="733" height="305" alt="image" src="https://github.com/user-attachments/assets/baef1535-c85f-40d5-8417-8480574a9034" />
## 🚀 快速开始

### 1. 克隆项目
git clone https://github.com/yjz1-lujiu/biomedical-rag-system.git
cd biomedical-rag-system

2. 安装依赖
推荐使用 Python 3.10 及以上版本。
pip install -r requirements.txt

4. 配置环境变量
复制示例文件并填写你的密钥：
cp .env.example .env
编辑 .env 文件，填入你的通义千问 API Key 和邮箱（用于 PubMed 检索）：
DASHSCOPE_API_KEY=你的通义千问APIKey
PUBMED_EMAIL=你的邮箱@example.com

📖 使用指南
第一步：上传 PDF 文献
在左侧边栏点击 “Browse files”，上传你的生物医药相关 PDF（支持批量）。

点击 “✅ 构建知识库”，系统将解析文档并建立向量索引（约需 1~5 分钟）。
<img width="1488" height="395" alt="image" src="https://github.com/user-attachments/assets/2c0d856d-63a0-46e8-add6-6097640351bb" />


第二步：选择功能或输入问题
快捷按钮：

🔍 实验方法快速定位 → 自动搜索脂质体包封率测定方法。

📝 自动生成综述框架 → 以“刺激响应型纳米药物递送系统”为例。

🌉 跨领域关联发现 → 探索材料科学在药物递送中的创新应用。

手动输入：在输入框中自由提问，点击 “🚀 开始分析”。
<img width="1827" height="574" alt="image" src="https://github.com/user-attachments/assets/4fa06112-b3b3-46b7-9223-ecc36860dfc8" />


第三步：查看结果
回答内容将展示在主界面，下方附带 来源文献片段，点击可展开查看详情。

🧪 示例效果
输入：刺激响应型纳米药物递送系统
<img width="1815" height="916" alt="image" src="https://github.com/user-attachments/assets/7c5961ad-265a-4d41-942d-c53b2b4bc89a" />


❗ 常见问题
Q1：构建知识库时提示 ModuleNotFoundError？
A：请确保已正确安装所有依赖：pip install -r requirements.txt

Q2：运行时出现 DashScope API Key 无效 错误？
A：检查 .env 文件中的 DASHSCOPE_API_KEY 是否正确，且账户余额充足。

Q3：PubMed 检索失败？
A：请确保 .env 中填入了有效的邮箱地址，且网络可访问 NCBI 服务。

Q4：如何更新知识库？
A：重新上传 PDF 文件并点击“构建知识库”，系统会覆盖旧的向量库。

🤝 贡献指南
欢迎提交 Issue 或 Pull Request！如果你想为项目添加新功能（例如支持更多文件格式、多模态图表解析等），请遵循以下流程：

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
LangChain - 提供了强大的 LLM 应用开发框架。

Streamlit - 让数据应用开发变得简单优雅。

DashScope - 稳定高效的中文 Embedding 与 LLM 服务。

PubMed - 权威的生物医学文献库。



