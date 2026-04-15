# 生物医药RAG系统2

一个基于多智能体协作的生物医药智能系统，集成了临床决策支持、药物重定位发现和多模态知识管理功能。

## 核心功能

### 1. 多智能体协作架构
- **基于LangGraph的状态图驱动流程**
- **中央控制单元**：Supervisor、Planner、Executor、Finisher
- **下游专家Agent团队**：Retriever、Evidence Grader、Synthesizer、Fact Checker
- **流式输出**：实时展示Agent思考过程

### 2. 药物重定位发现工具
- **多源数据整合**：DrugBank、ChEMBL、DisGeNET、Open Targets、STRING
- **文献关系抽取**：PubMed API集成
- **知识图谱补全与假设生成**：PyKEEN
- **假设验证与排序**：RAG系统集成

### 3. 临床决策支持系统
- **多模态文档解析**：文本/图像/表格/公式
- **知识图谱 + 跨模态检索**
- **临床决策支持**：患者信息抽取、诊断推理、文献支持
- **会话管理**：列表、新建、删除、自动标题、上下文窗口压缩

### 4. 幻觉控制与审计
- **强制事实核查**
- **合规性审计日志**
- **管理员面板**

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 前端界面 | Streamlit |
| 后端服务 | FastAPI |
| 多智能体框架 | LangGraph |
| RAG 编排 | LangChain |
| 大语言模型 | 通义千问 (Qwen-Max) |
| 文本向量化 | DashScope text-embedding-v2 |
| 多模态解析 | MinerU 2.0（降级兼容 PyPDF） |
| 知识图谱 | NetworkX、Neo4j |
| 向量数据库 | ChromaDB |
| 缓存 | Redis |
| 外部知识增强 | PubMed (Bio.Entrez) |

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/yjz1-lujiu/biomedical-rag-system.git
cd biomedical-rag-system
```

### 2. 安装依赖
推荐使用 Python 3.10 及以上版本
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
复制示例文件并填入你的密钥：
```bash
cp .env.example .env
```
编辑 .env 文件：
```
DASHSCOPE_API_KEY=你的通义千问APIKey
PUBMED_EMAIL=你的邮箱@example.com
```

### 4. 启动服务

#### 4.1 启动Redis服务
```bash
# Windows
E:\Redis\Redis-x64-5.0.14.1\redis-server.exe

# Linux/Mac
redis-server
```

#### 4.2 启动Neo4j数据库（可选，用于知识图谱）
- 打开Neo4j Desktop
- 创建或启动一个数据库
- 确保数据库在bolt://localhost:7687上运行
- 用户名: neo4j
- 密码: Yjz61925!

#### 4.3 启动多智能体系统后端
```bash
python medical_multi_agent.py --mode api --host 0.0.0.0 --port 8000
```

#### 4.4 启动前端应用

**临床决策支持系统**：
```bash
streamlit run cdss_app.py --server.port 8511
```

**药物重定位发现工具**：
```bash
streamlit run drug_repositioning_app.py --server.port 8510
```

**多智能体系统前端**：
```bash
streamlit run medical_multi_agent_frontend.py --server.port 8502
```

## 系统访问地址

- **临床决策支持系统**：http://localhost:8511
- **药物重定位发现工具**：http://localhost:8510
- **多智能体系统前端**：http://localhost:8502
- **多智能体系统后端**：http://localhost:8000
- **API文档**：http://localhost:8000/docs

## 使用指南

### 1. 临床决策支持
1. 上传PDF文献并构建知识库
2. 输入临床病历文本
3. 点击"分析病历"按钮
4. 查看诊断结果和证据

### 2. 药物重定位分析
1. 输入药物名称（如Metformin）
2. 选择知识图谱嵌入模型（RotatE或ComplEx）
3. 设置返回结果数量
4. 点击"分析药物重定位"按钮
5. 查看潜在新适应症、证据路径和支持文献

### 3. 多智能体系统
1. 输入临床问题
2. 点击"提交"按钮
3. 查看系统处理过程和最终回答

## 示例

### 药物重定位分析示例
**输入药物**：二甲双胍（Metformin）
**输出**：
- 候选新适应症：结直肠癌（关联得分0.87）、多囊卵巢综合征（已知，得分0.95）...
- 证据路径：二甲双胍 → 激活AMPK → 抑制mTOR通路 → 结直肠癌细胞增殖抑制
- 文献支持：[PMID:23456789] 二甲双胍在结直肠癌小鼠模型中显著抑制肿瘤生长

### 临床决策支持示例
**输入病历**：患者，男，65岁，因"发热、咳嗽、咳黄痰3天"入院。查体：T 38.5℃，双肺可闻及湿啰音。血常规：WBC 14.2×10^9/L，中性粒细胞百分比85%。胸部CT示右下肺斑片状高密度影。
**输出**：
- 鉴别诊断：社区获得性肺炎（置信度: 0.92）
- 诊断方法：基于临床症状、体征和影像学检查
- 证据：发热、咳嗽、咳黄痰，双肺湿啰音，白细胞升高，胸部CT示右下肺斑片状高密度影
- 文献支持：相关研究表明...

## 常见问题

### Q1：构建知识库时提示 ModuleNotFoundError？
A：请确保已正确安装所有依赖：`pip install -r requirements.txt`

### Q2：运行时出现 DashScope API Key 无效或 Arrearage？
A：检查 .env 中的 DASHSCOPE_API_KEY 是否正确，且阿里云账户余额充足。

### Q3：多模态解析失败，提示 MinerU not available？
A：MinerU 需要特定系统库支持，若安装失败，取消勾选"启用多模态解析"即可使用纯文本降级模式。

### Q4：如何更新已上传的文献？
A：重新上传 PDF 并点击"构建知识库"，系统会覆盖旧的向量库。

### Q5：PubMed 检索失败？
A：请确保 .env 中填入了有效的邮箱地址，且网络可访问 NCBI 服务。

## 贡献指南

欢迎提交 Issue 或 Pull Request！如果你想为项目添加新功能，请遵循以下流程：
1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- GitHub Issues：提交问题
- 邮箱：1814016448@qq.com

## 致谢

- LangGraph - 多智能体协作框架
- LangChain - LLM 应用开发框架
- Streamlit - 快速构建数据应用
- DashScope - 稳定高效的中文 Embedding 与 LLM 服务
- PubMed - 权威的生物医学文献库
- PyKEEN - 知识图谱嵌入库
- Neo4j - 图数据库
- Redis - 缓存服务
