#!/usr/bin/env python3
"""
多智能体协作系统 · 生物医药RAG增强版
- 使用LangGraph构建Agent流水线
- 多智能体协作工作流
- 幻觉控制机制
- 合规性审计日志
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import langgraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_community.llms import Tongyi
import guidance

# ====================== 环境配置 ======================
from dotenv import load_dotenv
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "请替换为你的通义千问APIKey")

# ====================== 状态定义 ======================
@dataclass
class AgentState:
    session_id: str
    user_query: str
    planner_output: Optional[Dict[str, Any]] = None
    retrieved_evidence: Optional[List[Dict[str, Any]]] = None
    graded_evidence: Optional[List[Dict[str, Any]]] = None
    synthesized_answer: Optional[str] = None
    fact_checked_answer: Optional[Dict[str, Any]] = None
    audit_log: Optional[Dict[str, Any]] = None

# ====================== 输出结构定义 ======================
class EvidenceGrade(BaseModel):
    evidence_id: str
    source: str
    content: str
    grade: str  # 1a-5级
    confidence: float

class Claim(BaseModel):
    claim: str
    evidence_source: str
    evidence_grade: str
    confidence: float

class FactCheckedAnswer(BaseModel):
    claims: List[Claim]
    final_answer: str
    overall_confidence: float

# ====================== LLM初始化 ======================
def get_llm(temperature=0.3, max_tokens=1500):
    return Tongyi(
        model_name="qwen-max",
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens
    )

# ====================== 智能体定义 ======================
class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def parse_query(self, state: AgentState) -> Dict[str, Any]:
        """解析用户临床问题，拆解为子任务"""
        prompt = f"""
你是一个医疗规划师，负责将复杂的临床问题拆解为具体的子任务。

用户问题：{state.user_query}

请将此问题拆解为2-4个子任务，每个子任务应该是具体的、可执行的，并且能够帮助回答原始问题。

输出格式：
{{
  "subtasks": [
    {{
      "id": "任务1",
      "description": "具体任务描述"
    }},
    ...
  ],
  "focus_areas": ["关键领域1", "关键领域2", ...]
}}
"""
        
        response = self.llm.invoke(prompt)
        try:
            output = json.loads(response)
        except:
            # 如果解析失败，返回默认结构
            output = {
                "subtasks": [
                    {"id": "任务1", "description": "检索相关医学知识"},
                    {"id": "任务2", "description": "分析临床证据"},
                    {"id": "任务3", "description": "生成专业回答"}
                ],
                "focus_areas": ["临床指南", "医学文献"]
            }
        
        return {"planner_output": output}

class RetrieverAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve_evidence(self, state: AgentState) -> Dict[str, Any]:
        """并行检索内部知识库和外部API"""
        evidence = []
        
        # 1. 内部知识库检索
        if self.vector_store:
            from main import CrossModalRetriever
            retriever = CrossModalRetriever(self.vector_store)
            blocks = retriever.retrieve(state.user_query, top_k=5)
            for i, block in enumerate(blocks):
                evidence.append({
                    "id": f"internal_{i}",
                    "source": "internal_knowledge_base",
                    "content": block.content,
                    "metadata": block.metadata
                })
        
        # 2. 外部API检索（PubMed）
        try:
            from main import search_pubmed_abstracts
            pubmed_results = search_pubmed_abstracts(state.user_query, max_results=3)
            for i, result in enumerate(pubmed_results):
                evidence.append({
                    "id": f"pubmed_{i}",
                    "source": "pubmed",
                    "content": result.get("abstract", ""),
                    "metadata": {"title": result.get("title", ""), "pmid": result.get("pmid", "")}
                })
        except:
            pass
        
        return {"retrieved_evidence": evidence}

class EvidenceGraderAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def grade_evidence(self, state: AgentState) -> Dict[str, Any]:
        """对检索到的每条证据进行等级评估"""
        graded = []
        
        for item in state.retrieved_evidence or []:
            prompt = f"""
你是一个医学证据评估专家，根据牛津循证医学中心的标准对医学证据进行等级评估。

证据内容：{item['content']}
证据来源：{item['source']}

请评估此证据的等级（1a-5级）和置信度（0-1）。

牛津循证医学中心证据等级标准：
1a: 系统评价或Meta分析，纳入多项随机对照试验
1b: 单个随机对照试验
2a: 系统评价或Meta分析，纳入多项队列研究
2b: 单个队列研究
3a: 系统评价或Meta分析，纳入多项病例对照研究
3b: 单个病例对照研究
4: 病例系列
5: 专家意见

输出格式：
{{
  "evidence_id": "{item['id']}",
  "source": "{item['source']}",
  "content": "{item['content']}",
  "grade": "等级",
  "confidence": 置信度
}}
"""
            
            response = self.llm.invoke(prompt)
            try:
                grade_info = json.loads(response)
                grade_info["evidence_id"] = item["id"]
                grade_info["source"] = item["source"]
                grade_info["content"] = item["content"]
                graded.append(grade_info)
            except:
                # 如果解析失败，返回默认等级
                graded.append({
                    "evidence_id": item["id"],
                    "source": item["source"],
                    "content": item["content"],
                    "grade": "5",
                    "confidence": 0.5
                })
        
        return {"graded_evidence": graded}

class SynthesizerAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def synthesize_answer(self, state: AgentState) -> Dict[str, Any]:
        """综合证据生成初步回答"""
        # 构建证据上下文
        evidence_context = ""
        for item in state.graded_evidence or []:
            evidence_context += f"【证据】{item['content']}（等级：{item['grade']}，来源：{item['source']}）\n"
        
        prompt = f"""
你是一个生物医药专家，根据提供的证据生成专业、准确的回答。

用户问题：{state.user_query}

【证据】
{evidence_context}

请基于以上证据，生成一个全面、准确的回答，引用相关证据支持你的观点。
"""
        
        response = self.llm.invoke(prompt)
        return {"synthesized_answer": response}

class FactCheckerAgent:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
    
    def check_facts(self, state: AgentState) -> Dict[str, Any]:
        """对回答中的每个断言进行逆向检索验证"""
        # 提取回答中的断言
        prompt = f"""
你是一个医疗事实核查员，负责从医学回答中提取具体的断言。

回答内容：{state.synthesized_answer}

请提取出回答中的每个具体断言，每个断言应该是一个独立的医学声明。

输出格式：
{{
  "claims": [
    "断言1",
    "断言2",
    ...
  ]
}}
"""
        
        response = self.llm.invoke(prompt)
        try:
            claims_data = json.loads(response)
            claims = claims_data.get("claims", [])
        except:
            claims = [state.synthesized_answer]
        
        # 对每个断言进行验证
        validated_claims = []
        for claim in claims:
            # 检索相关证据
            if self.vector_store:
                from main import CrossModalRetriever
                retriever = CrossModalRetriever(self.vector_store)
                blocks = retriever.retrieve(claim, top_k=3)
                evidence_content = "\n".join([block.content for block in blocks])
            else:
                evidence_content = ""
            
            # 评估断言
            validation_prompt = f"""
你是一个医疗事实核查员，负责评估医学断言的准确性。

断言：{claim}

相关证据：{evidence_content}

请评估此断言的准确性，并提供证据来源和置信度。

输出格式：
{{
  "claim": "{claim}",
  "evidence_source": "证据来源描述",
  "evidence_grade": "证据等级（1a-5）",
  "confidence": 置信度（0-1）
}}
"""
            
            validation_response = self.llm.invoke(validation_prompt)
            try:
                validated_claim = json.loads(validation_response)
                validated_claim["claim"] = claim
                validated_claims.append(validated_claim)
            except:
                validated_claims.append({
                    "claim": claim,
                    "evidence_source": "无",
                    "evidence_grade": "5",
                    "confidence": 0.5
                })
        
        # 计算整体置信度
        if validated_claims:
            overall_confidence = sum(c.get("confidence", 0) for c in validated_claims) / len(validated_claims)
        else:
            overall_confidence = 0.5
        
        # 生成最终回答
        final_answer = state.synthesized_answer
        
        fact_checked = {
            "claims": validated_claims,
            "final_answer": final_answer,
            "overall_confidence": overall_confidence
        }
        
        return {"fact_checked_answer": fact_checked}

# ====================== 审计日志 ======================
class AuditLogger:
    def __init__(self, db_url=None):
        self.db_url = db_url
    
    def create_audit_log(self, state: AgentState) -> Dict[str, Any]:
        """创建审计日志"""
        log = {
            "session_id": state.session_id,
            "timestamp": datetime.now().isoformat(),
            "user_query": state.user_query,
            "intermediate_steps": {
                "planner": state.planner_output,
                "retrieved_evidence_count": len(state.retrieved_evidence) if state.retrieved_evidence else 0,
                "graded_evidence_count": len(state.graded_evidence) if state.graded_evidence else 0,
                "synthesized_answer": state.synthesized_answer
            },
            "final_answer": state.fact_checked_answer.get("final_answer", "") if state.fact_checked_answer else "",
            "cited_evidence": [
                {"id": e["evidence_id"], "source": e["source"], "grade": e["grade"]}
                for e in state.graded_evidence or []
            ],
            "overall_confidence": state.fact_checked_answer.get("overall_confidence", 0) if state.fact_checked_answer else 0
        }
        
        # 保存到文件（实际应用中可以保存到PostgreSQL）
        log_dir = "./audit_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        
        return {"audit_log": log}

# ====================== 构建LangGraph ======================
def build_agent_graph(vector_store=None) -> Any:
    """构建多智能体协作图"""
    llm = get_llm()
    
    # 初始化智能体
    planner = PlannerAgent(llm)
    retriever = RetrieverAgent(vector_store)
    grader = EvidenceGraderAgent(llm)
    synthesizer = SynthesizerAgent(llm)
    fact_checker = FactCheckerAgent(llm, vector_store)
    auditor = AuditLogger()
    
    # 定义状态图
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("planner", planner.parse_query)
    graph.add_node("retriever", retriever.retrieve_evidence)
    graph.add_node("grader", grader.grade_evidence)
    graph.add_node("synthesizer", synthesizer.synthesize_answer)
    graph.add_node("fact_checker", fact_checker.check_facts)
    graph.add_node("auditor", auditor.create_audit_log)
    
    # 添加边
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "grader")
    graph.add_edge("grader", "synthesizer")
    graph.add_edge("synthesizer", "fact_checker")
    graph.add_edge("fact_checker", "auditor")
    graph.add_edge("auditor", END)
    
    # 编译图
    compiled = graph.compile()
    return compiled

# ====================== 主函数 ======================
def run_agent_workflow(user_query: str, session_id: str, vector_store=None) -> Dict[str, Any]:
    """运行智能体工作流"""
    try:
        # 构建图
        graph = build_agent_graph(vector_store)
        
        # 初始化状态
        initial_state = AgentState(
            session_id=session_id,
            user_query=user_query
        )
        
        # 运行工作流（添加超时处理）
        import time
        start_time = time.time()
        timeout = 300  # 5分钟超时
        
        # 流式执行，以便及时捕获错误
        result = None
        for chunk in graph.stream(initial_state):
            # 检查超时
            if time.time() - start_time > timeout:
                raise TimeoutError("多智能体系统执行超时")
            result = chunk
        
        if result:
            return result
        else:
            # 如果没有结果，返回默认值
            return {
                "fact_checked_answer": {
                    "final_answer": "多智能体系统执行失败，请稍后重试",
                    "claims": [],
                    "overall_confidence": 0.0
                }
            }
    except Exception as e:
        # 捕获所有异常，确保系统不会因为错误而完全挂起
        print(f"多智能体系统执行错误: {e}")
        return {
            "fact_checked_answer": {
                "final_answer": f"多智能体系统执行失败: {str(e)}",
                "claims": [],
                "overall_confidence": 0.0
            }
        }

if __name__ == "__main__":
    # 测试
    test_query = "对于eGFR<30的2型糖尿病患者，是否可以使用SGLT-2抑制剂？"
    test_session_id = str(uuid.uuid4())
    
    result = run_agent_workflow(test_query, test_session_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
