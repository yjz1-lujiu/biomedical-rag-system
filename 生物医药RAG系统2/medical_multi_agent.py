#!/usr/bin/env python3
"""
生物医药多智能体协作系统
- 基于LangGraph的多智能体架构
- 规划-执行双循环架构
- 下游专家Agent团队
- FastAPI服务化
- Streamlit前端适配
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_multi_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ====================== 缓存机制 ======================
class SimpleCache:
    """简单的内存缓存"""
    def __init__(self, max_size=1000, ttl=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        """获取缓存值"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        """设置缓存值"""
        if len(self.cache) >= self.max_size:
            # 移除最早的缓存项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()

# 创建全局缓存实例
cache = SimpleCache()

# 线程池执行器
executor = ThreadPoolExecutor(max_workers=4)

import langgraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_community.llms import Tongyi
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import streamlit as st
import requests

# ====================== 环境配置 ======================
from dotenv import load_dotenv
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "请替换为你的通义千问APIKey")

# ====================== 状态定义 ======================
@dataclass
class AgentState:
    """多智能体系统状态"""
    messages: List[BaseMessage] = field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None
    next_step: Optional[Literal["planner", "executor", "finisher"]] = "planner"
    current_task: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    graded_evidence: List[Dict[str, Any]] = field(default_factory=list)
    user_query: Optional[str] = None
    session_id: Optional[str] = None

# ====================== LLM初始化 ======================
def get_llm(temperature=0.3, max_tokens=1500):
    return Tongyi(
        model_name="qwen-max",
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens
    )

# ====================== 中央控制单元 ======================
class SupervisorNode:
    """中央协调器"""
    def __init__(self):
        pass
    
    def route(self, state: AgentState) -> Dict[str, Any]:
        """根据当前状态决定下一步路由"""
        if state.next_step == "planner":
            return {"__next__": "planner"}
        elif state.next_step == "executor":
            return {"__next__": "executor"}
        elif state.next_step == "finisher":
            return {"__next__": "finisher"}
        else:
            return {"__next__": END}

class PlannerNode:
    """任务规划器"""
    def __init__(self, llm):
        self.llm = llm
    
    def plan(self, state: AgentState) -> Dict[str, Any]:
        """将复杂用户问题拆解为子任务"""
        user_query = state.user_query
        
        try:
            logger.info(f"开始任务规划: {user_query}")
            
            # 直接返回默认规划，简化测试
            plan = {
                "subtasks": [
                    {"id": "任务1", "description": "检索相关医学知识"},
                    {"id": "任务2", "description": "分析临床证据"},
                    {"id": "任务3", "description": "生成专业回答"}
                ],
                "focus_areas": ["临床指南", "医学文献"]
            }
            logger.info(f"使用默认任务规划，生成 {len(plan.get('subtasks', []))} 个子任务")
        
            return {
                "plan": plan,
                "next_step": "executor",
                "current_task": "规划任务"
            }
        except Exception as e:
            logger.error(f"任务规划失败: {e}")
            # 错误处理：返回默认规划
            return {
                "plan": {
                    "subtasks": [
                        {"id": "任务1", "description": "检索相关医学知识"},
                        {"id": "任务2", "description": "分析临床证据"},
                        {"id": "任务3", "description": "生成专业回答"}
                    ],
                    "focus_areas": ["临床指南", "医学文献"]
                },
                "next_step": "executor",
                "current_task": "规划任务（失败）"
            }

class ExecutorNode:
    """任务执行器"""
    def __init__(self, retriever_agent, evidence_grader_agent, synthesizer_agent, fact_checker_agent, hypothesis_validator_agent):
        self.retriever_agent = retriever_agent
        self.evidence_grader_agent = evidence_grader_agent
        self.synthesizer_agent = synthesizer_agent
        self.fact_checker_agent = fact_checker_agent
        self.hypothesis_validator_agent = hypothesis_validator_agent
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """执行子任务"""
        intermediate_results = {}
        evidence = []
        graded_evidence = []
        
        try:
            logger.info(f"开始执行任务: {state.user_query}")
            
            # 1. 检索证据
            logger.info("步骤1: 开始检索证据")
            evidence = self.retriever_agent.retrieve(state.user_query)
            intermediate_results["retrieval"] = evidence
            logger.info(f"步骤1: 检索到 {len(evidence)} 条证据")
            
            # 2. 证据分级
            logger.info("步骤2: 开始证据分级")
            if evidence:
                graded_evidence = self.evidence_grader_agent.grade(evidence)
                intermediate_results["grading"] = graded_evidence
                logger.info(f"步骤2: 分级完成，保留 {len(graded_evidence)} 条高质量证据")
            else:
                logger.warning("步骤2: 没有证据需要分级")
            
            # 3. 药物重定位分析（如果问题涉及药物）
            # 改进药物相关检测逻辑，增加对常见药物名称的检测
            drug_keywords = ["药物", "药", "drug", "medicine", "treatment", "Metformin", "Aspirin", "Ibuprofen", "Simvastatin", "Citalopram"]
            drug_related = any(keyword.lower() in state.user_query.lower() for keyword in drug_keywords)
            if drug_related:
                logger.info("步骤3: 开始药物重定位分析")
                try:
                    from drug_repositioning_model import generate_repositioning_hypotheses
                    # 提取药物名称
                    import re
                    drug_names = re.findall(r'[\u4e00-\u9fa5]{2,10}|[A-Za-z]+', state.user_query)
                    if drug_names:
                        drug_name = drug_names[0]
                        logger.info(f"步骤3: 提取到药物名称: {drug_name}")
                        repositioning_results = generate_repositioning_hypotheses(drug_name, top_k=3)
                        if repositioning_results:
                            intermediate_results["drug_repositioning"] = repositioning_results
                            logger.info(f"步骤3: 生成 {len(repositioning_results)} 个重定位假设")
                            
                            # 对每个重定位结果进行假设验证
                            validated_results = []
                            for result in repositioning_results:
                                disease = result.get('disease', '')
                                if disease:
                                    logger.info(f"步骤3: 验证假设: {drug_name} → {disease}")
                                    validated_hypothesis = self.hypothesis_validator_agent.validate_hypotheses(drug_name, disease)
                                    if validated_hypothesis:
                                        validated_results.append(validated_hypothesis)
                            
                            if validated_results:
                                intermediate_results["validated_hypotheses"] = validated_results
                                logger.info(f"步骤3: 验证完成 {len(validated_results)} 个假设")
                            
                            # 将药物重定位结果添加到证据中
                            for i, result in enumerate(repositioning_results):
                                evidence.append({
                                    "id": f"repositioning_{i}",
                                    "source": "drug_repositioning",
                                    "content": f"潜在新适应症: {result.get('disease', '')}, 得分: {result.get('score', '')}",
                                    "metadata": result
                                })
                        else:
                            logger.warning("步骤3: 未生成重定位假设")
                    else:
                        logger.warning("步骤3: 未提取到药物名称")
                except Exception as e:
                    logger.error(f"步骤3: 药物重定位分析失败: {e}")
            else:
                logger.info("步骤3: 问题不涉及药物，跳过药物重定位分析")
            
            # 4. 生成回答草案
            logger.info("步骤4: 开始生成回答草案")
            if graded_evidence:
                draft_answer = self.synthesizer_agent.synthesize(graded_evidence, state.user_query)
                intermediate_results["synthesis"] = draft_answer
                logger.info("步骤4: 回答草案生成完成")
            else:
                draft_answer = "根据现有信息，无法生成详细回答。"
                intermediate_results["synthesis"] = draft_answer
                logger.warning("步骤4: 没有高质量证据，生成默认回答")
            
            # 5. 事实核查
            logger.info("步骤5: 开始事实核查")
            final_answer = self.fact_checker_agent.check(draft_answer, graded_evidence)
            intermediate_results["fact_checking"] = final_answer
            logger.info("步骤5: 事实核查完成")
            
            logger.info("执行任务完成")
            
            return {
                "intermediate_results": intermediate_results,
                "evidence": evidence,
                "graded_evidence": graded_evidence,
                "next_step": "finisher",
                "current_task": "执行任务"
            }
        except Exception as e:
            logger.error(f"执行任务失败: {e}")
            # 错误处理：返回已有的结果，确保系统能够继续运行
            return {
                "intermediate_results": intermediate_results,
                "evidence": evidence,
                "graded_evidence": graded_evidence,
                "next_step": "finisher",
                "current_task": "执行任务（部分失败）"
            }

class FinisherNode:
    """结果整合器"""
    def __init__(self):
        pass
    
    def finish(self, state: AgentState) -> Dict[str, Any]:
        """整合结果，形成最终回答"""
        try:
            logger.info("开始整合结果")
            
            final_answer = state.intermediate_results.get("fact_checking", "无法生成回答")
            
            # 构建可解释性信息
            steps = [
                "规划任务: 拆解用户问题为子任务",
                "检索证据: 从知识库和外部来源获取相关信息",
                "证据分级: 对证据进行质量评估",
            ]
            
            # 如果有药物重定位分析结果，添加到步骤中
            if "drug_repositioning" in state.intermediate_results:
                steps.append("药物重定位分析: 分析药物的潜在新适应症")
                # 如果有假设验证结果，添加验证步骤
                if "validated_hypotheses" in state.intermediate_results:
                    steps.append("假设验证: 验证药物重定位假设的合理性")
            
            steps.extend([
                "生成回答: 基于高质量证据生成回答草案",
                "事实核查: 验证回答中的每个断言"
            ])
            
            explanation = {
                "steps": steps,
                "evidence_count": len(state.graded_evidence),
                "evidence_grades": [e.get("grade", "") for e in state.graded_evidence]
            }
            
            # 构建最终回答
            final_response = f"""
{final_answer}

---

**推理过程**:
{"\n".join(explanation["steps"])}

**证据统计**:
- 总证据数: {explanation["evidence_count"]}
- 证据等级: {', '.join(explanation["evidence_grades"])}
"""
            
            # 如果有药物重定位分析结果，添加到最终回答中
            if "drug_repositioning" in state.intermediate_results:
                repositioning_results = state.intermediate_results["drug_repositioning"]
                if repositioning_results:
                    final_response += "\n\n**药物重定位分析**:\n"
                    for result in repositioning_results:
                        final_response += f"- 潜在新适应症: {result.get('disease', '')}, 得分: {result.get('score', '')}\n"
                        if 'pathway' in result:
                            final_response += f"  证据路径: {result.get('pathway', '')}\n"
            
            # 如果有假设验证结果，添加到最终回答中
            if "validated_hypotheses" in state.intermediate_results:
                validated_hypotheses = state.intermediate_results["validated_hypotheses"]
                if validated_hypotheses:
                    final_response += "\n\n**假设验证结果**:\n"
                    for i, hypothesis in enumerate(validated_hypotheses, 1):
                        validation = hypothesis.get("validation", {})
                        drug = hypothesis.get("drug", "")
                        disease = hypothesis.get("disease", "")
                        total_score = validation.get("total_score", 0)
                        literature_score = validation.get("literature_score", 0)
                        binding_score = validation.get("binding_score", 0)
                        pathway_score = validation.get("pathway_score", 0)
                        
                        final_response += f"\n{i}. {drug} → {disease}\n"
                        final_response += f"  综合得分: {total_score:.4f}\n"
                        final_response += f"  文献证据得分: {literature_score:.4f}\n"
                        final_response += f"  结合亲和力得分: {binding_score:.4f}\n"
                        final_response += f"  通路合理性得分: {pathway_score:.4f}\n"
                        
                        # 添加支持文献
                        evidence = validation.get("evidence", [])
                        if evidence:
                            final_response += "  支持文献:\n"
                            for j, item in enumerate(evidence, 1):
                                final_response += f"    {j}. {item.get('title', '未知标题')} (PMID: {item.get('pmid', '未知')})\n"
            
            logger.info("结果整合完成")
            
            return {
                "final_answer": final_response,
                "next_step": None,
                "current_task": "完成任务"
            }
        except Exception as e:
            logger.error(f"结果整合失败: {e}")
            # 错误处理：返回简单的最终回答
            return {
                "final_answer": f"无法生成完整回答: {str(e)}",
                "next_step": None,
                "current_task": "完成任务（失败）"
            }

# ====================== 下游专家Agent ======================
class RetrieverAgent:
    """检索专家"""
    def __init__(self, vector_store=None):
        self.vector_store = vector_store
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """检索内部知识库、图谱和外部文献"""
        # 检查缓存
        cache_key = f"retrieve_{query}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"使用缓存的检索结果: {query}")
            return cached_result
        
        evidence = []
        
        # 1. 内部知识库检索
        if self.vector_store:
            try:
                from main import CrossModalRetriever
                retriever = CrossModalRetriever(self.vector_store)
                blocks = retriever.retrieve(query, top_k=5)
                for i, block in enumerate(blocks):
                    evidence.append({
                        "id": f"internal_{i}",
                        "source": "internal_knowledge_base",
                        "content": block.content,
                        "metadata": block.metadata
                    })
                logger.info(f"内部知识库检索到 {len(evidence)} 条证据")
            except Exception as e:
                logger.error(f"内部知识库检索失败: {e}")
        
        # 2. 外部API检索（PubMed）
        try:
            from main import search_pubmed_abstracts
            pubmed_results = search_pubmed_abstracts(query, max_results=3)
            for i, result in enumerate(pubmed_results):
                evidence.append({
                    "id": f"pubmed_{i}",
                    "source": "pubmed",
                    "content": result.get("abstract", ""),
                    "metadata": {"title": result.get("title", ""), "pmid": result.get("pmid", "")}
                })
            logger.info(f"PubMed检索到 {len(pubmed_results)} 条证据")
        except Exception as e:
            logger.error(f"PubMed检索失败: {e}")
        
        # 3. 知识图谱检索（如果可用）
        try:
            from cdss_app import search_medical_knowledge_graph
            kg_results = search_medical_knowledge_graph(query)
            for i, result in enumerate(kg_results[:3]):  # 限制返回数量
                evidence.append({
                    "id": f"kg_{i}",
                    "source": "knowledge_graph",
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {})
                })
            logger.info(f"知识图谱检索到 {len(kg_results[:3])} 条证据")
        except Exception as e:
            logger.error(f"知识图谱检索失败: {e}")
        
        # 缓存结果
        cache.set(cache_key, evidence)
        logger.info(f"检索完成，获取 {len(evidence)} 条证据")
        
        return evidence
    


class HypothesisValidatorAgent:
    """假设验证专家"""
    def __init__(self, vector_store=None):
        self.vector_store = vector_store
    
    def validate_hypotheses(self, drug, disease):
        """验证药物重定位假设"""
        # 检查缓存
        cache_key = f"validate_{drug}_{disease}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"使用缓存的假设验证结果: {drug} → {disease}")
            return cached_result
        
        try:
            logger.info(f"开始验证假设: {drug} → {disease}")
            from hypothesis_validation import HypothesisValidator
            validator = HypothesisValidator()
            
            # 创建假设
            hypothesis = [{
                "drug": drug,
                "disease": disease,
                "score": 1.0,
                "evidence_paths": []
            }]
            
            # 验证假设
            validated_hypotheses = validator.validate_and_rank_hypotheses(hypothesis, vector_store=self.vector_store)
            validator.close()
            
            result = validated_hypotheses[0] if validated_hypotheses else None
            
            # 缓存结果
            if result:
                cache.set(cache_key, result)
                logger.info(f"假设验证完成: {drug} → {disease}, 得分: {result.get('validation', {}).get('total_score', 0):.4f}")
            
            return result
        except Exception as e:
            logger.error(f"假设验证失败: {e}")
            return None

class EvidenceGraderAgent:
    """证据分级专家"""
    def __init__(self, llm):
        self.llm = llm
    
    def grade(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对检索到的每条证据进行等级标注"""
        graded = []
        
        for item in evidence:
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
        
        # 过滤低质量证据
        filtered = [e for e in graded if e.get("confidence", 0) >= 0.5]
        return filtered

class SynthesizerAgent:
    """综合撰写专家"""
    def __init__(self, llm):
        self.llm = llm
    
    def synthesize(self, evidence: List[Dict[str, Any]], query: str) -> str:
        """基于高质量证据生成回答草案"""
        # 构建证据上下文
        evidence_context = ""
        for item in evidence:
            evidence_context += f"【证据】{item['content']}（等级：{item['grade']}，来源：{item['source']}）\n"
        
        prompt = f"""
你是一个生物医药专家，根据提供的证据生成专业、准确的回答。

用户问题：{query}

【证据】
{evidence_context}

请基于以上证据，生成一个全面、准确的回答，引用相关证据支持你的观点。
"""
        
        response = self.llm.invoke(prompt)
        return response

class FactCheckerAgent:
    """事实核查专家"""
    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store
    
    def check(self, draft_answer: str, evidence: List[Dict[str, Any]]) -> str:
        """将回答草案拆解为原子断言，逐一验证"""
        # 提取回答中的断言
        prompt = f"""
你是一个医疗事实核查员，负责从医学回答中提取具体的断言。

回答内容：{draft_answer}

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
            claims = [draft_answer]
        
        # 对每个断言进行验证
        validated_claims = []
        for claim in claims:
            # 检索相关证据
            evidence_content = ""
            for e in evidence:
                if claim.lower() in e['content'].lower():
                    evidence_content += e['content'] + "\n"
            
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
        
        # 生成最终回答
        final_answer = draft_answer
        
        # 如果有低置信度的断言，标记出来
        low_confidence_claims = [c for c in validated_claims if c.get("confidence", 0) < 0.7]
        if low_confidence_claims:
            final_answer += "\n\n**注意**：以下内容的证据支持不足，仅供参考：\n"
            for claim in low_confidence_claims:
                final_answer += f"- {claim['claim']}\n"
        
        return final_answer

# ====================== 构建LangGraph ======================
def build_agent_graph(vector_store=None) -> Any:
    """构建多智能体协作图"""
    logger.info("开始构建多智能体协作图")
    
    llm = get_llm()
    logger.info("初始化LLM完成")
    
    # 初始化智能体
    retriever_agent = RetrieverAgent(vector_store)
    evidence_grader_agent = EvidenceGraderAgent(llm)
    synthesizer_agent = SynthesizerAgent(llm)
    fact_checker_agent = FactCheckerAgent(llm, vector_store)
    hypothesis_validator_agent = HypothesisValidatorAgent(vector_store)
    logger.info("初始化智能体完成")
    
    # 初始化中央控制单元
    supervisor = SupervisorNode()
    planner = PlannerNode(llm)
    executor = ExecutorNode(retriever_agent, evidence_grader_agent, synthesizer_agent, fact_checker_agent, hypothesis_validator_agent)
    finisher = FinisherNode()
    logger.info("初始化中央控制单元完成")
    
    # 定义状态图
    graph = StateGraph(AgentState)
    logger.info("创建状态图完成")
    
    # 添加节点
    graph.add_node("supervisor", supervisor.route)
    graph.add_node("planner", planner.plan)
    graph.add_node("executor", executor.execute)
    graph.add_node("finisher", finisher.finish)
    logger.info("添加节点完成")
    
    # 添加边
    graph.set_entry_point("supervisor")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("executor", "supervisor")
    graph.add_edge("finisher", END)
    logger.info("添加边完成")
    
    # 编译图
    compiled = graph.compile()
    logger.info("编译图完成")
    
    return compiled

# ====================== 全局变量 ======================
# 全局向量存储实例，避免重复初始化
_global_vector_store = None

# ====================== 工具函数 ======================
def get_vector_store():
    """获取向量存储实例，避免重复初始化"""
    global _global_vector_store
    if _global_vector_store is None:
        try:
            from main import MultimodalVectorStore
            # 检查是否存在已构建的知识库
            import os
            if os.path.exists("multimodal_rag_db") and os.path.isdir("multimodal_rag_db"):
                # 加载已存在的知识库
                _global_vector_store = MultimodalVectorStore()
                _global_vector_store.initialize()
                print("已加载现有知识库")
            else:
                print("未找到现有知识库，使用空向量存储")
        except Exception as e:
            print(f"初始化向量存储失败: {e}")
    return _global_vector_store

# ====================== FastAPI服务 ======================
app = FastAPI(title="生物医药多智能体系统")

@app.post("/v1/chat")
async def chat(request: Request):
    """流式聊天接口"""
    data = await request.json()
    user_query = data.get("query")
    session_id = data.get("session_id", str(uuid.uuid4()))
    
    # 获取向量存储实例
    vector_store = get_vector_store()
    
    # 构建图
    graph = build_agent_graph(vector_store)
    
    # 初始化状态
    initial_state = AgentState(
        messages=[HumanMessage(content=user_query)],
        user_query=user_query,
        session_id=session_id
    )
    
    # 流式执行
    async def stream_generator():
        # 发送初始状态
        yield "data: {\"type\": \"status\", \"message\": \"开始处理请求...\"}\n\n"
        await asyncio.sleep(0.5)
        
        try:
            logger.info(f"开始处理请求: {user_query}")
            
            # 执行图（使用同步方法）
            logger.info("开始执行LangGraph")
            
            # 直接执行整个流程
            result = graph.invoke(initial_state)
            logger.info(f"LangGraph执行完成，结果: {result}")
            
            # 检查结果
            if result.get("final_answer") is None:
                # 如果没有最终回答，手动执行各个节点
                logger.info("没有最终回答，手动执行各个节点")
                
                # 1. 执行规划
                planner = PlannerNode(get_llm())
                plan_result = planner.plan(initial_state)
                logger.info(f"规划完成，结果: {plan_result}")
                
                # 2. 执行任务
                retriever_agent = RetrieverAgent(vector_store)
                evidence_grader_agent = EvidenceGraderAgent(get_llm())
                synthesizer_agent = SynthesizerAgent(get_llm())
                fact_checker_agent = FactCheckerAgent(get_llm(), vector_store)
                hypothesis_validator_agent = HypothesisValidatorAgent(vector_store)
                
                executor = ExecutorNode(retriever_agent, evidence_grader_agent, synthesizer_agent, fact_checker_agent, hypothesis_validator_agent)
                
                # 创建执行状态
                executor_state = AgentState(
                    messages=initial_state.messages,
                    user_query=initial_state.user_query,
                    session_id=initial_state.session_id,
                    plan=plan_result.get("plan")
                )
                executor_result = executor.execute(executor_state)
                logger.info(f"执行完成，结果: {executor_result}")
                
                # 3. 完成任务
                finisher = FinisherNode()
                
                # 创建完成状态
                finisher_state = AgentState(
                    messages=initial_state.messages,
                    user_query=initial_state.user_query,
                    session_id=initial_state.session_id,
                    plan=plan_result.get("plan"),
                    intermediate_results=executor_result.get("intermediate_results", {}),
                    evidence=executor_result.get("evidence", []),
                    graded_evidence=executor_result.get("graded_evidence", [])
                )
                finisher_result = finisher.finish(finisher_state)
                logger.info(f"完成任务，结果: {finisher_result}")
                
                # 使用完成结果
                result = finisher_result
            
            # 提取当前任务状态和最终回答
            current_task = "执行任务"
            final_answer = result.get("final_answer", "无法生成回答")
            
            # 确保final_answer是字符串
            if not isinstance(final_answer, str):
                final_answer = "无法生成回答"
            
            # 生成流式输出
            yield f"data: {{\"type\": \"status\", \"message\": \"{current_task}\"}}\n\n"
            await asyncio.sleep(0.5)
            
            yield f"data: {{\"type\": \"answer\", \"message\": \"{final_answer.replace('\n', '\\n')}\"}}\n\n"
            await asyncio.sleep(0.5)
            
            logger.info("请求处理完成")
        except Exception as e:
            logger.error(f"处理请求时发生错误: {e}")
            # 发送错误信息
            yield f"data: {{\"type\": \"error\", \"message\": \"处理请求时发生错误: {str(e)}\"}}\n\n"
        
        # 结束信号
        yield "data: {\"type\": \"end\"}\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# ====================== Streamlit前端改造 ======================
def streamlit_frontend():
    """Streamlit前端"""
    st.set_page_config(page_title="生物医药多智能体系统", layout="wide")
    st.title("🧬 生物医药多智能体系统")
    
    # 用户输入
    user_query = st.text_input("输入临床问题", placeholder="例如：对于eGFR<30的2型糖尿病患者，是否可以使用SGLT-2抑制剂？")
    
    if st.button("提交"):
        if not user_query:
            st.error("请输入问题")
            return
        
        # 显示状态
        status_container = st.empty()
        answer_container = st.empty()
        expander_container = st.empty()
        
        # 调用FastAPI接口
        url = "http://localhost:8000/v1/chat"
        data = {"query": user_query}
        
        try:
            response = requests.post(url, json=data, stream=True)
            
            final_answer = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_part = line[6:]
                        try:
                            json_data = json.loads(data_part)
                            if json_data.get("type") == "status":
                                status_container.info(json_data.get("message"))
                            elif json_data.get("type") == "answer":
                                final_answer = json_data.get("message")
                                # 显示回答
                                answer_container.markdown(final_answer)
                        except Exception as e:
                            logger.error(f"解析响应失败: {e}")
            
            # 处理完所有响应后，检查是否包含假设验证结果
            if final_answer and "假设验证结果" in final_answer:
                # 添加一个展开/折叠的详细信息
                with expander_container.expander("查看详细假设验证结果"):
                    st.markdown(final_answer)
        except Exception as e:
            logger.error(f"调用API失败: {e}")
            st.error(f"调用API失败: {e}")

# ====================== 主函数 ======================
def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="生物医药多智能体系统")
    parser.add_argument("--mode", choices=["api", "frontend"], default="api", help="运行模式")
    parser.add_argument("--host", default="0.0.0.0", help="API主机")
    parser.add_argument("--port", type=int, default=8000, help="API端口")
    args = parser.parse_args()
    
    if args.mode == "api":
        # 启动FastAPI服务
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # 启动Streamlit前端
        streamlit_frontend()

if __name__ == "__main__":
    import asyncio
    main()

# Streamlit入口点
if 'streamlit' in sys.modules:
    streamlit_frontend()
