#!/usr/bin/env python3
"""
幻觉控制机制
- 结构化JSON输出
- 两阶段重排
- 知识库回扫校验
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass, field

import guidance
from langchain_community.llms import Tongyi

# ====================== 环境配置 ======================
import os
from dotenv import load_dotenv
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "请替换为你的通义千问APIKey")

# ====================== LLM初始化 ======================
def get_llm(temperature=0.3, max_tokens=1500):
    return Tongyi(
        model_name="qwen-max",
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens
    )

# ====================== 结构化输出模板 ======================
def structured_output_template():
    """创建结构化输出模板"""
    return guidance("""
你是一个生物医药专家，需要根据提供的信息生成结构化的回答。

用户问题：{{user_query}}

相关信息：{{context}}

请按照以下JSON格式输出你的回答：
{
  "claims": [
    {
      "claim": "具体断言",
      "evidence_source": "证据来源",
      "evidence_grade": "证据等级（1a-5）",
      "confidence": 0.0-1.0
    }
  ],
  "final_answer": "综合回答",
  "overall_confidence": 0.0-1.0
}
""")

# ====================== 两阶段重排 ======================
class TwoStageReranker:
    def __init__(self, llm):
        self.llm = llm
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """两阶段重排"""
        # 第一阶段：基于相关性的初步排序
        initial_ranked = self._initial_rank(query, documents)
        
        # 第二阶段：基于医学相关性和证据质量的深度排序
        final_ranked = self._deep_rank(query, initial_ranked, top_k)
        
        return final_ranked
    
    def _initial_rank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """第一阶段：基于相关性的初步排序"""
        # 这里可以使用更复杂的算法，这里简化处理
        # 假设documents已经有score字段
        return sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
    
    def _deep_rank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """第二阶段：基于医学相关性和证据质量的深度排序"""
        # 对每个文档进行医学相关性评估
        for doc in documents:
            prompt = f"""
请评估以下文档与医学问题的相关性和证据质量：

问题：{query}

文档：{doc['content']}

请从以下方面评估：
1. 医学相关性（0-1）
2. 证据质量（0-1）
3. 整体评分（0-1）

输出格式：
{
  "medical_relevance": 0.0,
  "evidence_quality": 0.0,
  "overall_score": 0.0
}
"""
            
            response = self.llm.invoke(prompt)
            try:
                scores = json.loads(response)
                doc["medical_relevance"] = scores.get("medical_relevance", 0)
                doc["evidence_quality"] = scores.get("evidence_quality", 0)
                doc["overall_score"] = scores.get("overall_score", 0)
            except:
                doc["medical_relevance"] = 0.5
                doc["evidence_quality"] = 0.5
                doc["overall_score"] = 0.5
        
        # 按整体评分排序并返回前top_k个
        return sorted(documents, key=lambda x: x.get("overall_score", 0), reverse=True)[:top_k]

# ====================== 知识库回扫校验 ======================
class KnowledgeBaseValidator:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def validate_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对断言进行知识库回扫校验"""
        if not self.vector_store:
            return claims
        
        from main import CrossModalRetriever
        retriever = CrossModalRetriever(self.vector_store)
        
        for claim in claims:
            # 检索相关知识
            blocks = retriever.retrieve(claim["claim"], top_k=3)
            
            # 评估断言与知识的一致性
            if blocks:
                evidence_content = "\n".join([block.content for block in blocks])
                claim["validation_evidence"] = evidence_content[:500]  # 限制长度
                claim["consistency_score"] = self._evaluate_consistency(claim["claim"], evidence_content)
            else:
                claim["validation_evidence"] = ""
                claim["consistency_score"] = 0.5
        
        return claims
    
    def _evaluate_consistency(self, claim: str, evidence: str) -> float:
        """评估断言与证据的一致性"""
        llm = get_llm()
        prompt = f"""
请评估以下断言与证据的一致性：

断言：{claim}

证据：{evidence}

一致性评分（0-1）：
"""
        
        response = llm.invoke(prompt)
        try:
            score = float(response.strip())
            return max(0, min(1, score))
        except:
            return 0.5

# ====================== 主函数 ======================
def control_hallucination(query: str, context: str, vector_store=None) -> Dict[str, Any]:
    """控制幻觉，生成结构化输出"""
    llm = get_llm()
    
    # 使用结构化输出模板
    template = structured_output_template()
    result = template(user_query=query, context=context, llm=llm)
    
    # 解析结果
    try:
        output = json.loads(result.text)
    except:
        # 如果解析失败，返回默认结构
        output = {
            "claims": [
                {
                    "claim": query,
                    "evidence_source": "无",
                    "evidence_grade": "5",
                    "confidence": 0.5
                }
            ],
            "final_answer": "无法生成结构化回答",
            "overall_confidence": 0.5
        }
    
    # 知识库回扫校验
    if vector_store:
        validator = KnowledgeBaseValidator(vector_store)
        output["claims"] = validator.validate_claims(output["claims"])
        
        # 更新整体置信度
        if output["claims"]:
            consistency_scores = [c.get("consistency_score", 0.5) for c in output["claims"]]
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            output["overall_confidence"] = (output["overall_confidence"] + avg_consistency) / 2
    
    return output

if __name__ == "__main__":
    # 测试
    test_query = "对于eGFR<30的2型糖尿病患者，是否可以使用SGLT-2抑制剂？"
    test_context = "根据KDIGO指南，eGFR<30时不推荐使用SGLT-2抑制剂，建议改用胰岛素或GLP-1受体激动剂。"
    
    result = control_hallucination(test_query, test_context)
    print(json.dumps(result, ensure_ascii=False, indent=2))
