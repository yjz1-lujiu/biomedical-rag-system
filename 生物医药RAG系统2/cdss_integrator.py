#!/usr/bin/env python3
"""
临床决策支持系统集成模块
- 整合患者信息抽取、知识图谱推理和RAG系统
- 为诊断结果提供文献支持
- 生成最终的诊断建议
"""

import os
import json
from typing import List, Dict, Any
from patient_info_extractor import PatientInfoExtractor
from diagnosis_engine import DiagnosisEngine
from main import generate_response_with_memory, get_llm
from session_manager import SessionManager, ChromaChatMessageHistory

class CDSSIntegrator:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        """初始化CDSS集成器"""
        self.extractor = PatientInfoExtractor()
        try:
            self.diagnosis_engine = DiagnosisEngine(neo4j_uri, neo4j_user, neo4j_password)
            self.neo4j_available = True
        except Exception as e:
            print(f"Neo4j连接失败: {e}")
            self.diagnosis_engine = None
            self.neo4j_available = False
        self.session_manager = SessionManager()
    
    def close(self):
        """关闭资源"""
        if self.diagnosis_engine:
            self.diagnosis_engine.close()
    
    def process_clinical_text(self, clinical_text: str, vector_store=None) -> Dict[str, Any]:
        """处理临床文本，生成诊断建议"""
        # 1. 提取患者信息
        extraction_result = self.extractor.extract_and_map(clinical_text)
        entities = extraction_result['entities']
        
        # 2. 进行诊断推理
        if self.neo4j_available and self.diagnosis_engine:
            diagnosis_result = self.diagnosis_engine.get_diagnosis_with_evidence(entities)
        else:
            # Neo4j不可用时的默认诊断结果
            diagnosis_result = {
                'diagnoses': [],
                'top_diagnosis': None,
                'summary': {
                    'total_diagnoses': 0,
                    'confidence': 0
                }
            }
        
        # 3. 检索相关文献（如果有向量存储）
        literature_evidence = []
        if vector_store:
            literature_evidence = self._retrieve_literature_evidence(
                clinical_text, diagnosis_result, vector_store
            )
        
        # 4. 生成最终诊断建议
        final_advice = self._generate_final_advice(
            clinical_text, extraction_result, diagnosis_result, literature_evidence
        )
        
        # 5. 构建返回结果
        result = {
            'clinical_text': clinical_text,
            'extraction_result': extraction_result,
            'diagnosis_result': diagnosis_result,
            'literature_evidence': literature_evidence,
            'final_advice': final_advice,
            'neo4j_available': self.neo4j_available
        }
        
        return result
    
    def _retrieve_literature_evidence(self, clinical_text: str, diagnosis_result: Dict[str, Any], vector_store) -> List[Dict[str, Any]]:
        """检索相关文献证据"""
        literature_evidence = []
        
        # 获取主要诊断
        if diagnosis_result['top_diagnosis']:
            top_disease = diagnosis_result['top_diagnosis']['disease_name']
            
            # 构建查询
            query = f"{clinical_text} {top_disease} 诊断 治疗"
            
            # 创建临时会话历史
            temp_session_id = "temp_cdss_session"
            history = ChromaChatMessageHistory(temp_session_id)
            
            # 使用RAG系统检索文献
            llm = get_llm()
            try:
                stream_func, blocks = generate_response_with_memory(
                    query, vector_store, llm, history, self.session_manager
                )
                
                # 提取文献证据
                for block in blocks:
                    if block.block_type == "text":
                        literature_evidence.append({
                            'type': 'text',
                            'content': block.content,
                            'source': block.metadata.get('source_file', '未知')
                        })
            except Exception as e:
                print(f"检索文献失败: {e}")
        
        return literature_evidence[:3]  # 最多返回3条文献证据
    
    def _generate_final_advice(self, clinical_text: str, extraction_result: Dict[str, Any], 
                              diagnosis_result: Dict[str, Any], literature_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成最终诊断建议"""
        # 构建提示词
        prompt = f"""你是一个临床决策支持系统，需要基于以下信息生成诊断建议：

【临床文本】
{clinical_text}

【提取的实体】
{json.dumps(extraction_result['entities'], ensure_ascii=False)}

【诊断结果】
{json.dumps(diagnosis_result['diagnoses'], ensure_ascii=False)}

【文献证据】
{json.dumps(literature_evidence, ensure_ascii=False)}

请生成一份完整的诊断建议，包括：
1. 鉴别诊断列表（按可能性排序）
2. 每条诊断的证据路径
3. 基于文献的治疗建议
4. 进一步检查建议

诊断建议应该专业、准确，并且包含证据支持。"""
        
        # 使用大模型生成建议
        llm = get_llm()
        try:
            advice = llm.invoke(prompt)
        except Exception as e:
            advice = f"生成诊断建议失败: {e}"
        
        return {
            'advice': advice,
            'timestamp': os.path.getmtime(__file__)
        }

def main():
    """主函数"""
    # Neo4j连接信息
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "Yjz61925!"
    
    # 创建CDSS集成器
    cdss = CDSSIntegrator(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # 示例临床文本
        clinical_text = "患者，男，65岁，因\"发热、咳嗽、咳黄痰3天\"入院。查体：T 38.5℃，双肺可闻及湿啰音。血常规：WBC 14.2×10^9/L，中性粒细胞百分比85%。胸部CT示右下肺斑片状高密度影。"
        
        # 处理临床文本
        result = cdss.process_clinical_text(clinical_text)
        
        # 输出结果
        print("临床文本:")
        print(clinical_text)
        print("\n提取的实体:")
        print(json.dumps(result['extraction_result']['entities'], ensure_ascii=False, indent=2))
        print("\n诊断结果:")
        for i, diagnosis in enumerate(result['diagnosis_result']['diagnoses']):
            print(f"\n{i+1}. {diagnosis['disease_name']} (置信度: {diagnosis['score']:.2f})")
            print(f"   方法: {diagnosis['method']}")
            print(f"   证据: {diagnosis['evidence']} (等级: {diagnosis['evidence_level']})")
        print("\n最终诊断建议:")
        print(result['final_advice']['advice'])
        
    finally:
        # 关闭资源
        cdss.close()

if __name__ == "__main__":
    main()
