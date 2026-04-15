#!/usr/bin/env python3
"""
诊断推理引擎
- 基于Neo4j Cypher查询的诊断推理
- 接收实体列表，返回候选疾病及证据路径
- 支持规则推理和相似病例检索
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any

class DiagnosisEngine:
    def __init__(self, uri, user, password):
        """初始化诊断推理引擎"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # 定义诊断规则
        self.diagnosis_rules = {
            '社区获得性肺炎': {
                'required_symptoms': ['S001', 'S002'],  # 发热、咳嗽
                'required_tests': ['T001', 'T002'],     # 血常规、胸部CT
                'evidence': '中国成人社区获得性肺炎诊断和治疗指南(2016版)',
                'evidence_level': 'A'
            },
            '慢性阻塞性肺疾病': {
                'required_symptoms': ['S002', 'S003', 'S004'],  # 咳嗽、咳痰、呼吸困难
                'required_tests': ['T003'],                    # 血气分析
                'evidence': '慢性阻塞性肺疾病全球倡议(GOLD)',
                'evidence_level': 'A'
            },
            '哮喘': {
                'required_symptoms': ['S002', 'S004'],  # 咳嗽、呼吸困难
                'required_tests': [],
                'evidence': '全球哮喘防治倡议(GINA)',
                'evidence_level': 'A'
            }
        }
    
    def close(self):
        """关闭Neo4j连接"""
        self.driver.close()
    
    def diagnose(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于实体列表进行诊断"""
        # 提取症状和检查实体
        symptoms = [e['concept_id'] for e in entities if e['type'] in ['SYMPTOM', 'symptom']]
        tests = [e['concept_id'] for e in entities if e['type'] in ['TEST', 'test']]
        signs = [e['concept_id'] for e in entities if e['type'] in ['SIGN', 'sign']]
        
        # 执行规则推理
        rule_based_diagnoses = []
        try:
            rule_based_diagnoses = self._rule_based_diagnosis(symptoms, tests)
        except Exception as e:
            print(f"规则推理失败: {e}")
        
        # 执行相似病例检索
        similarity_diagnoses = []
        try:
            similarity_diagnoses = self._similarity_based_diagnosis(symptoms, tests, signs)
        except Exception as e:
            print(f"相似病例检索失败: {e}")
        
        # 合并诊断结果
        all_diagnoses = self._merge_diagnoses(rule_based_diagnoses, similarity_diagnoses)
        
        # 为每个诊断生成证据路径
        for diagnosis in all_diagnoses:
            try:
                diagnosis['evidence_path'] = self._generate_evidence_path(diagnosis['disease_id'])
            except Exception as e:
                print(f"生成证据路径失败: {e}")
                diagnosis['evidence_path'] = []
        
        return all_diagnoses
    
    def _rule_based_diagnosis(self, symptoms: List[str], tests: List[str]) -> List[Dict[str, Any]]:
        """基于规则的诊断"""
        diagnoses = []
        
        for disease_name, rule in self.diagnosis_rules.items():
            # 检查是否满足所有必需的症状
            required_symptoms_met = all(s in symptoms for s in rule['required_symptoms'])
            # 检查是否满足所有必需的检查
            required_tests_met = all(t in tests for t in rule['required_tests'])
            
            if required_symptoms_met and required_tests_met:
                # 计算匹配分数
                total_required = len(rule['required_symptoms']) + len(rule['required_tests'])
                matched = (len([s for s in rule['required_symptoms'] if s in symptoms]) + 
                         len([t for t in rule['required_tests'] if t in tests]))
                score = matched / total_required if total_required > 0 else 0
                
                # 获取疾病ID
                disease_id = self._get_disease_id(disease_name)
                
                diagnoses.append({
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'score': score,
                    'method': 'rule-based',
                    'evidence': rule['evidence'],
                    'evidence_level': rule['evidence_level']
                })
        
        # 按分数排序
        diagnoses.sort(key=lambda x: x['score'], reverse=True)
        return diagnoses
    
    def _similarity_based_diagnosis(self, symptoms: List[str], tests: List[str], signs: List[str]) -> List[Dict[str, Any]]:
        """基于相似性的诊断"""
        diagnoses = []
        
        with self.driver.session() as session:
            # 构建Cypher查询
            query = """
            MATCH (d:Disease)
            OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
            OPTIONAL MATCH (d)-[:REQUIRES_TEST]->(t:Test)
            WITH d, COLLECT(DISTINCT s.id) AS disease_symptoms, COLLECT(DISTINCT t.id) AS disease_tests
            RETURN d.id AS disease_id, d.name AS disease_name, disease_symptoms, disease_tests
            """
            
            results = session.run(query)
            
            for record in results:
                disease_id = record['disease_id']
                disease_name = record['disease_name']
                disease_symptoms = record['disease_symptoms']
                disease_tests = record['disease_tests']
                
                # 计算症状匹配度
                symptom_match = len([s for s in disease_symptoms if s in symptoms])
                symptom_score = symptom_match / len(disease_symptoms) if disease_symptoms else 0
                
                # 计算检查匹配度
                test_match = len([t for t in disease_tests if t in tests])
                test_score = test_match / len(disease_tests) if disease_tests else 0
                
                # 计算总匹配度
                total_score = (symptom_score * 0.7) + (test_score * 0.3)
                
                if total_score > 0.3:  # 阈值
                    diagnoses.append({
                        'disease_id': disease_id,
                        'disease_name': disease_name,
                        'score': total_score,
                        'method': 'similarity-based',
                        'evidence': '基于症状和检查的相似性匹配',
                        'evidence_level': 'C'
                    })
        
        # 按分数排序
        diagnoses.sort(key=lambda x: x['score'], reverse=True)
        return diagnoses
    
    def _merge_diagnoses(self, rule_based: List[Dict[str, Any]], similarity_based: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并诊断结果"""
        merged = {}
        
        # 先添加规则-based诊断
        for diagnosis in rule_based:
            merged[diagnosis['disease_id']] = diagnosis
        
        # 再添加相似性-based诊断，如果分数更高
        for diagnosis in similarity_based:
            disease_id = diagnosis['disease_id']
            if disease_id not in merged or diagnosis['score'] > merged[disease_id]['score']:
                merged[disease_id] = diagnosis
        
        # 转换为列表并按分数排序
        result = list(merged.values())
        result.sort(key=lambda x: x['score'], reverse=True)
        
        return result
    
    def _generate_evidence_path(self, disease_id: str) -> List[Dict[str, Any]]:
        """生成证据路径"""
        evidence_path = []
        
        with self.driver.session() as session:
            # 查询疾病的症状
            symptom_query = """
            MATCH (d:Disease {id: $disease_id})-[r:HAS_SYMPTOM]->(s:Symptom)
            RETURN s.name AS name, r.evidence AS evidence, r.evidence_level AS evidence_level
            """
            symptom_results = session.run(symptom_query, disease_id=disease_id)
            
            for record in symptom_results:
                evidence_path.append({
                    'type': 'symptom',
                    'name': record['name'],
                    'evidence': record['evidence'],
                    'evidence_level': record['evidence_level']
                })
            
            # 查询疾病的检查
            test_query = """
            MATCH (d:Disease {id: $disease_id})-[r:REQUIRES_TEST]->(t:Test)
            RETURN t.name AS name, r.evidence AS evidence, r.evidence_level AS evidence_level
            """
            test_results = session.run(test_query, disease_id=disease_id)
            
            for record in test_results:
                evidence_path.append({
                    'type': 'test',
                    'name': record['name'],
                    'evidence': record['evidence'],
                    'evidence_level': record['evidence_level']
                })
            
            # 查询疾病的治疗
            treatment_query = """
            MATCH (d:Disease {id: $disease_id})-[r]->(t)
            WHERE type(r) IN ['TREATED_WITH_MEDICATION', 'TREATED_WITH']
            RETURN t.name AS name, type(r) AS relation_type, r.evidence AS evidence, r.evidence_level AS evidence_level
            """
            treatment_results = session.run(treatment_query, disease_id=disease_id)
            
            for record in treatment_results:
                evidence_path.append({
                    'type': 'treatment',
                    'name': record['name'],
                    'relation_type': record['relation_type'],
                    'evidence': record['evidence'],
                    'evidence_level': record['evidence_level']
                })
        
        return evidence_path
    
    def _get_disease_id(self, disease_name: str) -> str:
        """根据疾病名称获取疾病ID"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (d:Disease {name: $disease_name})
                RETURN d.id AS disease_id
                """
                result = session.run(query, disease_name=disease_name)
                record = result.single()
                if record:
                    return record['disease_id']
                else:
                    # 如果没有找到，返回默认ID
                    return f"D{hash(disease_name) % 1000}"
        except Exception as e:
            # 如果连接失败，返回默认ID
            print(f"获取疾病ID失败: {e}")
            return f"D{hash(disease_name) % 1000}"
    
    def get_diagnosis_with_evidence(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取诊断结果和证据"""
        diagnoses = self.diagnose(entities)
        
        # 构建返回结果
        result = {
            'diagnoses': diagnoses,
            'top_diagnosis': diagnoses[0] if diagnoses else None,
            'summary': {
                'total_diagnoses': len(diagnoses),
                'confidence': diagnoses[0]['score'] if diagnoses else 0
            }
        }
        
        return result

def main():
    """主函数"""
    # Neo4j连接信息
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "Yjz61925!"
    
    # 创建诊断引擎实例
    engine = DiagnosisEngine(uri, user, password)
    
    try:
        # 示例实体列表（从患者信息抽取模块获取）
        entities = [
            {'text': '发热', 'type': 'SYMPTOM', 'concept_id': 'S001', 'value': '', 'start': 10, 'end': 12},
            {'text': '咳嗽', 'type': 'SYMPTOM', 'concept_id': 'S002', 'value': '', 'start': 13, 'end': 15},
            {'text': '血常规', 'type': 'TEST', 'concept_id': 'T001', 'value': 'WBC 14.2×10^9/L', 'start': 45, 'end': 48},
            {'text': '胸部CT', 'type': 'TEST', 'concept_id': 'T002', 'value': '右下肺斑片状高密度影', 'start': 65, 'end': 68}
        ]
        
        # 获取诊断结果
        result = engine.get_diagnosis_with_evidence(entities)
        
        # 输出结果
        print("诊断结果:")
        for i, diagnosis in enumerate(result['diagnoses']):
            print(f"\n{i+1}. {diagnosis['disease_name']} (置信度: {diagnosis['score']:.2f})")
            print(f"   方法: {diagnosis['method']}")
            print(f"   证据: {diagnosis['evidence']} (等级: {diagnosis['evidence_level']})")
            print("   证据路径:")
            for item in diagnosis['evidence_path']:
                print(f"     - {item['type']}: {item['name']} (证据: {item['evidence']}, 等级: {item['evidence_level']})")
                
    finally:
        # 关闭连接
        engine.close()

if __name__ == "__main__":
    main()
