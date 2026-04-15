#!/usr/bin/env python3
"""
临床知识图谱构建脚本
- 从CSV文件导入疾病、症状、检查、药物、治疗之间的结构化关系
- 使用Neo4j作为图数据库
- 支持UMLS子集数据格式
"""

import os
import csv
import json
from neo4j import GraphDatabase

class ClinicalKnowledgeGraph:
    def __init__(self, uri, user, password):
        """初始化Neo4j连接"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """关闭Neo4j连接"""
        self.driver.close()
    
    def create_constraints(self):
        """创建唯一性约束"""
        with self.driver.session() as session:
            # 创建疾病节点的唯一性约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE")
            # 创建症状节点的唯一性约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.id IS UNIQUE")
            # 创建检查节点的唯一性约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Test) REQUIRE t.id IS UNIQUE")
            # 创建药物节点的唯一性约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medication) REQUIRE m.id IS UNIQUE")
            # 创建治疗节点的唯一性约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Treatment) REQUIRE r.id IS UNIQUE")
    
    def import_diseases(self, csv_file):
        """导入疾病数据"""
        with self.driver.session() as session:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    session.run(
                        """
                        MERGE (d:Disease {
                            id: $id,
                            name: $name,
                            description: $description,
                            severity: $severity
                        })
                        """,
                        id=row['id'],
                        name=row['name'],
                        description=row.get('description', ''),
                        severity=row.get('severity', '')
                    )
    
    def import_symptoms(self, csv_file):
        """导入症状数据"""
        with self.driver.session() as session:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    session.run(
                        """
                        MERGE (s:Symptom {
                            id: $id,
                            name: $name,
                            description: $description
                        })
                        """,
                        id=row['id'],
                        name=row['name'],
                        description=row.get('description', '')
                    )
    
    def import_tests(self, csv_file):
        """导入检查数据"""
        with self.driver.session() as session:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    session.run(
                        """
                        MERGE (t:Test {
                            id: $id,
                            name: $name,
                            description: $description,
                            normal_range: $normal_range
                        })
                        """,
                        id=row['id'],
                        name=row['name'],
                        description=row.get('description', ''),
                        normal_range=row.get('normal_range', '')
                    )
    
    def import_medications(self, csv_file):
        """导入药物数据"""
        with self.driver.session() as session:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    session.run(
                        """
                        MERGE (m:Medication {
                            id: $id,
                            name: $name,
                            description: $description,
                            dosage: $dosage,
                            side_effects: $side_effects
                        })
                        """,
                        id=row['id'],
                        name=row['name'],
                        description=row.get('description', ''),
                        dosage=row.get('dosage', ''),
                        side_effects=row.get('side_effects', '')
                    )
    
    def import_treatments(self, csv_file):
        """导入治疗数据"""
        with self.driver.session() as session:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    session.run(
                        """
                        MERGE (r:Treatment {
                            id: $id,
                            name: $name,
                            description: $description
                        })
                        """,
                        id=row['id'],
                        name=row['name'],
                        description=row.get('description', '')
                    )
    
    def import_relationships(self, csv_file):
        """导入关系数据"""
        with self.driver.session() as session:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source_id = row['source_id']
                    target_id = row['target_id']
                    relation_type = row['relation_type']
                    evidence = row.get('evidence', '')
                    evidence_level = row.get('evidence_level', '')
                    
                    # 根据关系类型创建不同的关系
                    if relation_type == 'HAS_SYMPTOM':
                        session.run(
                            """
                            MATCH (d:Disease {id: $source_id})
                            MATCH (s:Symptom {id: $target_id})
                            MERGE (d)-[:HAS_SYMPTOM {evidence: $evidence, evidence_level: $evidence_level}]->(s)
                            """,
                            source_id=source_id,
                            target_id=target_id,
                            evidence=evidence,
                            evidence_level=evidence_level
                        )
                    elif relation_type == 'REQUIRES_TEST':
                        session.run(
                            """
                            MATCH (d:Disease {id: $source_id})
                            MATCH (t:Test {id: $target_id})
                            MERGE (d)-[:REQUIRES_TEST {evidence: $evidence, evidence_level: $evidence_level}]->(t)
                            """,
                            source_id=source_id,
                            target_id=target_id,
                            evidence=evidence,
                            evidence_level=evidence_level
                        )
                    elif relation_type == 'TREATED_WITH_MEDICATION':
                        session.run(
                            """
                            MATCH (d:Disease {id: $source_id})
                            MATCH (m:Medication {id: $target_id})
                            MERGE (d)-[:TREATED_WITH_MEDICATION {evidence: $evidence, evidence_level: $evidence_level}]->(m)
                            """,
                            source_id=source_id,
                            target_id=target_id,
                            evidence=evidence,
                            evidence_level=evidence_level
                        )
                    elif relation_type == 'TREATED_WITH':
                        session.run(
                            """
                            MATCH (d:Disease {id: $source_id})
                            MATCH (r:Treatment {id: $target_id})
                            MERGE (d)-[:TREATED_WITH {evidence: $evidence, evidence_level: $evidence_level}]->(r)
                            """,
                            source_id=source_id,
                            target_id=target_id,
                            evidence=evidence,
                            evidence_level=evidence_level
                        )
                    elif relation_type == 'CAUSES':
                        session.run(
                            """
                            MATCH (d1:Disease {id: $source_id})
                            MATCH (d2:Disease {id: $target_id})
                            MERGE (d1)-[:CAUSES {evidence: $evidence, evidence_level: $evidence_level}]->(d2)
                            """,
                            source_id=source_id,
                            target_id=target_id,
                            evidence=evidence,
                            evidence_level=evidence_level
                        )
    
    def build_sample_knowledge_graph(self):
        """构建示例知识图谱（呼吸系统常见疾病）"""
        with self.driver.session() as session:
            # 创建疾病节点
            session.run("MERGE (d:Disease {id: 'D001', name: '社区获得性肺炎', description: '由细菌、病毒或其他病原体引起的肺部感染', severity: '中等'})")
            session.run("MERGE (d:Disease {id: 'D002', name: '慢性阻塞性肺疾病', description: '以气流受限为特征的慢性气道疾病', severity: '慢性'})")
            session.run("MERGE (d:Disease {id: 'D003', name: '哮喘', description: '气道慢性炎症性疾病', severity: '慢性'})")
            
            # 创建症状节点
            session.run("MERGE (s:Symptom {id: 'S001', name: '发热', description: '体温升高超过正常范围'})")
            session.run("MERGE (s:Symptom {id: 'S002', name: '咳嗽', description: '呼吸道防御反射'})")
            session.run("MERGE (s:Symptom {id: 'S003', name: '咳痰', description: '呼吸道分泌物排出'})")
            session.run("MERGE (s:Symptom {id: 'S004', name: '呼吸困难', description: '呼吸费力或困难'})")
            session.run("MERGE (s:Symptom {id: 'S005', name: '胸痛', description: '胸部疼痛'})")
            
            # 创建检查节点
            session.run("MERGE (t:Test {id: 'T001', name: '血常规', description: '检测血液中各种细胞的数量和比例', normal_range: '白细胞: 4-10×10^9/L'})")
            session.run("MERGE (t:Test {id: 'T002', name: '胸部CT', description: '胸部计算机断层扫描', normal_range: '无异常密度影'})")
            session.run("MERGE (t:Test {id: 'T003', name: '血气分析', description: '检测血液中的氧气和二氧化碳水平', normal_range: 'PaO2: 80-100mmHg'})")
            
            # 创建药物节点
            session.run("MERGE (m:Medication {id: 'M001', name: '阿莫西林', description: 'β-内酰胺类抗生素', dosage: '500mg, 3次/日', side_effects: '胃肠道反应'})")
            session.run("MERGE (m:Medication {id: 'M002', name: '沙丁胺醇', description: 'β2受体激动剂', dosage: '200μg, 按需使用', side_effects: '心悸、震颤'})")
            session.run("MERGE (m:Medication {id: 'M003', name: '布地奈德', description: '糖皮质激素', dosage: '200μg, 2次/日', side_effects: '口腔念珠菌感染'})")
            
            # 创建治疗节点
            session.run("MERGE (r:Treatment {id: 'R001', name: '氧疗', description: '通过鼻导管或面罩给予氧气'})")
            session.run("MERGE (r:Treatment {id: 'R002', name: '机械通气', description: '通过呼吸机辅助呼吸'})")
            
            # 创建关系
            session.run("MATCH (d:Disease {id: 'D001'}) MATCH (s:Symptom {id: 'S001'}) MERGE (d)-[:HAS_SYMPTOM {evidence: '中国成人社区获得性肺炎诊断和治疗指南(2016版)', evidence_level: 'A'}]->(s)")
            session.run("MATCH (d:Disease {id: 'D001'}) MATCH (s:Symptom {id: 'S002'}) MERGE (d)-[:HAS_SYMPTOM {evidence: '中国成人社区获得性肺炎诊断和治疗指南(2016版)', evidence_level: 'A'}]->(s)")
            session.run("MATCH (d:Disease {id: 'D001'}) MATCH (s:Symptom {id: 'S003'}) MERGE (d)-[:HAS_SYMPTOM {evidence: '中国成人社区获得性肺炎诊断和治疗指南(2016版)', evidence_level: 'A'}]->(s)")
            session.run("MATCH (d:Disease {id: 'D001'}) MATCH (s:Symptom {id: 'S004'}) MERGE (d)-[:HAS_SYMPTOM {evidence: '中国成人社区获得性肺炎诊断和治疗指南(2016版)', evidence_level: 'A'}]->(s)")
            session.run("MATCH (d:Disease {id: 'D001'}) MATCH (t:Test {id: 'T001'}) MERGE (d)-[:REQUIRES_TEST {evidence: '中国成人社区获得性肺炎诊断和治疗指南(2016版)', evidence_level: 'A'}]->(t)")
            session.run("MATCH (d:Disease {id: 'D001'}) MATCH (t:Test {id: 'T002'}) MERGE (d)-[:REQUIRES_TEST {evidence: '中国成人社区获得性肺炎诊断和治疗指南(2016版)', evidence_level: 'A'}]->(t)")
            session.run("MATCH (d:Disease {id: 'D001'}) MATCH (m:Medication {id: 'M001'}) MERGE (d)-[:TREATED_WITH_MEDICATION {evidence: '中国成人社区获得性肺炎诊断和治疗指南(2016版)', evidence_level: 'A'}]->(m)")
            session.run("MATCH (d:Disease {id: 'D001'}) MATCH (r:Treatment {id: 'R001'}) MERGE (d)-[:TREATED_WITH {evidence: '中国成人社区获得性肺炎诊断和治疗指南(2016版)', evidence_level: 'A'}]->(r)")
            
            session.run("MATCH (d:Disease {id: 'D002'}) MATCH (s:Symptom {id: 'S002'}) MERGE (d)-[:HAS_SYMPTOM {evidence: '慢性阻塞性肺疾病全球倡议(GOLD)', evidence_level: 'A'}]->(s)")
            session.run("MATCH (d:Disease {id: 'D002'}) MATCH (s:Symptom {id: 'S003'}) MERGE (d)-[:HAS_SYMPTOM {evidence: '慢性阻塞性肺疾病全球倡议(GOLD)', evidence_level: 'A'}]->(s)")
            session.run("MATCH (d:Disease {id: 'D002'}) MATCH (s:Symptom {id: 'S004'}) MERGE (d)-[:HAS_SYMPTOM {evidence: '慢性阻塞性肺疾病全球倡议(GOLD)', evidence_level: 'A'}]->(s)")
            session.run("MATCH (d:Disease {id: 'D002'}) MATCH (t:Test {id: 'T003'}) MERGE (d)-[:REQUIRES_TEST {evidence: '慢性阻塞性肺疾病全球倡议(GOLD)', evidence_level: 'A'}]->(t)")
            session.run("MATCH (d:Disease {id: 'D002'}) MATCH (m:Medication {id: 'M002'}) MERGE (d)-[:TREATED_WITH_MEDICATION {evidence: '慢性阻塞性肺疾病全球倡议(GOLD)', evidence_level: 'A'}]->(m)")
            session.run("MATCH (d:Disease {id: 'D002'}) MATCH (m:Medication {id: 'M003'}) MERGE (d)-[:TREATED_WITH_MEDICATION {evidence: '慢性阻塞性肺疾病全球倡议(GOLD)', evidence_level: 'A'}]->(m)")
            
            session.run("MATCH (d:Disease {id: 'D003'}) MATCH (s:Symptom {id: 'S002'}) MERGE (d)-[:HAS_SYMPTOM {evidence: '全球哮喘防治倡议(GINA)', evidence_level: 'A'}]->(s)")
            session.run("MATCH (d:Disease {id: 'D003'}) MATCH (s:Symptom {id: 'S004'}) MERGE (d)-[:HAS_SYMPTOM {evidence: '全球哮喘防治倡议(GINA)', evidence_level: 'A'}]->(s)")
            session.run("MATCH (d:Disease {id: 'D003'}) MATCH (m:Medication {id: 'M002'}) MERGE (d)-[:TREATED_WITH_MEDICATION {evidence: '全球哮喘防治倡议(GINA)', evidence_level: 'A'}]->(m)")
            session.run("MATCH (d:Disease {id: 'D003'}) MATCH (m:Medication {id: 'M003'}) MERGE (d)-[:TREATED_WITH_MEDICATION {evidence: '全球哮喘防治倡议(GINA)', evidence_level: 'A'}]->(m)")

def main():
    """主函数"""
    # Neo4j连接信息
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "Yjz61925!"
    
    # 创建知识图谱实例
    kg = ClinicalKnowledgeGraph(uri, user, password)
    
    try:
        # 创建约束
        kg.create_constraints()
        print("创建约束完成")
        
        # 构建示例知识图谱
        kg.build_sample_knowledge_graph()
        print("构建示例知识图谱完成")
        
        # 提示用户如何使用
        print("\n知识图谱构建完成！")
        print("您可以使用Neo4j Browser (http://localhost:7474) 查看和查询知识图谱")
        print("\n示例查询：")
        print("1. 查找所有疾病及其症状：MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom) RETURN d.name, s.name, r.evidence")
        print("2. 查找社区获得性肺炎的诊断检查：MATCH (d:Disease {name: '社区获得性肺炎'})-[r:REQUIRES_TEST]->(t:Test) RETURN d.name, t.name, r.evidence")
        print("3. 查找哮喘的治疗方案：MATCH (d:Disease {name: '哮喘'})-[r]->(t) RETURN d.name, type(r), t.name, r.evidence")
    finally:
        # 关闭连接
        kg.close()

if __name__ == "__main__":
    main()
