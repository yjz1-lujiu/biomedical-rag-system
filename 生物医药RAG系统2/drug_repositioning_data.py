#!/usr/bin/env python3
"""
药物重定位数据导入脚本
- 从公开源下载并构建初始药物-靶点-疾病图谱
- 支持DrugBank、ChEMBL、DisGeNET、Open Targets、STRING数据库
"""

import os
import json
import requests
import pandas as pd
from neo4j import GraphDatabase
from datetime import datetime

# Neo4j连接信息
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Yjz61925!"

# 数据存储目录
DATA_DIR = "./drug_repositioning_data"
os.makedirs(DATA_DIR, exist_ok=True)

class DrugRepositioningDataLoader:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.data_dir = DATA_DIR
    
    def close(self):
        self.driver.close()
    
    def download_drugbank_data(self):
        """下载DrugBank数据（示例数据）"""
        print("正在下载DrugBank数据...")
        # 这里使用示例数据，实际项目中可以从DrugBank API或下载完整数据集
        drug_target_data = [
            {"drug": "Metformin", "target": "AMPK", "evidence": "DrugBank:DB00393"},
            {"drug": "Metformin", "target": "mTOR", "evidence": "DrugBank:DB00393"},
            {"drug": "Aspirin", "target": "COX-1", "evidence": "DrugBank:DB00945"},
            {"drug": "Aspirin", "target": "COX-2", "evidence": "DrugBank:DB00945"},
            {"drug": "Ibuprofen", "target": "COX-1", "evidence": "DrugBank:DB01050"},
            {"drug": "Ibuprofen", "target": "COX-2", "evidence": "DrugBank:DB01050"},
            {"drug": "Simvastatin", "target": "HMG-CoA reductase", "evidence": "DrugBank:DB00641"},
            {"drug": "Citalopram", "target": "5-HT transporter", "evidence": "DrugBank:DB00215"},
        ]
        
        df = pd.DataFrame(drug_target_data)
        df.to_csv(os.path.join(self.data_dir, "drugbank_drug_target.csv"), index=False)
        print("DrugBank数据下载完成")
        return drug_target_data
    
    def download_chembl_data(self):
        """下载ChEMBL数据（示例数据）"""
        print("正在下载ChEMBL数据...")
        # 这里使用示例数据，实际项目中可以从ChEMBL API获取
        drug_target_data = [
            {"drug": "Metformin", "target": "AMPK", "evidence": "ChEMBL:CHEMBL1201419"},
            {"drug": "Metformin", "target": "GLUT4", "evidence": "ChEMBL:CHEMBL1201419"},
            {"drug": "Aspirin", "target": "COX-1", "evidence": "ChEMBL:CHEMBL25"},
            {"drug": "Aspirin", "target": "COX-2", "evidence": "ChEMBL:CHEMBL25"},
        ]
        
        df = pd.DataFrame(drug_target_data)
        df.to_csv(os.path.join(self.data_dir, "chembl_drug_target.csv"), index=False)
        print("ChEMBL数据下载完成")
        return drug_target_data
    
    def download_disgenet_data(self):
        """下载DisGeNET数据（示例数据）"""
        print("正在下载DisGeNET数据...")
        # 这里使用示例数据，实际项目中可以从DisGeNET API获取
        target_disease_data = [
            {"target": "AMPK", "disease": "Type 2 Diabetes", "evidence": "DisGeNET:CURATED"},
            {"target": "AMPK", "disease": "Colorectal Cancer", "evidence": "DisGeNET:CURATED"},
            {"target": "mTOR", "disease": "Colorectal Cancer", "evidence": "DisGeNET:CURATED"},
            {"target": "mTOR", "disease": "Breast Cancer", "evidence": "DisGeNET:CURATED"},
            {"target": "COX-1", "disease": "Inflammation", "evidence": "DisGeNET:CURATED"},
            {"target": "COX-2", "disease": "Inflammation", "evidence": "DisGeNET:CURATED"},
            {"target": "COX-2", "disease": "Colorectal Cancer", "evidence": "DisGeNET:CURATED"},
            {"target": "HMG-CoA reductase", "disease": "Hypercholesterolemia", "evidence": "DisGeNET:CURATED"},
            {"target": "5-HT transporter", "disease": "Depression", "evidence": "DisGeNET:CURATED"},
        ]
        
        df = pd.DataFrame(target_disease_data)
        df.to_csv(os.path.join(self.data_dir, "disgenet_target_disease.csv"), index=False)
        print("DisGeNET数据下载完成")
        return target_disease_data
    
    def download_open_targets_data(self):
        """下载Open Targets数据（示例数据）"""
        print("正在下载Open Targets数据...")
        # 这里使用示例数据，实际项目中可以从Open Targets API获取
        target_disease_data = [
            {"target": "AMPK", "disease": "Type 2 Diabetes", "evidence": "Open Targets:GENETIC_ASSOCIATION"},
            {"target": "AMPK", "disease": "Obesity", "evidence": "Open Targets:GENETIC_ASSOCIATION"},
            {"target": "mTOR", "disease": "Colorectal Cancer", "evidence": "Open Targets:GENETIC_ASSOCIATION"},
            {"target": "COX-2", "disease": "Colorectal Cancer", "evidence": "Open Targets:GENETIC_ASSOCIATION"},
        ]
        
        df = pd.DataFrame(target_disease_data)
        df.to_csv(os.path.join(self.data_dir, "open_targets_target_disease.csv"), index=False)
        print("Open Targets数据下载完成")
        return target_disease_data
    
    def download_string_data(self):
        """下载STRING数据库数据（示例数据）"""
        print("正在下载STRING数据库数据...")
        # 这里使用示例数据，实际项目中可以从STRING API获取
        ppi_data = [
            {"protein1": "AMPK", "protein2": "mTOR", "score": 0.75, "evidence": "STRING:9606.ENSP00000369699"},
            {"protein1": "AMPK", "protein2": "GLUT4", "score": 0.65, "evidence": "STRING:9606.ENSP00000369699"},
            {"protein1": "COX-1", "protein2": "COX-2", "score": 0.85, "evidence": "STRING:9606.ENSP00000263436"},
        ]
        
        df = pd.DataFrame(ppi_data)
        df.to_csv(os.path.join(self.data_dir, "string_ppi.csv"), index=False)
        print("STRING数据库数据下载完成")
        return ppi_data
    
    def build_neo4j_graph(self):
        """构建Neo4j知识图谱"""
        print("正在构建Neo4j知识图谱...")
        
        # 下载数据
        drugbank_data = self.download_drugbank_data()
        chembl_data = self.download_chembl_data()
        disgenet_data = self.download_disgenet_data()
        open_targets_data = self.download_open_targets_data()
        string_data = self.download_string_data()
        
        # 合并药物-靶点数据
        all_drug_target = drugbank_data + chembl_data
        
        # 合并靶点-疾病数据
        all_target_disease = disgenet_data + open_targets_data
        
        # 去重
        seen_drug_target = set()
        unique_drug_target = []
        for item in all_drug_target:
            key = (item["drug"], item["target"])
            if key not in seen_drug_target:
                seen_drug_target.add(key)
                unique_drug_target.append(item)
        
        seen_target_disease = set()
        unique_target_disease = []
        for item in all_target_disease:
            key = (item["target"], item["disease"])
            if key not in seen_target_disease:
                seen_target_disease.add(key)
                unique_target_disease.append(item)
        
        # 构建图谱
        with self.driver.session() as session:
            # 创建约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Target) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
            
            # 清空现有数据
            session.run("MATCH (n) DETACH DELETE n")
            
            # 导入药物-靶点关系
            for item in unique_drug_target:
                session.run(
                    "MERGE (d:Drug {name: $drug}) "
                    "MERGE (t:Target {name: $target}) "
                    "MERGE (d)-[r:BINDS_TO {evidence: $evidence}]->(t)",
                    drug=item["drug"], target=item["target"], evidence=item["evidence"]
                )
            
            # 导入靶点-疾病关系
            for item in unique_target_disease:
                session.run(
                    "MERGE (t:Target {name: $target}) "
                    "MERGE (d:Disease {name: $disease}) "
                    "MERGE (t)-[r:ASSOCIATED_WITH {evidence: $evidence}]->(d)",
                    target=item["target"], disease=item["disease"], evidence=item["evidence"]
                )
            
            # 导入蛋白质-蛋白质互作
            for item in string_data:
                session.run(
                    "MERGE (p1:Target {name: $protein1}) "
                    "MERGE (p2:Target {name: $protein2}) "
                    "MERGE (p1)-[r:INTERACTS_WITH {score: $score, evidence: $evidence}]->(p2)",
                    protein1=item["protein1"], protein2=item["protein2"], 
                    score=item["score"], evidence=item["evidence"]
                )
        
        print("Neo4j知识图谱构建完成")
        print(f"导入药物-靶点关系: {len(unique_drug_target)}")
        print(f"导入靶点-疾病关系: {len(unique_target_disease)}")
        print(f"导入蛋白质-蛋白质互作: {len(string_data)}")
    
    def verify_graph(self):
        """验证图谱构建结果"""
        print("正在验证图谱构建结果...")
        
        with self.driver.session() as session:
            # 统计节点数量
            drug_count = session.run("MATCH (d:Drug) RETURN count(d) as count").single()
            target_count = session.run("MATCH (t:Target) RETURN count(t) as count").single()
            disease_count = session.run("MATCH (d:Disease) RETURN count(d) as count").single()
            
            # 统计关系数量
            binds_to_count = session.run("MATCH ()-[r:BINDS_TO]->() RETURN count(r) as count").single()
            associated_with_count = session.run("MATCH ()-[r:ASSOCIATED_WITH]->() RETURN count(r) as count").single()
            interacts_with_count = session.run("MATCH ()-[r:INTERACTS_WITH]->() RETURN count(r) as count").single()
            
            print(f"药物节点: {drug_count['count']}")
            print(f"靶点节点: {target_count['count']}")
            print(f"疾病节点: {disease_count['count']}")
            print(f"药物-靶点关系: {binds_to_count['count']}")
            print(f"靶点-疾病关系: {associated_with_count['count']}")
            print(f"蛋白质-蛋白质互作: {interacts_with_count['count']}")
            
            # 查看部分数据
            print("\n示例药物-靶点关系:")
            drug_targets = session.run("MATCH (d:Drug)-[r:BINDS_TO]->(t:Target) RETURN d.name, t.name, r.evidence LIMIT 5").data()
            for item in drug_targets:
                print(f"{item['d.name']} -> {item['t.name']} (证据: {item['r.evidence']})")
            
            print("\n示例靶点-疾病关系:")
            target_diseases = session.run("MATCH (t:Target)-[r:ASSOCIATED_WITH]->(d:Disease) RETURN t.name, d.name, r.evidence LIMIT 5").data()
            for item in target_diseases:
                print(f"{item['t.name']} -> {item['d.name']} (证据: {item['r.evidence']})")

def main():
    """主函数"""
    print("=== 药物重定位数据导入 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    loader = DrugRepositioningDataLoader()
    
    try:
        # 构建知识图谱
        loader.build_neo4j_graph()
        
        # 验证图谱
        loader.verify_graph()
        
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=== 数据导入完成 ===")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        loader.close()

if __name__ == "__main__":
    main()
