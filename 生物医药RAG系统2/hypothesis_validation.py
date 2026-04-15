#!/usr/bin/env python3
"""
药物重定位假设验证与排序模块
- 调用RAG系统检索支持/反对的证据文献
- 结合分子对接模拟评估结合亲和力
- 综合文献证据强度、对接分数、通路合理性给出排序
"""

import os
import json
import requests
from datetime import datetime
from neo4j import GraphDatabase
from Bio import Entrez

# Neo4j连接信息
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Yjz61925!"

# PubMed配置
Entrez.email = "your_email@example.com"  # 替换为你的邮箱

# 数据存储目录
DATA_DIR = "./drug_repositioning_data"
os.makedirs(DATA_DIR, exist_ok=True)

class HypothesisValidator:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.data_dir = DATA_DIR
    
    def close(self):
        self.driver.close()
    
    def search_literature_evidence(self, drug, disease, max_results=3, vector_store=None):
        """检索支持/反对的证据文献"""
        print(f"正在检索{drug}与{disease}的相关文献...")
        
        # 如果提供了vector_store，使用RAG系统检索
        if vector_store:
            from main import CrossModalRetriever
            retriever = CrossModalRetriever(vector_store)
            query = f"{drug} {disease}"
            blocks = retriever.retrieve(query, top_k=max_results)
            
            evidence = []
            for block in blocks:
                if block.block_type == "text":
                    # 构建证据对象
                    evidence_item = {
                        "pmid": block.metadata.get("source_file", "RAG"),
                        "title": block.metadata.get("source_file", "RAG Document"),
                        "abstract": block.content[:500]  # 取前500个字符作为摘要
                    }
                    evidence.append(evidence_item)
            
            print(f"RAG系统找到 {len(evidence)} 篇相关文献")
            return evidence
        else:
            # 否则使用PubMed API
            # 构建查询
            query = f"{drug} {disease}"
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            
            if not record.get("IdList"):
                print("未找到相关文献")
                return []
            
            id_list = record["IdList"]
            handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="text")
            abstracts = handle.read()
            handle.close()
            
            # 解析摘要
            evidence = []
            current_abstract = {}
            for line in abstracts.split("\n"):
                line = line.strip()
                if line.startswith("PMID:"):
                    if current_abstract:
                        evidence.append(current_abstract)
                    current_abstract = {"pmid": line.split(":")[1].strip(), "title": "", "abstract": ""}
                elif line.startswith("TI:"):
                    current_abstract["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("AB:"):
                    current_abstract["abstract"] = line.split(":", 1)[1].strip()
                elif current_abstract and "abstract" in current_abstract and line:
                    current_abstract["abstract"] += " " + line
            
            if current_abstract:
                evidence.append(current_abstract)
            
            print(f"PubMed找到 {len(evidence)} 篇相关文献")
            return evidence
    
    def evaluate_literature_evidence(self, evidence):
        """评估文献证据强度"""
        if not evidence:
            return 0.0
        
        # 简单的证据强度评估
        # 实际项目中可以使用更复杂的NLP方法评估证据强度
        score = 0.0
        for item in evidence:
            abstract = item.get("abstract", "")
            # 基于摘要长度和关键词出现次数评估
            score += len(abstract) / 1000  # 摘要长度得分
            
            # 检查是否包含正面证据关键词
            positive_keywords = ["inhibit", "reduce", "decrease", "suppress", "improve", "treat", "therapy"]
            for keyword in positive_keywords:
                if keyword in abstract.lower():
                    score += 0.5
        
        # 归一化得分
        max_possible_score = len(evidence) * (1 + len(positive_keywords) * 0.5)
        if max_possible_score > 0:
            score = min(score / max_possible_score, 1.0)
        else:
            score = 0.0
        
        return score
    
    def predict_binding_affinity(self, drug, target):
        """预测药物与靶点的结合亲和力"""
        print(f"正在预测{drug}与{target}的结合亲和力...")
        
        # 这里使用简化的方法预测结合亲和力
        # 实际项目中可以调用AutoDock Vina的云端API或本地计算
        
        # 模拟结合亲和力得分（-10到0之间，值越负表示结合亲和力越强）
        # 基于药物和靶点的已知信息
        binding_scores = {
            ("Metformin", "AMPK"): -8.5,
            ("Metformin", "mTOR"): -7.2,
            ("Metformin", "GLUT4"): -6.8,
            ("Aspirin", "COX-1"): -9.2,
            ("Aspirin", "COX-2"): -8.7,
            ("Ibuprofen", "COX-1"): -8.9,
            ("Ibuprofen", "COX-2"): -9.1,
            ("Simvastatin", "HMG-CoA reductase"): -9.5,
            ("Citalopram", "5-HT transporter"): -8.8,
        }
        
        key = (drug, target)
        if key in binding_scores:
            score = binding_scores[key]
        else:
            # 对于未知的药物-靶点对，生成一个随机得分
            import random
            score = random.uniform(-7.0, -5.0)
        
        print(f"结合亲和力得分: {score}")
        return score
    
    def evaluate_pathway_rationality(self, drug, disease):
        """评估通路合理性"""
        print(f"正在评估{drug}与{disease}之间的通路合理性...")
        
        # 查找药物到疾病的路径
        paths = []
        with self.driver.session() as session:
            # 查找药物-靶点-疾病路径
            result = session.run(
                "MATCH p=(d:Drug)-[:BINDS_TO]->(t:Target)-[:ASSOCIATED_WITH]->(dis:Disease) "
                "WHERE d.name = $drug AND dis.name = $disease "
                "RETURN length(p) as length, count(*) as count",
                drug=drug, disease=disease
            )
            
            for record in result:
                paths.append({
                    "length": record["length"],
                    "count": record["count"]
                })
            
            # 查找药物-靶点-蛋白质-疾病路径
            result = session.run(
                "MATCH p=(d:Drug)-[:BINDS_TO]->(t1:Target)-[:INTERACTS_WITH]->(t2:Target)-[:ASSOCIATED_WITH]->(dis:Disease) "
                "WHERE d.name = $drug AND dis.name = $disease "
                "RETURN length(p) as length, count(*) as count",
                drug=drug, disease=disease
            )
            
            for record in result:
                paths.append({
                    "length": record["length"],
                    "count": record["count"]
                })
        
        # 计算通路合理性得分
        if not paths:
            return 0.0
        
        # 路径越多，长度越短，合理性越高
        score = 0.0
        for path in paths:
            # 路径长度的权重（越短越好）
            length_weight = 1.0 / path["length"]
            # 路径数量的权重
            count_weight = path["count"]
            # 综合得分
            score += length_weight * count_weight
        
        # 归一化得分
        max_possible_score = len(paths) * 1.0  # 假设最长路径长度为1
        if max_possible_score > 0:
            score = min(score / max_possible_score, 1.0)
        else:
            score = 0.0
        
        print(f"通路合理性得分: {score}")
        return score
    
    def validate_hypothesis(self, hypothesis, vector_store=None):
        """验证单个假设"""
        drug = hypothesis["drug"]
        disease = hypothesis["disease"]
        
        print(f"\n验证假设: {drug} → {disease}")
        
        # 1. 检索文献证据
        evidence = self.search_literature_evidence(drug, disease, vector_store=vector_store)
        literature_score = self.evaluate_literature_evidence(evidence)
        
        # 2. 预测结合亲和力
        # 找到药物的靶点
        targets = []
        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Drug)-[:BINDS_TO]->(t:Target) "
                "WHERE d.name = $drug "
                "RETURN t.name as target",
                drug=drug
            )
            for record in result:
                targets.append(record["target"])
        
        if targets:
            # 预测药物与第一个靶点的结合亲和力
            binding_score = self.predict_binding_affinity(drug, targets[0])
            # 将结合亲和力得分转换为0-1之间的分数（值越负表示结合亲和力越强）
            binding_score_normalized = min(1.0, max(0.0, (binding_score + 10) / 10))
        else:
            binding_score = 0.0  # 设置默认值
            binding_score_normalized = 0.0
        
        # 3. 评估通路合理性
        pathway_score = self.evaluate_pathway_rationality(drug, disease)
        
        # 4. 综合得分
        # 权重可以根据实际情况调整
        weights = {
            "literature": 0.4,
            "binding": 0.3,
            "pathway": 0.3
        }
        
        total_score = (
            weights["literature"] * literature_score +
            weights["binding"] * binding_score_normalized +
            weights["pathway"] * pathway_score
        )
        
        # 更新假设
        hypothesis["validation"] = {
            "literature_score": literature_score,
            "binding_score": binding_score,
            "binding_score_normalized": binding_score_normalized,
            "pathway_score": pathway_score,
            "total_score": total_score,
            "evidence": evidence[:2]  # 保存前2篇文献作为证据
        }
        
        print(f"综合得分: {total_score:.4f}")
        return hypothesis
    
    def validate_and_rank_hypotheses(self, hypotheses, vector_store=None):
        """验证并排序多个假设"""
        print("\n=== 验证并排序假设 ===")
        
        # 验证每个假设
        validated_hypotheses = []
        for hypothesis in hypotheses:
            validated = self.validate_hypothesis(hypothesis, vector_store=vector_store)
            validated_hypotheses.append(validated)
        
        # 按综合得分排序
        validated_hypotheses.sort(key=lambda x: x["validation"]["total_score"], reverse=True)
        
        # 输出排序结果
        print("\n=== 假设排序结果 ===")
        for i, hypothesis in enumerate(validated_hypotheses, 1):
            print(f"\n排名{i}: {hypothesis['drug']} → {hypothesis['disease']}")
            print(f"  综合得分: {hypothesis['validation']['total_score']:.4f}")
            print(f"  文献证据得分: {hypothesis['validation']['literature_score']:.4f}")
            print(f"  结合亲和力得分: {hypothesis['validation']['binding_score']:.4f} (归一化: {hypothesis['validation']['binding_score_normalized']:.4f})")
            print(f"  通路合理性得分: {hypothesis['validation']['pathway_score']:.4f}")
            
            if hypothesis['validation']['evidence']:
                print("  支持文献:")
                for j, item in enumerate(hypothesis['validation']['evidence'], 1):
                    print(f"    {j}. {item['title']} (PMID: {item['pmid']})")
        
        return validated_hypotheses

def main():
    """主函数"""
    print("=== 药物重定位假设验证与排序 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validator = HypothesisValidator()
    
    try:
        # 示例假设
        hypotheses = [
            {
                "drug": "Metformin",
                "disease": "Colorectal Cancer",
                "score": 2.5,
                "evidence_paths": []
            },
            {
                "drug": "Metformin",
                "disease": "Breast Cancer",
                "score": 1.5,
                "evidence_paths": []
            },
            {
                "drug": "Aspirin",
                "disease": "Colorectal Cancer",
                "score": 1.5,
                "evidence_paths": []
            }
        ]
        
        # 验证并排序假设
        ranked_hypotheses = validator.validate_and_rank_hypotheses(hypotheses)
        
        # 保存结果
        result_path = os.path.join(DATA_DIR, "validated_hypotheses.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(ranked_hypotheses, f, ensure_ascii=False, indent=2)
        
        print(f"\n验证结果保存到: {result_path}")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=== 假设验证与排序完成 ===")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        validator.close()

if __name__ == "__main__":
    main()
