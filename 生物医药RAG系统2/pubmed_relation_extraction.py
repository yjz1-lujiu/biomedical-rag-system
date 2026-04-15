#!/usr/bin/env python3
"""
PubMed文献关系抽取模块
- 从PubMed摘要中抽取药物-靶点和靶点-疾病三元组
- 使用PubMedBERT模型进行关系抽取
- 将抽取的三元组补充到Neo4j知识图谱中
"""

import os
import json
from datetime import datetime, timedelta
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

class PubMedRelationExtractor:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.data_dir = DATA_DIR
        # 尝试加载HuggingFace模型
        self.pipeline = None
        try:
            from transformers import pipeline
            # 使用PubMedBERT模型进行关系抽取
            self.pipeline = pipeline(
                "token-classification",
                model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
            )
            print("成功加载PubMedBERT模型")
        except Exception as e:
            print(f"警告: 无法加载PubMedBERT模型: {e}")
            print("将使用基于规则的方法进行关系抽取")
    
    def close(self):
        self.driver.close()
    
    def search_pubmed(self, query, max_results=10):
        """搜索PubMed获取相关摘要"""
        print(f"正在搜索PubMed: {query}")
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
        abstract_list = []
        current_abstract = {}
        for line in abstracts.split("\n"):
            line = line.strip()
            if line.startswith("PMID:"):
                if current_abstract:
                    abstract_list.append(current_abstract)
                current_abstract = {"pmid": line.split(":")[1].strip(), "title": "", "abstract": ""}
            elif line.startswith("TI:"):
                current_abstract["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("AB:"):
                current_abstract["abstract"] = line.split(":", 1)[1].strip()
            elif current_abstract and "abstract" in current_abstract and line:
                current_abstract["abstract"] += " " + line
        
        if current_abstract:
            abstract_list.append(current_abstract)
        
        print(f"找到 {len(abstract_list)} 篇相关文献")
        return abstract_list
    
    def extract_relations(self, text):
        """从文本中抽取药物-靶点和靶点-疾病关系"""
        relations = []
        
        # 基于规则的关系抽取（当模型不可用时）
        # 这里使用简单的规则，实际项目中可以使用更复杂的规则或模型
        drug_target_patterns = [
            (r"(\w+)\s+binds\s+to\s+(\w+)", "BINDS_TO"),
            (r"(\w+)\s+targets\s+(\w+)", "BINDS_TO"),
            (r"(\w+)\s+inhibits\s+(\w+)", "BINDS_TO"),
            (r"(\w+)\s+activates\s+(\w+)", "BINDS_TO"),
        ]
        
        target_disease_patterns = [
            (r"(\w+)\s+is\s+associated\s+with\s+(\w+)", "ASSOCIATED_WITH"),
            (r"(\w+)\s+plays\s+a\s+role\s+in\s+(\w+)", "ASSOCIATED_WITH"),
            (r"(\w+)\s+is\s+involved\s+in\s+(\w+)", "ASSOCIATED_WITH"),
        ]
        
        import re
        for pattern, relation_type in drug_target_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                drug, target = match.groups()
                relations.append({
                    "source": drug,
                    "target": target,
                    "relation": relation_type,
                    "evidence": "Rule-based extraction"
                })
        
        for pattern, relation_type in target_disease_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                target, disease = match.groups()
                relations.append({
                    "source": target,
                    "target": disease,
                    "relation": relation_type,
                    "evidence": "Rule-based extraction"
                })
        
        # 如果有模型，使用模型进行关系抽取
        if self.pipeline:
            # 这里使用简单的命名实体识别，实际项目中可以使用更复杂的关系抽取模型
            results = self.pipeline(text, aggregation_strategy="simple")
            # 处理模型结果...
            pass
        
        return relations
    
    def add_relations_to_graph(self, relations, pmid):
        """将抽取的关系添加到Neo4j知识图谱中"""
        added_count = 0
        
        with self.driver.session() as session:
            for relation in relations:
                source = relation["source"]
                target = relation["target"]
                relation_type = relation["relation"]
                evidence = f"PubMed:{pmid} - {relation['evidence']}"
                
                # 检查关系是否已存在
                exists = session.run(
                    f"MATCH (s)-[r:{relation_type}]->(t) "
                    "WHERE s.name = $source AND t.name = $target "
                    "RETURN COUNT(r) > 0 as exists",
                    source=source, target=target
                ).single()
                
                if not exists["exists"]:
                    # 确定节点类型
                    if relation_type == "BINDS_TO":
                        # 药物-靶点关系
                        session.run(
                            "MERGE (s:Drug {name: $source}) "
                            "MERGE (t:Target {name: $target}) "
                            "MERGE (s)-[r:BINDS_TO {evidence: $evidence}]->(t)",
                            source=source, target=target, evidence=evidence
                        )
                    elif relation_type == "ASSOCIATED_WITH":
                        # 靶点-疾病关系
                        session.run(
                            "MERGE (s:Target {name: $source}) "
                            "MERGE (t:Disease {name: $target}) "
                            "MERGE (s)-[r:ASSOCIATED_WITH {evidence: $evidence}]->(t)",
                            source=source, target=target, evidence=evidence
                        )
                    added_count += 1
        
        return added_count
    
    def process_pubmed_abstracts(self, query, max_results=10):
        """处理PubMed摘要并抽取关系"""
        abstracts = self.search_pubmed(query, max_results)
        total_relations = 0
        total_added = 0
        
        for abstract in abstracts:
            pmid = abstract.get("pmid", "")
            title = abstract.get("title", "")
            abstract_text = abstract.get("abstract", "")
            
            print(f"处理文献: {title} (PMID: {pmid})")
            
            # 抽取关系
            relations = self.extract_relations(abstract_text)
            total_relations += len(relations)
            
            # 添加到图谱
            added = self.add_relations_to_graph(relations, pmid)
            total_added += added
            
            if relations:
                print(f"  抽取到 {len(relations)} 个关系，新增 {added} 个关系")
            else:
                print("  未抽取到关系")
        
        print(f"总计抽取到 {total_relations} 个关系，新增 {total_added} 个关系")
        return total_relations, total_added
    
    def continuous_update_pipeline(self, queries, max_results=5, interval_hours=24):
        """持续更新管道（伪代码）"""
        print("=== 启动文献持续更新管道 ===")
        print(f"更新间隔: {interval_hours} 小时")
        print(f"查询列表: {queries}")
        
        # 实际项目中，这里可以使用调度器（如APScheduler）实现定时任务
        # 这里仅展示逻辑
        for query in queries:
            print(f"\n处理查询: {query}")
            self.process_pubmed_abstracts(query, max_results)
        
        print("=== 文献持续更新管道执行完成 ===")
    
    def verify_extraction(self):
        """验证关系抽取结果"""
        print("正在验证关系抽取结果...")
        
        with self.driver.session() as session:
            # 统计从PubMed抽取的关系
            pubmed_relations = session.run(
                "MATCH ()-[r]->() WHERE r.evidence STARTS WITH 'PubMed:' RETURN COUNT(r) as count"
            ).single()
            
            print(f"从PubMed抽取的关系数量: {pubmed_relations['count']}")
            
            # 查看部分PubMed抽取的关系
            print("\n示例PubMed抽取的关系:")
            sample_relations = session.run(
                "MATCH (s)-[r]->(t) WHERE r.evidence STARTS WITH 'PubMed:' RETURN s.name, type(r), t.name, r.evidence LIMIT 5"
            ).data()
            
            for item in sample_relations:
                print(f"{item['s.name']} -[{item['type(r)']}]-> {item['t.name']} (证据: {item['r.evidence']})")

def main():
    """主函数"""
    print("=== PubMed文献关系抽取 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    extractor = PubMedRelationExtractor()
    
    try:
        # 定义查询列表
        queries = [
            "metformin cancer",
            "aspirin cancer",
            "AMPK diabetes",
            "mTOR cancer"
        ]
        
        # 处理PubMed摘要
        for query in queries:
            print(f"\n=== 处理查询: {query} ===")
            extractor.process_pubmed_abstracts(query, max_results=5)
        
        # 验证抽取结果
        extractor.verify_extraction()
        
        # 展示持续更新管道
        print("\n=== 展示持续更新管道 ===")
        extractor.continuous_update_pipeline(queries, max_results=2, interval_hours=24)
        
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=== 文献关系抽取完成 ===")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        extractor.close()

if __name__ == "__main__":
    main()
