#!/usr/bin/env python3
"""
运行Cypher查询，从Neo4j知识图谱中查找所有疾病及其症状
"""

from neo4j import GraphDatabase

# Neo4j连接信息
uri = "bolt://localhost:7687"
user = "neo4j"
password = "Yjz61925!"

# 连接到Neo4j
driver = GraphDatabase.driver(uri, auth=(user, password))

# 执行Cypher查询
with driver.session() as session:
    # 查询所有疾病及其症状
    result = session.run("MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom) RETURN d.name, s.name, r.evidence")
    
    print("疾病\t症状\t证据")
    print("-" * 80)
    for record in result:
        print(f"{record['d.name']}\t{record['s.name']}\t{record['r.evidence']}")

# 关闭连接
driver.close()
