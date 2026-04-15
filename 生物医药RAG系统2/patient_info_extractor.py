#!/usr/bin/env python3
"""
患者信息抽取模块
- 从临床病历文本中提取关键实体（症状、检验指标、体征）
- 使用医疗领域NER模型（BioBERT）
- 将提取的实体映射到知识图谱中的标准概念
- 输出标准化实体列表（JSON格式）
"""

import re
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class PatientInfoExtractor:
    def __init__(self):
        """初始化患者信息抽取器"""
        # 尝试加载BioBERT模型，如果失败则使用基于规则的方法
        try:
            # 加载预训练的BioBERT模型
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
            self.model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")
            self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
            self.use_ml_model = True
            print("使用BioBERT模型进行实体提取")
        except Exception as e:
            print(f"加载BioBERT模型失败，使用基于规则的方法: {e}")
            self.use_ml_model = False
        
        # 定义症状、检验指标和体征的模式
        self.symptom_patterns = [
            r'发热', r'咳嗽', r'咳痰', r'呼吸困难', r'胸痛', r'头痛', r'腹痛', r'腹泻',
            r'恶心', r'呕吐', r'乏力', r'疲劳', r'头晕', r'心悸', r'水肿', r'皮疹'
        ]
        
        self.test_patterns = [
            r'血常规', r'白细胞|WBC', r'红细胞|RBC', r'血红蛋白|Hb', r'血小板|PLT',
            r'中性粒细胞|中性', r'淋巴细胞|淋巴', r'胸部CT', r'胸部X线', r'血气分析',
            r'血糖', r'血压|BP', r'心率|HR', r'呼吸|RR', r'体温|T'
        ]
        
        self.sign_patterns = [
            r'T\s*[0-9]+(\.[0-9]+)?°?C?', r'BP\s*[0-9]+/[0-9]+',
            r'HR\s*[0-9]+', r'RR\s*[0-9]+', r'氧饱和度\s*[0-9]+%'
        ]
        
        # 实体到知识图谱概念的映射
        self.entity_mapping = {
            # 症状映射
            '发热': 'S001',
            '咳嗽': 'S002',
            '咳痰': 'S003',
            '呼吸困难': 'S004',
            '胸痛': 'S005',
            # 检验指标映射
            '血常规': 'T001',
            '胸部CT': 'T002',
            '血气分析': 'T003',
            # 体征映射
            '体温': 'T001',
        }
    
    def extract_entities(self, clinical_text: str) -> List[Dict[str, Any]]:
        """从临床文本中提取实体"""
        entities = []
        
        if self.use_ml_model:
            # 使用BioBERT模型提取实体
            try:
                results = self.ner_pipeline(clinical_text)
                for result in results:
                    entity = {
                        'text': result['word'],
                        'type': result['entity'],
                        'score': result['score'],
                        'start': result['start'],
                        'end': result['end']
                    }
                    entities.append(entity)
            except Exception as e:
                print(f"使用BioBERT模型提取实体失败: {e}")
                # 失败时使用基于规则的方法
                entities = self._extract_entities_rule_based(clinical_text)
        else:
            # 使用基于规则的方法提取实体
            entities = self._extract_entities_rule_based(clinical_text)
        
        # 标准化实体
        standardized_entities = self._standardize_entities(entities, clinical_text)
        return standardized_entities
    
    def _extract_entities_rule_based(self, clinical_text: str) -> List[Dict[str, Any]]:
        """基于规则的实体提取"""
        entities = []
        
        # 提取症状
        for pattern in self.symptom_patterns:
            for match in re.finditer(pattern, clinical_text):
                entities.append({
                    'text': match.group(),
                    'type': 'SYMPTOM',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # 提取检验指标
        for pattern in self.test_patterns:
            for match in re.finditer(pattern, clinical_text):
                entities.append({
                    'text': match.group(),
                    'type': 'TEST',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # 提取体征
        for pattern in self.sign_patterns:
            for match in re.finditer(pattern, clinical_text):
                entities.append({
                    'text': match.group(),
                    'type': 'SIGN',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # 去重
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity['text'], entity['type'], entity['start'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _standardize_entities(self, entities: List[Dict[str, Any]], clinical_text: str) -> List[Dict[str, Any]]:
        """标准化实体，映射到知识图谱概念"""
        standardized_entities = []
        
        for entity in entities:
            text = entity['text']
            entity_type = entity['type']
            
            # 映射到知识图谱概念
            concept_id = self._map_to_concept(text)
            
            # 提取数值（如果有）
            value = self._extract_value(text, clinical_text, entity['end'])
            
            standardized_entity = {
                'text': text,
                'type': entity_type,
                'concept_id': concept_id,
                'value': value,
                'start': entity['start'],
                'end': entity['end']
            }
            
            standardized_entities.append(standardized_entity)
        
        return standardized_entities
    
    def _map_to_concept(self, text: str) -> str:
        """将实体映射到知识图谱概念"""
        # 简单的映射逻辑
        for key, value in self.entity_mapping.items():
            if key in text:
                return value
        
        # 如果没有找到映射，返回默认值
        return f"UNMAPPED_{hash(text) % 10000}"
    
    def _extract_value(self, text: str, clinical_text: str, end_pos: int) -> str:
        """提取实体的数值"""
        # 检查是否有数值跟随
        # 例如："WBC 14.2×10^9/L"
        pattern = r'\s*([0-9]+(\.[0-9]+)?(×10\^[0-9]+/[A-Z]+)?)'
        match = re.search(pattern, clinical_text[end_pos:end_pos+20])
        if match:
            return match.group(1)
        
        # 检查实体本身是否包含数值
        pattern = r'[0-9]+(\.[0-9]+)?'
        match = re.search(pattern, text)
        if match:
            return match.group()
        
        return ""
    
    def extract_and_map(self, clinical_text: str) -> Dict[str, Any]:
        """提取并映射实体，返回JSON格式"""
        entities = self.extract_entities(clinical_text)
        
        # 按类型分组
        entities_by_type = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # 构建返回结果
        result = {
            'clinical_text': clinical_text,
            'entities': entities,
            'entities_by_type': entities_by_type,
            'summary': {
                'total_entities': len(entities),
                'symptoms': len([e for e in entities if e['type'] in ['SYMPTOM', 'symptom']]),
                'tests': len([e for e in entities if e['type'] in ['TEST', 'test']]),
                'signs': len([e for e in entities if e['type'] in ['SIGN', 'sign']])
            }
        }
        
        return result

def main():
    """主函数"""
    extractor = PatientInfoExtractor()
    
    # 示例临床文本
    clinical_text = "患者，男，65岁，因\"发热、咳嗽、咳黄痰3天\"入院。查体：T 38.5℃，双肺可闻及湿啰音。血常规：WBC 14.2×10^9/L，中性粒细胞百分比85%。胸部CT示右下肺斑片状高密度影。"
    
    # 提取实体
    result = extractor.extract_and_map(clinical_text)
    
    # 输出结果
    print("临床文本:")
    print(clinical_text)
    print("\n提取的实体:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
