#!/usr/bin/env python3
"""
合规性审计日志
- 记录完整上下文
- 持久化到PostgreSQL
- 支持日志查询和导出
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# 尝试导入PostgreSQL
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

# ====================== 环境配置 ======================
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/rag_audit")

# ====================== 审计日志管理器 ======================
class AuditLogger:
    def __init__(self, db_url: str = None):
        self.db_url = db_url or DATABASE_URL
        self.use_postgres = PSYCOPG2_AVAILABLE
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        if self.use_postgres:
            try:
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        # 创建审计日志表
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS audit_logs (
                                log_id TEXT PRIMARY KEY,
                                session_id TEXT,
                                timestamp TIMESTAMP,
                                user_query TEXT,
                                intermediate_steps JSONB,
                                final_answer TEXT,
                                cited_evidence JSONB,
                                overall_confidence FLOAT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """)
                        conn.commit()
            except Exception as e:
                print(f"初始化PostgreSQL失败: {e}")
                self.use_postgres = False
        
        # 创建本地日志目录作为备份
        self.log_dir = "./audit_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def create_log(self, session_id: str, user_query: str, intermediate_steps: Dict[str, Any],
                   final_answer: str, cited_evidence: List[Dict[str, Any]], overall_confidence: float) -> Dict[str, Any]:
        """创建审计日志"""
        log = {
            "log_id": str(uuid.uuid4()),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "user_query": self._sanitize_pii(user_query),  # 脱敏处理
            "intermediate_steps": intermediate_steps,
            "final_answer": final_answer,
            "cited_evidence": cited_evidence,
            "overall_confidence": overall_confidence
        }
        
        # 保存到PostgreSQL
        if self.use_postgres:
            try:
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO audit_logs (log_id, session_id, timestamp, user_query, 
                                                  intermediate_steps, final_answer, cited_evidence, 
                                                  overall_confidence)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                log["log_id"],
                                log["session_id"],
                                datetime.now(),
                                log["user_query"],
                                json.dumps(log["intermediate_steps"], ensure_ascii=False),
                                log["final_answer"],
                                json.dumps(log["cited_evidence"], ensure_ascii=False),
                                log["overall_confidence"]
                            )
                        )
                        conn.commit()
            except Exception as e:
                print(f"保存到PostgreSQL失败: {e}")
        
        # 保存到本地文件作为备份
        log_file = os.path.join(self.log_dir, f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        
        return log
    
    def get_logs(self, session_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取日志"""
        logs = []
        
        # 从PostgreSQL获取
        if self.use_postgres:
            try:
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        if session_id:
                            cur.execute(
                                """
                                SELECT * FROM audit_logs 
                                WHERE session_id = %s 
                                ORDER BY timestamp DESC 
                                LIMIT %s
                                """,
                                (session_id, limit)
                            )
                        else:
                            cur.execute(
                                """
                                SELECT * FROM audit_logs 
                                ORDER BY timestamp DESC 
                                LIMIT %s
                                """,
                                (limit,)
                            )
                        rows = cur.fetchall()
                        for row in rows:
                            log = dict(row)
                            log["timestamp"] = log["timestamp"].isoformat()
                            log["created_at"] = log["created_at"].isoformat()
                            logs.append(log)
            except Exception as e:
                print(f"从PostgreSQL获取日志失败: {e}")
        
        # 如果PostgreSQL失败，从本地文件获取
        if not logs and session_id:
            import glob
            session_files = glob.glob(os.path.join(self.log_dir, f"{session_id}_*.json"))
            session_files.sort(reverse=True)[:limit]
            for file_path in session_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        log = json.load(f)
                        logs.append(log)
                except:
                    pass
        
        return logs
    
    def export_logs(self, session_id: str = None, format: str = "json") -> str:
        """导出日志"""
        logs = self.get_logs(session_id)
        
        if format == "json":
            export_file = os.path.join(self.log_dir, f"export_{session_id or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            return export_file
        elif format == "csv":
            import csv
            export_file = os.path.join(self.log_dir, f"export_{session_id or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            with open(export_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(["log_id", "session_id", "timestamp", "user_query", "final_answer", "overall_confidence"])
                # 写入数据
                for log in logs:
                    writer.writerow([
                        log.get("log_id"),
                        log.get("session_id"),
                        log.get("timestamp"),
                        log.get("user_query"),
                        log.get("final_answer"),
                        log.get("overall_confidence")
                    ])
            return export_file
        
        return None
    
    def _sanitize_pii(self, text: str) -> str:
        """脱敏处理，移除患者个人身份信息"""
        # 这里可以添加更复杂的PII检测和脱敏逻辑
        import re
        
        # 移除身份证号
        text = re.sub(r"\d{17}[\dXx]", "[身份证号]", text)
        
        # 移除手机号
        text = re.sub(r"1[3-9]\d{9}", "[手机号]", text)
        
        # 移除姓名（简单处理，实际应用中可能需要更复杂的逻辑）
        # 这里只是一个示例，实际应用中可能需要更精确的姓名识别
        
        return text

# ====================== 主函数 ======================
def main():
    """测试审计日志功能"""
    logger = AuditLogger()
    
    # 测试创建日志
    test_log = logger.create_log(
        session_id="test_session",
        user_query="患者，男，65岁，因发热、咳嗽3天入院",
        intermediate_steps={
            "planner": {"subtasks": ["检索相关医学知识", "分析临床证据"]},
            "retrieved_evidence_count": 5,
            "graded_evidence_count": 5
        },
        final_answer="根据症状和检查结果，考虑为上呼吸道感染",
        cited_evidence=[
            {"id": "evidence1", "source": "internal", "grade": "2a"},
            {"id": "evidence2", "source": "pubmed", "grade": "1b"}
        ],
        overall_confidence=0.85
    )
    
    print("创建的日志:")
    print(json.dumps(test_log, ensure_ascii=False, indent=2))
    
    # 测试获取日志
    logs = logger.get_logs("test_session")
    print("\n获取的日志:")
    print(f"共 {len(logs)} 条日志")
    
    # 测试导出日志
    export_file = logger.export_logs("test_session", format="json")
    print(f"\n导出的日志文件: {export_file}")

if __name__ == "__main__":
    main()
