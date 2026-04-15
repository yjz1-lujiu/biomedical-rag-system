# migrate_history.py
import os
import json
from session_manager import SessionManager, ChromaChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, messages_from_dict

OLD_JSON_DIR = "./chat_histories"
NEW_SQLITE_DB = "./sessions.db"
NEW_CHROMA_DIR = "./chat_histories_chroma"


def migrate():
    manager = SessionManager(db_path=NEW_SQLITE_DB, chroma_dir=NEW_CHROMA_DIR)
    for filename in os.listdir(OLD_JSON_DIR):
        if not filename.endswith(".json"):
            continue
        session_id = filename.replace(".json", "")
        filepath = os.path.join(OLD_JSON_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = messages_from_dict(data)
        # 创建会话记录
        manager.store.create_session(session_id=session_id, title=f"旧对话 {session_id[:8]}")
        # 写入 ChromaDB
        history = ChromaChatMessageHistory(session_id, NEW_CHROMA_DIR)
        history.add_messages(messages)
        print(f"迁移完成：{session_id}，消息数：{len(messages)}")


if __name__ == "__main__":
    migrate()