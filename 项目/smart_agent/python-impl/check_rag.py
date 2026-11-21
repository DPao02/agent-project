"""
独立检查知识库导入和检索。

这个脚本会实际调用百炼 Embedding，但不会调用 DeepSeek。
用来区分向量检索问题和聊天编排问题。
"""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
# 本地测试时优先使用当前项目配置，避免终端旧环境变量覆盖它。
load_dotenv(PROJECT_DIR / ".env", override=True)

# 必须先加载环境变量，再创建知识库对象。
from memory.long_term import LongTermMemory


def main():
    memory = LongTermMemory()

    try:
        print(f"目标集合：{memory.collection_name}")

        count = memory.load_knowledge_base(
            str(PROJECT_DIR / "knowledge_base")
        )
        if count == 0:
            raise RuntimeError("未提取到知识库文本")

        memory.save()
        print(f"知识库处理完成，共 {count} 个文本块")

        query = "申请退款需要满足哪些条件？"
        results = memory.search(query, top_k=3)

        print(f"\n检索问题：{query}")

        for number, doc in enumerate(results, start=1):
            print(f"\n结果 {number}")
            print("来源：", doc["source"])
            print("相似度：", round(doc["score"], 4))
            print("正文：", doc["content"][:400])

    finally:
        memory.close()


if __name__ == "__main__":
    main()