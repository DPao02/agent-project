"""只检查百炼 Embedding 连接，不操作 Milvus。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

with OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    timeout=20,
    max_retries=0,
) as client:
    result = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
        input="申请退款需要提供订单号",
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
        encoding_format="float",
    )

    print("百炼调用成功，向量维度：", len(result.data[0].embedding))