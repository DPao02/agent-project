"""线上入库与离线评估共用分块和ID算法，避免评测使用另一份语料。"""

import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHUNK_SETTINGS = {"chunk_size": 512, "chunk_overlap": 128,
                  "separators": ["\n\n", "\n", "。", "！", "？", "；", " ", ""]}


def document_id(model: str, source: str, content: str) -> str:
    return hashlib.sha256(f"{model}\n{source}\n{content.strip()}".encode("utf-8")).hexdigest()


def knowledge_chunks(directory: str | Path, model: str) -> list[dict]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"知识库目录不存在：{root}")
    splitter = RecursiveCharacterTextSplitter(**CHUNK_SETTINGS)
    documents = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".txt"}:
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf":
            pages = [(number, page.extract_text() or "")
                     for number, page in enumerate(PdfReader(str(path)).pages, 1)]
        else:
            pages = [(None, path.read_text(encoding="utf-8-sig"))]
        for number, text in pages:
            source = relative if number is None else f"{relative}#page={number}"
            for index, content in enumerate(splitter.split_text(text), 1):
                content = content.strip()
                if content:
                    documents.append({"id": document_id(model, source, content), "content": content,
                        "source": source, "metadata": {"file": relative, "page": number, "chunk": index},
                        "embedding_model": model})
    return documents
