"""有界的模型上下文：保留最近对话，避免只有最后一句或无限堆积历史。"""

from langchain_core.messages import AIMessage, HumanMessage


def recent_messages(messages: list, max_messages: int = 12, max_chars: int = 16000) -> list:
    result = []
    remaining = max_chars
    for message in reversed(messages):
        if not isinstance(message, (HumanMessage, AIMessage)) or remaining <= 0:
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        if len(content) <= remaining:
            text = content
        else:
            prefix = "[历史内容截断]"
            text = prefix + content[-(remaining - len(prefix)):] if remaining > len(prefix) else content[-remaining:]
        result.append(message.model_copy(update={"content": text}))
        remaining -= len(text)
        if len(result) >= max_messages:
            break
    return list(reversed(result))


def latest_user_text(messages: list) -> str:
    return next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
