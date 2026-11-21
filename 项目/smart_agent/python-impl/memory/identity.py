"""所有记忆和图检查点使用相同的用户/会话命名空间。"""

import hashlib
import json


def session_scope(user_id: str, session_id: str) -> str:
    if not user_id.strip() or not session_id.strip():
        raise ValueError("user_id和session_id不能为空")
    # 不能简单拼接 user:session，否则 (a:b,c) 与 (a,b:c) 会碰撞。
    identity = json.dumps([user_id.strip(), session_id.strip()], ensure_ascii=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
