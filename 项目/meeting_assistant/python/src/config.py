"""本地配置加载：会议项目.env优先，缺失模型配置再从工作区补充。"""

from pathlib import Path
import os

from dotenv import dotenv_values, load_dotenv

#以当前文件为基准，找向上2级的父目录，也就是项目根目录
PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_environment() -> None:
    """使用明确路径，保证 python -m 和 uvicorn 两种入口读取同一份配置。

    本项目位于共享 uv 工作区时，允许复用工作区 .env 中已有的 DeepSeek
    密钥。项目.env优先于依赖提前加载的环境变量，与smart_agent保持一致。
    """
    # pymilvus导入时会自动load_dotenv()，可能先加载工作区的另一把密钥。
    # 显式覆盖才能保证实际请求使用用户在会议项目.env中填写的值。
    load_dotenv(PROJECT_DIR / ".env", override=True)
    for parent in PROJECT_DIR.parents:
        if (parent / "pyproject.toml").is_file():
            # 模板中空的DEEPSEEK_API_KEY也视作未配置，可以复用工作区密钥。
            # 只复用模型配置，避免把其他项目的追踪或业务开关带进来。
            shared = dotenv_values(parent / ".env")
            for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"):
                value = shared.get(name)
                if value is not None and not os.environ.get(name):
                    os.environ[name] = value
            break
