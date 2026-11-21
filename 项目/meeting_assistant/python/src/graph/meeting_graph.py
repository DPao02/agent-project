"""
LangGraph 会议处理图 —— 多Agent编排核心

编排模式: Pipeline + 并行 (Fan-out / Fan-in)

    ┌─────────────┐
    │   START     │
    └──────┬──────┘
           │
           ▼
    ┌──────────────┐
    │ Transcription│  ← Pipeline 阶段
    │    Agent     │
    └──────┬───────┘
           │
    ┌──────┼───────┐  ← Fan-out (并行)
    │      │       │
    ▼      ▼       ▼
  Summary Action Unresolved
  Agent   Agent  Agent
    │      │       │
    └──────┼───────┘  ← Fan-in (汇聚)
           │
           ▼
    ┌──────────────┐
    │  Follow-up   │
    │    Agent     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │     END      │
    └──────────────┘

面试考点:
- LangGraph 的 State/Node/Edge 分别是什么？
- 并行执行是怎么实现的？（Fan-out + Fan-in）
- 如果某个并行节点失败了怎么办？（错误写入state，不阻塞其他节点）
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.graph import StateGraph, START, END
from loguru import logger

from ..agents.transcription_agent import TranscriptionAgent, TranscriptionConfig
from ..agents.summary_agent import SummaryAgent
from ..agents.action_agent import ActionAgent
from ..agents.unresolved_agent import UnresolvedAgent
from ..agents.followup_agent import FollowUpAgent
from ..integrations.deepseek_client import LLMClient
from ..integrations.jira_client import JiraClient
from ..integrations.feishu_client import FeishuClient
from ..config import load_environment
from ..models.schemas import (
    MeetingState,
    MeetingStatus,
    MeetingMode,
    TranscriptResult,
    create_initial_state,
)


# ============================================================
# 状态统一定义在models/schemas.py，避免API模型和图结构各维护一份。
# ============================================================

GraphState = MeetingState


# ============================================================
# 构建 Meeting Graph
# ============================================================

def build_meeting_graph(
    llm_client: LLMClient | None = None,
    jira_client: JiraClient | None = None,
    feishu_client: FeishuClient | None = None,
    transcription_config: TranscriptionConfig | None = None,
    transcription_agent: TranscriptionAgent | None = None,
) -> StateGraph:
    """
    构建会议处理 StateGraph

    这是整个系统的编排核心：
    1. 创建 5 个 Agent 实例
    2. 将它们注册为 Graph 的 Node
    3. 定义 Edge（流转关系）
    4. 编译为可执行的 Graph

    Args:
        llm_client: LLM 客户端（共享，避免重复创建）
        jira_client: Jira 客户端
        feishu_client: 飞书客户端
        transcription_config: 转写配置

    Returns:
        未编译的StateGraph，由compile_meeting_graph统一编译
    """
    # 共享依赖
    llm = llm_client or LLMClient()
    jira = jira_client or JiraClient()
    feishu = feishu_client or FeishuClient()

    # 创建 Agent 实例
    transcription_agent = transcription_agent or TranscriptionAgent(transcription_config)
    summary_agent = SummaryAgent(llm)
    action_agent = ActionAgent(llm, jira, feishu)
    unresolved_agent = UnresolvedAgent(llm)
    followup_agent = FollowUpAgent(feishu)

    # ---- 构建 StateGraph ----
    graph = StateGraph(GraphState)

    # 注册节点（Node = Agent）
    graph.add_node("transcription", transcription_agent.process)
    graph.add_node("summary", summary_agent.process)
    graph.add_node("action", action_agent.process)
    graph.add_node("unresolved", unresolved_agent.process)
    graph.add_node("followup", followup_agent.process)

    # ---- 定义边（Edge = 流转关系）----

    # Pipeline 阶段: START → Transcription
    graph.add_edge(START, "transcription")

    # Fan-out 并行: Transcription → [Summary, Action, Unresolved]
    # 真实转写失败直接结束，不允许用空转写继续生成纪要或外部任务。
    def after_transcription(state):
        return END if state["status"] == MeetingStatus.FAILED else ["summary", "action", "unresolved"]

    graph.add_conditional_edges("transcription", after_transcription, {
        "summary": "summary", "action": "action", "unresolved": "unresolved", END: END,
    })

    # Fan-in 汇聚: [Summary, Action, Unresolved] → Follow-up
    # 显式等待三个分支全部结束，再执行一次跟进节点。
    graph.add_edge(["summary", "action", "unresolved"], "followup")

    # 结束: Follow-up → END
    graph.add_edge("followup", END)

    logger.info("Meeting graph built successfully")
    return graph


def compile_meeting_graph(**kwargs) -> Any:
    """构建并编译 Graph（编译后可直接调用）"""
    graph = build_meeting_graph(**kwargs)
    compiled = graph.compile()
    logger.info("Meeting graph compiled successfully")
    return compiled


async def run_meeting_pipeline(
    meeting_id: str,
    audio_data: bytes = b"",
    *,
    mode: MeetingMode | str = MeetingMode.REAL,
    **kwargs,
) -> dict:
    """供独立脚本单次调用；API使用lifespan中创建的共享MeetingPipeline。"""
    async with MeetingPipeline(**kwargs) as pipeline:
        return await pipeline.run(meeting_id, audio_data, mode=mode)


class MeetingPipeline:
    """共享执行器，不在实例字段中保存任何一场会议的输入或结果。"""
    def __init__(self, llm_client=None, jira_client=None, feishu_client=None,
                 transcription_config=None, transcription_agent=None):
        load_environment()
        self._owns_llm = llm_client is None
        self._owns_feishu = feishu_client is None
        self._owns_jira = jira_client is None
        self.llm = llm_client if llm_client is not None else LLMClient()
        self.feishu = feishu_client if feishu_client is not None else FeishuClient()
        self.jira = jira_client if jira_client is not None else JiraClient()
        self.transcriber = transcription_agent or TranscriptionAgent(transcription_config)
        self.graph = compile_meeting_graph(
            llm_client=self.llm, jira_client=self.jira, feishu_client=self.feishu,
            transcription_agent=self.transcriber,
        )

    async def run(self, meeting_id: str, audio_data: bytes = b"", *, mode=MeetingMode.REAL,
                  transcript: TranscriptResult | None = None, transcribe_only=False):
        # 每次都新建State；共享的是执行逻辑和模型权重，不是历史或音频缓存。
        state = create_initial_state(meeting_id, audio_data, mode=mode, transcript=transcript)
        if transcribe_only:
            return {**state, **await self.transcriber.process(state)}
        return await self.graph.ainvoke(state)

    async def close(self):
        try:
            if self._owns_llm:
                await self.llm.close()
        finally:
            try:
                if self._owns_feishu:
                    await self.feishu.close()
            finally:
                if self._owns_jira:
                    await asyncio.to_thread(self.jira.close)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
