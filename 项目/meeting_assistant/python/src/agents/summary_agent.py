"""
Summary Agent（摘要Agent）
- 接收转写文本，生成结构化会议纪要
- 使用 LLM 进行内容提取和组织
- 输出: 议题/讨论要点/结论/决策 四层结构
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from ..integrations.deepseek_client import LLMClient
from ..models.schemas import (
    MeetingStatus,
    MeetingSummary,
    TopicSummary,
    SpeakerSummary,
)


SUMMARY_SYSTEM_PROMPT = """你是一位专业的会议纪要助手。你的任务是根据会议转写文本，生成清晰、结构化的会议纪要。

要求：
1. 准确提取会议中的每个议题
2. 每个议题包含讨论要点、参与人、结论
3. 明确列出会议做出的决策
4. 列出下一步行动计划
5. 使用中文输出
6. 按原始说话人编号组织输出：只使用SPEAKER_00等实际出现的编号，未知者使用UNKNOWN。
7. 分别总结每个编号的发言要点；不要猜测编号对应的姓名、领导/同事身份或职务。

你必须严格按照以下JSON格式输出，不要添加任何其他文字："""

SUMMARY_USER_PROMPT = """请根据以下会议转写文本生成结构化会议纪要。

## 会议转写文本
{transcript}

## 输出格式（严格JSON）
{{
  "title": "会议主题（从内容推断）",
  "date": "会议日期（如无法确定则写今天）",
  "participants": ["SPEAKER_00", "SPEAKER_01"],
  "speaker_summaries": [
    {{"speaker_id": "SPEAKER_00", "points": ["该编号提出的要点"]}}
  ],
  "topics": [
    {{
      "title": "议题名称",
      "discussion_points": ["要点1", "要点2"],
      "participants": ["SPEAKER_00"],
      "conclusion": "该议题的结论"
    }}
  ],
  "decisions": ["决策1", "决策2"],
  "next_steps": ["下一步1", "下一步2"]
}}"""


class SummaryAgent:
    """
    摘要Agent - 并行阶段的节点之一

    架构说明:
    1. 从 state 读取 transcript_text
    2. 构造 Few-shot Prompt 调用 LLM
    3. 约束 JSON Schema 输出格式
    4. 解析并验证结果
    5. 写入 state["summary"]

    面试考点:
    - Prompt 设计策略？（System Prompt + Few-shot + JSON Schema约束）
    - 如何保证输出格式正确？（response_format + 解析降级）
    - 长文本如何处理？（分块摘要 + 合并，MapReduce策略）
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 生成会议摘要

        与 Action Agent、Unresolved Agent 并行执行。
        """
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[SummaryAgent] Processing meeting: {meeting_id}")

        transcript_text = state.get("transcript_text", "")
        if not transcript_text:
            logger.warning("[SummaryAgent] No transcript text available")
            return {"summary": MeetingSummary(
                title="未知会议", date="", participants=[], topics=[],
                decisions=[], next_steps=[],
            )}

        # 只返回本节点负责的字段，不能把meeting_id等输入再写回并行图。
        speaker_ids = list(getattr(state.get("transcript"), "speaker_ids", []))
        if not speaker_ids:
            speaker_ids = list(dict.fromkeys(re.findall(r"\b(?:SPEAKER_\d+|UNKNOWN)\b", transcript_text)))
        errors = []
        try:
            summary = await self._generate_summary(transcript_text, speaker_ids)
            logger.info(
                f"[SummaryAgent] Summary generated: {summary.title}, "
                f"{len(summary.topics)} topics"
            )
        except Exception as e:
            logger.error(f"[SummaryAgent] Error: {e}")
            errors = [
                f"SummaryAgent: {str(e)}"
            ]
            summary = self._generate_fallback_summary(transcript_text, speaker_ids)

        return {"summary": summary, "errors": errors}

    async def _generate_summary(self, transcript: str, speaker_ids: list[str]) -> MeetingSummary:
        """调用 LLM 生成结构化摘要"""
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SUMMARY_USER_PROMPT.format(transcript=transcript),
            },
        ]

        result = await self.llm.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )

        # 标签由实际转写约束，不能把模型猜出的姓名或编号当成参会身份。
        allowed = set(speaker_ids)
        topics = [TopicSummary(**{**topic, "participants": [speaker for speaker in topic.get("participants", [])
                                                           if speaker in allowed]})
                  for topic in result.get("topics", [])]
        grouped = {group["speaker_id"]: group.get("points", []) for group in result.get("speaker_summaries", [])
                   if group.get("speaker_id") in allowed}

        return MeetingSummary(
            title=result.get("title", "会议纪要"),
            date=result.get("date", ""),
            participants=speaker_ids,
            speaker_summaries=[SpeakerSummary(speaker_id=speaker, points=grouped.get(speaker, [])) for speaker in speaker_ids],
            topics=topics,
            decisions=result.get("decisions", []),
            next_steps=result.get("next_steps", []),
        )

    @staticmethod
    def _generate_fallback_summary(transcript: str, speaker_ids: list[str]) -> MeetingSummary:
        """LLM 调用失败时的降级方案 —— 基于规则的简单摘要"""
        return MeetingSummary(
            title="会议纪要（自动摘要降级模式）",
            date="",
            participants=speaker_ids,
            speaker_summaries=[SpeakerSummary(speaker_id=speaker) for speaker in speaker_ids],
            topics=[
                TopicSummary(
                    title="会议内容",
                    discussion_points=["（LLM调用失败，请查看原始转写文本）"],
                    participants=speaker_ids,
                    conclusion="",
                )
            ],
            decisions=[],
            next_steps=[],
        )
