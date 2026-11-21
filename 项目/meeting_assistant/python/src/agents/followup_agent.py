"""
Follow-up Agent（跟进Agent）
- 汇聚 Summary + Action + Unresolved 三个Agent的结果
- 生成并发送会议纪要到飞书群
- 确认所有待办已同步
- 生成待办报告；数据库保存结果后生成跟踪提醒计划
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from ..integrations.feishu_client import FeishuClient
from ..models.schemas import (
    ActionResult,
    FollowUpResult,
    UnresolvedResult,
    MeetingStatus,
    MeetingSummary,
)


class FollowUpAgent:
    """
    跟进Agent - Pipeline的最后一个节点（Fan-in汇聚）

    架构说明:
    1. 等待 Summary/Action/Unresolved 三个并行Agent全部完成
    2. 汇聚结果，生成完整的会议报告
    3. 推送到飞书群
    4. 检查待办同步状态
    5. 提醒由持久化层生成计划、后台定时任务执行

    面试考点:
    - Fan-in 汇聚是如何实现的？（列表边等待三个分支完成后统一执行）
    - 如果某个并行Agent失败了怎么办？（部分降级，跳过失败部分继续）
    - 提醒机制怎么实现？（MySQL持久化计划，后台任务定时扫描并记录发送结果）
    """

    def __init__(self, feishu_client: FeishuClient | None = None):
        self.feishu = feishu_client or FeishuClient()

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 会后跟进

        这是 Pipeline 的最后一个节点，汇聚所有并行结果。
        """
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[FollowUpAgent] Processing meeting: {meeting_id}")

        summary: MeetingSummary | None = state.get("summary")
        actions: ActionResult | None = state.get("actions")
        unresolved: UnresolvedResult | None = state.get("unresolved")

        result = FollowUpResult(meeting_id=meeting_id)
        errors = []

        try:
            # Step 1: 生成会议纪要 Markdown
            summary_md = self._format_summary_markdown(summary)
            actions_md = self._format_actions_markdown(actions)
            unresolved_md = self._format_unresolved_markdown(unresolved)

            # Step 2: 发送到飞书
            if self.feishu.is_enabled:
                title = summary.title if summary else f"会议 {meeting_id}"
                sent = await self.feishu.send_meeting_summary(
                    title=title,
                    summary_md=summary_md,
                    action_items_md=actions_md,
                    unresolved_md=unresolved_md,
                )
                result.summary_sent = sent

            # Step 3: 统计同步结果
            if actions:
                result.jira_issues_created = [
                    item.jira_issue_key
                    for item in actions.action_items
                    if item.jira_issue_key
                ]
                result.feishu_tasks_created = [
                    item.feishu_task_id
                    for item in actions.action_items
                    if item.feishu_task_id
                ]

            # Step 4: 保存会议结果时才真正生成提醒计划，这里不再把日期数冒充提醒数。
            result.reminders_scheduled = 0

            # Step 5: 生成报告
            result.report_markdown = self._generate_report(
                meeting_id, summary_md, actions_md, unresolved_md
            )
            result.report_url = f"/api/v1/meeting/{meeting_id}/report/download"

            logger.info(
                f"[FollowUpAgent] Follow-up complete: "
                f"sent={result.summary_sent}, "
                f"jira_issues={len(result.jira_issues_created)}, "
                f"feishu_tasks={len(result.feishu_tasks_created)}, "
                f"reminders={result.reminders_scheduled}"
            )

        except Exception as e:
            logger.error(f"[FollowUpAgent] Error: {e}")
            errors = [
                f"FollowUpAgent: {str(e)}"
            ]

        # 不重新返回输入的errors，否则reducer会把历史错误重复追加。
        status = MeetingStatus.PARTIAL if errors or state.get("errors") else MeetingStatus.COMPLETED
        return {"followup": result, "status": status, "errors": errors}

    @staticmethod
    def _format_summary_markdown(summary: MeetingSummary | None) -> str:
        """将会议摘要格式化为 Markdown"""
        if not summary:
            return "*（摘要生成失败）*"

        lines = [f"## {summary.title}\n"]

        if summary.participants:
            lines.append(f"**说话人编号**: {', '.join(summary.participants)}\n")

        if summary.speaker_summaries:
            lines.append("### 按说话人分组的要点")
            for group in summary.speaker_summaries:
                lines.append(f"#### {group.speaker_id}")
                lines.extend(f"- {point}" for point in group.points)
            lines.append("")

        for i, topic in enumerate(summary.topics, 1):
            lines.append(f"### 议题{i}: {topic.title}")
            for point in topic.discussion_points:
                lines.append(f"- {point}")
            if topic.conclusion:
                lines.append(f"- **结论**: {topic.conclusion}")
            lines.append("")

        if summary.decisions:
            lines.append("### 会议决策")
            for d in summary.decisions:
                lines.append(f"- {d}")
            lines.append("")

        if summary.next_steps:
            lines.append("### 下一步计划")
            for s in summary.next_steps:
                lines.append(f"- {s}")

        return "\n".join(lines)

    @staticmethod
    def _format_actions_markdown(actions: ActionResult | None) -> str:
        """将待办事项格式化为 Markdown"""
        if not actions or not actions.action_items:
            return "*（无待办事项）*"

        lines = []
        for i, item in enumerate(actions.action_items, 1):
            status_parts = []
            if item.jira_issue_key:
                status_parts.append(f"Jira: {item.jira_issue_key}")
            if item.feishu_task_id:
                status_parts.append(f"飞书: {item.feishu_task_id}")
            status = f" ({', '.join(status_parts)})" if status_parts else ""

            deadline_str = f" | 截止: {item.deadline}" if item.deadline else ""
            lines.append(
                f"{i}. **{item.assignee}**: {item.task}"
                f"{deadline_str} [{item.priority.value}]{status}"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_unresolved_markdown(unresolved: UnresolvedResult | None) -> str:
        """展示仍待明确的问题，附原话和时间位置，方便参会者回查录音。"""
        if unresolved is None:
            return "*（未决事项分析失败）*"
        if not unresolved.issues:
            return "*（未发现需要继续明确的问题）*"

        categories = {"decision": "待决策", "disagreement": "待达成一致",
                      "missing_information": "待补充信息", "dependency": "待确认依赖"}
        lines = []
        for index, issue in enumerate(unresolved.issues, 1):
            lines.extend([
                f"### {index}. {issue.topic}",
                f"- **类型**: {categories[issue.category.value]} | **优先级**: {issue.priority.value}",
                f"- **未决问题**: {issue.description}",
            ])
            if issue.impact:
                lines.append(f"- **影响**: {issue.impact}")
            if issue.suggested_next_step:
                lines.append(f"- **建议下一步**: {issue.suggested_next_step}")
            lines.append("- **原文依据**:")
            for evidence in issue.evidence:
                lines.append(f"  - [{evidence.start:.1f}s-{evidence.end:.1f}s] "
                             f"{evidence.speaker_id}: {evidence.quote}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _generate_report(
        meeting_id: str,
        summary_md: str,
        actions_md: str,
        unresolved_md: str,
    ) -> str:
        """生成报告正文，由数据库保存，下载接口直接返回这份正文。"""
        report = (
            f"# 会议报告 - {meeting_id}\n\n"
            f"生成时间: {datetime.now().isoformat()}\n\n"
            f"---\n\n"
            f"## 会议纪要\n\n{summary_md}\n\n"
            f"---\n\n"
            f"## 待办事项\n\n{actions_md}\n\n"
            f"---\n\n"
            f"## 未决事项\n\n{unresolved_md}\n"
        )
        logger.info(f"Report generated for meeting {meeting_id}")
        return report
