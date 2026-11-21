"""待办提醒：日期规则、后台扫描和发送。业务状态及发送记录都保存在MySQL。

北京时间截止前3天/1天的09:00各提醒一次。每个提醒只在对应日有效，
服务恢复后补发当天尚未发送的提醒，过期窗口跳过，不集中补发历史提醒。
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
import os

import httpx
from loguru import logger


BEIJING = timezone(timedelta(hours=8))


def reminder_plan(deadline: str, *, hour: int = 9):
    """计划使用UTC存库，接口时间带UTC偏移；deadline仍表示北京时间日期。"""
    if not deadline:
        return []
    due_date = date.fromisoformat(deadline)
    result = []
    for days_before in (3, 1):
        day = due_date - timedelta(days=days_before)
        scheduled = datetime.combine(day, time(hour=hour), BEIJING)
        expires = datetime.combine(day + timedelta(days=1), time(), BEIJING)
        result.append((days_before,
                       scheduled.astimezone(timezone.utc).replace(tzinfo=None),
                       expires.astimezone(timezone.utc).replace(tzinfo=None)))
    return result


class ReminderService:
    """应用启动时创建一份扫描器；每轮从数据库领取任务，不靠进程内列表记住已发送任务。"""

    def __init__(self, repository, feishu_client, *, interval=None):
        self.repository = repository
        self.feishu = feishu_client
        self.interval = float(interval if interval is not None else os.getenv("REMINDER_POLL_SECONDS", "60"))
        if self.interval <= 0:
            raise ValueError("REMINDER_POLL_SECONDS必须大于0")
        self._stop = asyncio.Event()
        self._task = None

    def start(self):
        self._task = asyncio.create_task(self._loop(), name="meeting-reminders")

    async def close(self):
        self._stop.set()
        if self._task is not None:
            await self._task

    async def _loop(self):
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("Reminder scan failed: {}", type(exc).__name__)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except TimeoutError:
                pass

    async def run_once(self, *, now=None, meeting_id=None):
        """测试可传入时钟，验收提醒日期时不需要修改电脑时间。"""
        count = 0
        # 每次只领取即将发送的一条，避免批量领取后关机让其余任务悬在sending。
        for _ in range(100):
            if self._stop.is_set():
                break
            jobs = await asyncio.to_thread(self.repository.claim_due_reminders,
                                           now=now, limit=1, meeting_id=meeting_id)
            if not jobs:
                break
            job = jobs[0]
            # 领取到发送之间用户可能已经完成任务，发送前再查一次当前状态。
            if not await asyncio.to_thread(self.repository.reminder_is_active,
                                           job["reminder_id"], job["claim_token"], now=now):
                continue
            message = (f"会议：{job['title'] or job['meeting_id']}\n"
                       f"负责人：{job['assignee']}\n待办：{job['task']}\n"
                       f"截止日期：{job['deadline']}（提前{job['days_before']}天提醒）")
            channel = "in_app"
            status, error = "sent", None
            result = {"success": True, "channel": channel, "message": message}
            try:
                # 演示数据留在站内；真实会议配置飞书Webhook后推送到群。
                if (job["mode"] == "real" and self.feishu.is_enabled
                        and getattr(self.feishu, "webhook_url", "")):
                    channel = "feishu"
                    result = await self.feishu.send_action_reminder(message)
                    if not result.get("success"):
                        status, error = "failed", "飞书明确返回发送失败"
            except (httpx.TransportError, TimeoutError):
                # 请求可能已到达飞书，响应丢失时不自动重发同一提醒。
                status, error = "uncertain", "网络异常，发送结果待核对"
                result = {"success": False, "message": message}
            except Exception as exc:
                status, error = "failed", f"发送失败：{type(exc).__name__}"
                result = {"success": False, "message": message}
            await asyncio.to_thread(self.repository.finish_reminder, job["reminder_id"],
                job["claim_token"], status=status, channel=channel, result=result, error=error, now=now)
            count += 1
        return count
