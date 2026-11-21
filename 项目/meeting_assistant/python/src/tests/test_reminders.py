"""用可控时钟验收真实存库、两次提醒、完成即停、重启与发送结果，不调用真实外部平台。"""
import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from src.integrations.feishu_client import FeishuClient
from src.models.schemas import ActionItem, ActionResult, FollowUpResult, MeetingStatus, create_initial_state
from src.reminders import BEIJING, ReminderService, reminder_plan
from src.storage import ReminderRecord, StorageError, serialize_state
from src.websocket.server import create_app
from test_meeting_features import DisabledVectors, pipeline, repository
from test_pipeline import DisabledIntegration


_tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})


def setUpModule():
    _tracing.start()


def tearDownModule():
    _tracing.stop()


def local_day(day, hour=9):
    return datetime(2099, 5, day, hour, tzinfo=BEIJING)


def save_actions(repo, *, meeting_id="reminder-test", deadline="2099-05-10", mode="real"):
    repo.create(meeting_id, "alice", allow_existing=True)
    run = repo.claim(meeting_id, "alice", mode)
    state = create_initial_state(meeting_id, mode=mode)
    state.update(status=MeetingStatus.COMPLETED,
        actions=ActionResult(meeting_id=meeting_id, action_items=[
            ActionItem(assignee="李明", task="整理预算", deadline=deadline)]),
        followup=FollowUpResult(meeting_id=meeting_id))
    return repo.finish(meeting_id, "alice", run["run_token"], serialize_state(state))


class FakeFeishu:
    is_enabled = True
    webhook_url = "https://example.test/hook"

    def __init__(self, *, fail=None):
        self.messages = []
        self.fail = fail

    async def send_action_reminder(self, message):
        self.messages.append(message)
        if self.fail:
            raise self.fail
        return {"success": True, "message": message, "code": 0}


class ReminderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = repository()
        self.record = save_actions(self.repo)
        self.action_id = self.record["result"]["actions"]["action_items"][0]["action_id"]

    def tearDown(self):
        self.repo.close()

    async def test_exact_dates_no_early_send_and_only_two_reminders(self):
        sender = FakeFeishu()
        worker = ReminderService(self.repo, sender)
        self.assertEqual(await worker.run_once(now=local_day(7, 8)), 0)
        self.assertEqual(await worker.run_once(now=local_day(7, 9)), 1)
        self.assertEqual(await worker.run_once(now=local_day(7, 10)), 0)
        self.assertEqual(await worker.run_once(now=local_day(8)), 0)
        self.assertEqual(await worker.run_once(now=local_day(9)), 1)
        self.assertEqual(await worker.run_once(now=local_day(10)), 0)
        self.assertEqual(len(sender.messages), 2)
        rows = self.repo.list_reminders("reminder-test", "alice")
        self.assertEqual([r["status"] for r in rows], ["sent", "sent"])
        self.assertTrue(all(r["attempts"] == 1 and r["sent_at"] for r in rows))
        self.assertIn("李明", rows[0]["send_result"]["message"])
        self.assertEqual(self.repo.get("reminder-test", "alice")["result"]["followup"]["reminders_scheduled"], 0)

    async def test_completed_task_stops_day_one_reminder(self):
        sender = FakeFeishu()
        worker = ReminderService(self.repo, sender)
        await worker.run_once(now=local_day(7))
        self.repo.update_action("reminder-test", "alice", self.action_id,
                                status="completed", now=local_day(8))
        self.assertEqual(await worker.run_once(now=local_day(9)), 0)
        self.assertEqual(len(sender.messages), 1)
        rows = self.repo.list_reminders("reminder-test", "alice")
        self.assertEqual([r["status"] for r in rows], ["sent", "cancelled"])
        action = self.repo.get("reminder-test", "alice")["result"]["actions"]["action_items"][0]
        self.assertEqual(action["status"], "completed")
        self.assertIsNotNone(action["completed_at"])

    async def test_completion_after_claim_prevents_delivery(self):
        job = self.repo.claim_due_reminders(now=local_day(7))[0]
        self.repo.update_action("reminder-test", "alice", self.action_id, status="completed", now=local_day(7))
        self.assertFalse(self.repo.reminder_is_active(job["reminder_id"], job["claim_token"], now=local_day(7)))

    async def test_completion_during_send_still_records_actual_result(self):
        repo, action_id = self.repo, self.action_id
        class CompletingSender(FakeFeishu):
            async def send_action_reminder(self, message):
                repo.update_action("reminder-test", "alice", action_id, status="completed", now=local_day(7))
                return await super().send_action_reminder(message)
        await ReminderService(self.repo, CompletingSender()).run_once(now=local_day(7))
        self.assertEqual([r["status"] for r in self.repo.list_reminders("reminder-test", "alice")],
                         ["sent", "cancelled"])

    async def test_expired_window_skipped_and_later_reminder_still_sent(self):
        sender = FakeFeishu()
        worker = ReminderService(self.repo, sender)
        self.assertEqual(await worker.run_once(now=local_day(8)), 0)
        self.assertEqual(await worker.run_once(now=local_day(9, 18)), 1)
        self.assertEqual([r["status"] for r in self.repo.list_reminders("reminder-test", "alice")],
                         ["skipped", "sent"])

    async def test_timeout_records_uncertain_and_does_not_resend(self):
        sender = FakeFeishu(fail=httpx.ReadTimeout("response missing"))
        worker = ReminderService(self.repo, sender)
        await worker.run_once(now=local_day(7))
        await worker.run_once(now=local_day(7, 11))
        self.assertEqual(len(sender.messages), 1)
        row = self.repo.list_reminders("reminder-test", "alice")[0]
        self.assertEqual(row["status"], "uncertain")
        self.assertIsNotNone(row["last_error"])

    async def test_rejection_records_failure_without_success_count(self):
        class Rejecting(FakeFeishu):
            async def send_action_reminder(self, message):
                self.messages.append(message)
                return {"success": False, "code": 19001}
        sender = Rejecting()
        worker = ReminderService(self.repo, sender)
        await worker.run_once(now=local_day(7))
        await worker.run_once(now=local_day(7, 11))
        row = self.repo.list_reminders("reminder-test", "alice")[0]
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["sent_at"])
        self.assertEqual(len(sender.messages), 1)

    async def test_cancel_reschedule_and_missing_deadline(self):
        self.repo.update_action("reminder-test", "alice", self.action_id, deadline="2099-05-11")
        rows = self.repo.list_reminders("reminder-test", "alice")
        self.assertEqual(sum(r["status"] == "cancelled" for r in rows), 2)
        self.assertEqual(sum(r["status"] == "pending" for r in rows), 2)
        self.repo.update_action("reminder-test", "alice", self.action_id, deadline="")
        self.assertTrue(all(r["status"] == "cancelled" for r in self.repo.list_reminders("reminder-test", "alice")))
        self.assertEqual(await ReminderService(self.repo, FakeFeishu()).run_once(now=local_day(9)), 0)

    async def test_reanalysis_preserves_id_completed_status_and_plan_uniqueness(self):
        self.repo.update_action("reminder-test", "alice", self.action_id, status="completed")
        latest = save_actions(self.repo)
        item = latest["result"]["actions"]["action_items"][0]
        self.assertEqual(item["action_id"], self.action_id)
        self.assertEqual(item["status"], "completed")
        self.assertEqual(len(self.repo.list_reminders("reminder-test", "alice")), 2)
        self.assertEqual(latest["result"]["followup"]["reminders_scheduled"], 0)

    async def test_demo_and_disabled_integration_deliver_in_app(self):
        record = save_actions(self.repo, meeting_id="demo-only", mode="demo")
        sender = FakeFeishu()
        # 真实会议也可在未配置飞书时使用站内提醒。
        await ReminderService(self.repo, DisabledIntegration()).run_once(now=local_day(7))
        self.assertEqual(self.repo.list_reminders("reminder-test", "alice")[0]["channel"], "in_app")
        self.assertEqual(self.repo.list_reminders("demo-only", "alice")[0]["channel"], "in_app")
        self.assertEqual(sender.messages, [])

    async def test_demo_never_pushes_configured_feishu(self):
        save_actions(self.repo, meeting_id="demo-only", mode="demo")
        sender = FakeFeishu()
        await ReminderService(self.repo, sender).run_once(now=local_day(7))
        self.assertEqual(len(sender.messages), 1)  # 只有real会议进入外部通道。
        self.assertEqual(self.repo.list_reminders("demo-only", "alice")[0]["channel"], "in_app")

    async def test_two_claimants_and_abandoned_send_do_not_duplicate(self):
        first = self.repo.claim_due_reminders(now=local_day(7))
        self.assertEqual(len(first), 1)
        self.assertEqual(self.repo.claim_due_reminders(now=local_day(7)), [])
        self.assertEqual(self.repo.claim_due_reminders(now=local_day(7, 10)), [])
        self.assertEqual(self.repo.list_reminders("reminder-test", "alice")[0]["status"], "uncertain")

    async def test_scoped_scan_leaves_other_meetings_pending(self):
        save_actions(self.repo, meeting_id="other-meeting")
        await ReminderService(self.repo, DisabledIntegration()).run_once(
            now=local_day(7), meeting_id="reminder-test")
        self.assertEqual(self.repo.list_reminders("reminder-test", "alice")[0]["status"], "sent")
        self.assertEqual(self.repo.list_reminders("other-meeting", "alice")[0]["status"], "pending")

    async def test_busy_meeting_does_not_block_other_meeting_reminders(self):
        save_actions(self.repo, meeting_id="other-meeting")
        self.repo.claim("reminder-test", "alice", "real")
        jobs = self.repo.claim_due_reminders(now=local_day(7), limit=1)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["meeting_id"], "other-meeting")

    async def test_ownership_is_checked_on_update_and_reminder_read(self):
        for operation in (lambda: self.repo.update_action("reminder-test", "bob", self.action_id, status="completed"),
                          lambda: self.repo.list_reminders("reminder-test", "bob")):
            with self.assertRaises(StorageError) as caught:
                operation()
            self.assertEqual(caught.exception.status_code, 404)


class ReminderPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_does_not_resend_and_completion_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            url = "sqlite:///" + (Path(temp) / "reminders.db").as_posix()
            repo = repository(url)
            record = save_actions(repo)
            action_id = record["result"]["actions"]["action_items"][0]["action_id"]
            sender = FakeFeishu()
            await ReminderService(repo, sender).run_once(now=local_day(7))
            repo.close()
            repo = repository(url)
            try:
                await ReminderService(repo, sender).run_once(now=local_day(7, 10))
                self.assertEqual(len(sender.messages), 1)
                repo.update_action("reminder-test", "alice", action_id, status="completed")
                self.assertEqual(await ReminderService(repo, sender).run_once(now=local_day(9)), 0)
            finally:
                repo.close()

    async def test_actions_api_updates_persist_and_rejects_bad_dates(self):
        app = create_app(repository=repository(), pipeline=pipeline(), vectors=DisabledVectors())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                record = (await client.post("/api/v1/meeting/api-reminders/demo?user_id=alice")).json()
                action_id = record["actions"]["action_items"][0]["action_id"]
                url = f"/api/v1/meeting/api-reminders/actions/{action_id}"
                response = await client.patch(url, params={"user_id": "alice"}, json={"deadline": "2099-05-10"})
                self.assertEqual(response.status_code, 200)
                reminders = (await client.get("/api/v1/meeting/api-reminders/reminders?user_id=alice")).json()
                self.assertEqual(len(reminders["reminders"]), 2)
                bad = await client.patch(url, params={"user_id": "alice"}, json={"deadline": "2099-02-30"})
                self.assertEqual(bad.status_code, 422)
                bad = await client.patch(url, params={"user_id": "alice"}, json={"status": "done"})
                self.assertEqual(bad.status_code, 422)
                await client.patch(url, params={"user_id": "alice"}, json={"status": "completed"})
                current = (await client.get("/api/v1/meeting/api-reminders/actions?user_id=alice")).json()
                self.assertEqual(current["action_items"][0]["status"], "completed")
                denied = await client.patch(url, params={"user_id": "bob"}, json={"status": "pending"})
                self.assertEqual(denied.status_code, 404)

    async def test_lifespan_starts_and_stops_scanner(self):
        app = create_app(repository=repository(), pipeline=pipeline(), vectors=DisabledVectors(), reminders_enabled=True)
        with patch.object(ReminderService, "run_once", return_value=0) as scan:
            async with app.router.lifespan_context(app):
                await asyncio.sleep(0.01)
                self.assertGreaterEqual(scan.await_count, 1)
                task = app.state.reminders._task
            self.assertTrue(task.done())

    async def test_webhook_payload_is_sent_once_and_response_checked(self):
        sender = FeishuClient(webhook_url="https://example.test/hook")
        await sender._client.aclose()
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(200, json={"code": 0})
        sender._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            result = await sender.send_action_reminder("Reminder")
            self.assertTrue(result["success"])
            self.assertEqual(len(calls), 1)
            self.assertIn(b'Reminder', calls[0].content)
        finally:
            await sender.close()
