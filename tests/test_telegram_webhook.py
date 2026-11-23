import asyncio
from types import SimpleNamespace

import httpx
import pytest

import app.main as main
from app.signal_journal import SignalJournal
from app.ml_shadow import ShadowML
from app.state_store import SignalStore


class SyncASGIClient:
    """Small synchronous wrapper around HTTPX ASGITransport without TestClient warnings."""

    def __init__(self, app):
        self.transport = httpx.ASGITransport(app=app)

    def request(self, method, path, **kwargs):
        async def send():
            async with httpx.AsyncClient(
                transport=self.transport, base_url="http://test"
            ) as client:
                return await client.request(method, path, **kwargs)
        return asyncio.run(send())

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)


class FakeNotifier:
    def __init__(self):
        self.acknowledgements = []
        self.cleared_buttons = []
        self.messages = []

    async def answer_callback_query(self, callback_id, message):
        self.acknowledgements.append((callback_id, message))
        return True

    async def clear_signal_review_buttons(self, chat_id, message_id):
        self.cleared_buttons.append((chat_id, message_id))
        return True

    async def send_text(self, message):
        self.messages.append(message)
        return True


@pytest.fixture
def webhook_env(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "webhook-secret")
    monkeypatch.setattr(main.settings, "telegram_chat_id", "-100123")
    monkeypatch.setattr(main.settings, "telegram_allowed_user_ids", "42, 84")
    monkeypatch.setattr(main.settings, "journal_admin_key", "admin-secret")
    journal = SignalJournal(str(tmp_path / "feedback.sqlite3"))
    candle_time = "2026-11-21T09:15:00+00:00"
    journal.record_evaluation("MEX", "BTCUSD", candle_time, "setup", {}, {"side": "LONG"})
    evaluation_id = journal.evaluation_id("MEX", "BTCUSD", candle_time)
    notifier = FakeNotifier()
    store = SignalStore(str(tmp_path / "outbox.sqlite3"))
    previous_engine = main.engine
    main.engine = SimpleNamespace(journal=journal, notifier=notifier, store=store)
    try:
        yield {
            "client": SyncASGIClient(main.app),
            "journal": journal,
            "evaluation_id": evaluation_id,
            "notifier": notifier,
            "store": store,
        }
    finally:
        main.engine = previous_engine


def callback_update(token, user_id=42, label="good"):
    return {
        "callback_query": {
            "id": "callback-1",
            "from": {"id": user_id},
            "message": {"message_id": 17, "chat": {"id": -100123}},
            "data": f"r|{label}|{token}",
        }
    }


def post_callback(env, update):
    return env["client"].post(
        "/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "webhook-secret"},
        json=update,
    )


def test_inline_feedback_checks_user_and_persists_audit_metadata(webhook_env):
    env = webhook_env
    token = env["journal"].create_feedback_token(env["evaluation_id"])

    response = post_callback(env, callback_update(token, user_id=42, label="needs_review"))

    assert response.status_code == 200
    assert response.json()["evaluation_id"] == env["evaluation_id"]
    record = env["journal"].list_records(limit=1)[0]
    assert record["feedback"] == "needs_review"
    assert record["feedback_source"] == "telegram_inline"
    assert record["feedback_user_id"] == "42"
    assert record["reviewed_at"]
    assert env["notifier"].acknowledgements == [
        ("callback-1", "Review saved: Needs Review")
    ]
    assert env["notifier"].cleared_buttons == [(-100123, 17)]


def test_inline_feedback_rejects_unapproved_user_without_consuming_token(webhook_env):
    env = webhook_env
    token = env["journal"].create_feedback_token(env["evaluation_id"])

    response = post_callback(env, callback_update(token, user_id=999))

    assert response.status_code == 403
    assert env["journal"].list_records(limit=1)[0]["feedback"] is None
    assert env["journal"].save_feedback_from_token(token, "good") == env["evaluation_id"]


def test_inline_feedback_replay_is_acknowledged_but_never_overwrites(webhook_env):
    env = webhook_env
    token = env["journal"].create_feedback_token(env["evaluation_id"])

    first = post_callback(env, callback_update(token, label="good"))
    replay = post_callback(env, callback_update(token, label="bad"))

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json()["ignored"] is True
    record = env["journal"].list_records(limit=1)[0]
    assert record["feedback"] == "good"
    assert record["feedback_user_id"] == "42"
    assert len(env["notifier"].acknowledgements) == 2


def test_inline_feedback_database_failure_is_retryable_and_not_acknowledged_as_saved(
    webhook_env, monkeypatch
):
    env = webhook_env
    token = env["journal"].create_feedback_token(env["evaluation_id"])

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(env["journal"], "save_feedback_from_token", fail_save)
    response = post_callback(env, callback_update(token))

    assert response.status_code == 503
    assert env["notifier"].acknowledgements == [
        ("callback-1", "Temporary database error. Please try again.")
    ]


def test_inline_feedback_rejects_invalid_category_and_missing_message_identity(webhook_env):
    env = webhook_env
    token = env["journal"].create_feedback_token(env["evaluation_id"])

    invalid_label = post_callback(env, callback_update(token, label="buy"))
    malformed = callback_update(token)
    malformed["callback_query"]["message"].pop("message_id")
    missing_message_id = post_callback(env, malformed)

    assert invalid_label.status_code == 400
    assert missing_message_id.status_code == 400
    assert env["journal"].list_records(limit=1)[0]["feedback"] is None


def test_telegram_command_feedback_records_sender_and_source(webhook_env):
    env = webhook_env
    update = {
        "message": {
            "from": {"id": 84},
            "chat": {"id": -100123},
            "text": f"/good {env['evaluation_id']} manual review",
        }
    }

    response = env["client"].post(
        "/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "webhook-secret"},
        json=update,
    )

    assert response.status_code == 200
    record = env["journal"].list_records(limit=1)[0]
    assert record["feedback"] == "good"
    assert record["feedback_source"] == "telegram_command"
    assert record["feedback_user_id"] == "84"
    assert env["notifier"].messages


def test_admin_delivery_endpoint_reads_actual_outbox_and_sent_rows(webhook_env):
    env = webhook_env
    signal_key = "BTCUSD:LONG:2026-11-21T10:00:00+00:00"
    payload = {
        "symbol": "BTCUSD", "side": "LONG",
        "setup_time": "2026-11-21T10:00:00+00:00",
        "entry": 100.0, "stop_loss": 98.0, "take_profit_1": 103.0,
    }
    env["store"].enqueue_alert(signal_key, payload)
    lease = env["store"].claim_due_alerts()[0][2]
    assert env["store"].mark_sent(signal_key, payload, lease)

    response = env["client"].get(
        f"/admin/telegram-delivery/{signal_key}",
        headers={"X-Journal-Key": "admin-secret"},
    )

    assert response.status_code == 200
    delivery = response.json()["delivery"]
    assert delivery["status"] == "sent"
    assert delivery["sent_signal_recorded"] is True
    assert delivery["attempts"] == 1
    assert "payload" not in delivery


def test_inline_feedback_rejects_wrong_chat_and_invalid_webhook_secret(webhook_env):
    env = webhook_env
    token = env["journal"].create_feedback_token(env["evaluation_id"])
    wrong_chat = callback_update(token)
    wrong_chat["callback_query"]["message"]["chat"]["id"] = -100999

    rejected_chat = post_callback(env, wrong_chat)
    rejected_secret = env["client"].post(
        "/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
        json=callback_update(token),
    )

    assert rejected_chat.status_code == 403
    assert rejected_secret.status_code == 401
    assert env["journal"].list_records(limit=1)[0]["feedback"] is None


def test_telegram_commands_reject_non_allowlisted_sender(webhook_env):
    env = webhook_env
    response = env["client"].post(
        "/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "webhook-secret"},
        json={"message": {"from": {"id": 999}, "chat": {"id": -100123}, "text": "/learning"}},
    )
    assert response.status_code == 403
    assert env["notifier"].messages == []


def test_market_outcome_admin_endpoint_reads_normalized_database_rows(webhook_env):
    env = webhook_env
    env["journal"].update_lifecycle(
        "MEX", "BTCUSD", "2026-11-21T09:15:00+00:00",
        "trigger_confirmed", 100.0,
        event_at="2026-11-21T09:17:00+00:00",
        side="LONG", stop_loss=98.0, target_price=103.0, target_price_2=104.0,
        atr=1.2, horizon_bars=12, fee_pct=0.08, slippage_pct=0.04,
    )
    env["journal"].update_lifecycle(
        "MEX", "BTCUSD", "2026-11-21T09:15:00+00:00",
        "telegram_sent", event_at="2026-11-21T09:17:05+00:00",
    )

    response = env["client"].get(
        "/admin/market-outcomes?status=pending",
        headers={"X-Journal-Key": "admin-secret"},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 1
    row = response.json()["outcomes"][0]
    assert row["evaluation_id"] == env["evaluation_id"]
    assert row["status"] == "pending"
    assert row["tp1_price"] == 103.0
    assert row["tp2_price"] == 104.0
    assert row["fee_pct"] == 0.08
    assert row["slippage_pct"] == 0.04
    assert response.json()["execution"] == "disabled"


def test_shadow_ml_admin_status_and_controlled_training(webhook_env):
    env = webhook_env
    main.engine.shadow_ml = ShadowML(env["journal"])
    status = env["client"].get(
        "/admin/ml/status",
        headers={"X-Journal-Key": "admin-secret"},
    )
    assert status.status_code == 200
    assert status.json()["mode"] == "shadow_only"
    assert status.json()["live_signal_decisions_modified"] is False
    assert status.json()["automatic_promotion"] is False
    assert status.json()["promotion_requires_explicit_approval"] is True
    operational = env["client"].get("/status")
    assert operational.status_code == 200
    assert operational.json()["shadow_ml"]["mode"] == "shadow_only"
    assert operational.json()["signal_journal"] == env["journal"].counts()
    assert operational.json()["market_outcomes"] == env["journal"].market_outcome_counts()

    invalid = env["client"].post(
        "/admin/ml/train?strategy=TPR",
        headers={"X-Journal-Key": "admin-secret"},
    )
    assert invalid.status_code == 400

    trained = env["client"].post(
        "/admin/ml/train?strategy=ORB",
        headers={"X-Journal-Key": "admin-secret"},
    )
    assert trained.status_code == 200
    body = trained.json()
    assert body["mode"] == "shadow_only"
    assert body["automatic_promotion"] is False
    assert body["promotion_requires_explicit_approval"] is True
    assert body["live_signal_decisions_modified"] is False
    assert body["results"][0]["strategy"] == "ORB"
    assert body["results"][0]["status"] == "insufficient_data"


def test_monitoring_routes_separate_liveness_readiness_and_status(webhook_env, monkeypatch):
    env = webhook_env
    monkeypatch.setattr(main, "engine", None)
    monkeypatch.setattr(main, "engine_task", None)
    monkeypatch.setattr(main.settings, "telegram_bot_token", "TELEGRAM_SECRET_SENTINEL", raising=False)
    monkeypatch.setattr(main.settings, "journal_admin_key", "JOURNAL_SECRET_SENTINEL", raising=False)

    ping = env["client"].get("/ping")
    health = env["client"].get("/health")
    ready = env["client"].get("/ready")
    status = env["client"].get("/status")

    assert ping.status_code == 200
    assert ping.content == b""
    assert health.status_code == 200
    assert health.content == b""
    assert ready.status_code == 503
    assert status.status_code == 200
    assert status.json()["status"] == "starting"
    response_text = health.text + ready.text + status.text
    assert "TELEGRAM_SECRET_SENTINEL" not in response_text
    assert "JOURNAL_SECRET_SENTINEL" not in response_text


def test_health_and_status_redact_sensitive_engine_diagnostics(webhook_env, monkeypatch):
    env = webhook_env

    class DiagnosticEngine:
        active_symbols = []

        @staticmethod
        def health():
            return {
                "api_key": "ENGINE_API_SECRET",
                "database_url": "postgresql://user:ENGINE_DB_SECRET@db.example.com/bot",
                "last_ws_disconnect_reason": "password=ENGINE_PASSWORD_SECRET",
                "state_store_ok": True,
            }

    monkeypatch.setattr(main, "engine", DiagnosticEngine())
    monkeypatch.setattr(main, "engine_task", None)

    health = env["client"].get("/health")
    status = env["client"].get("/status")
    ready = env["client"].get("/ready")

    assert health.status_code == 200
    assert health.content == b""
    assert status.status_code == 200
    assert ready.status_code == 503
    combined = health.text + status.text + ready.text
    for secret in ("ENGINE_API_SECRET", "ENGINE_DB_SECRET", "ENGINE_PASSWORD_SECRET"):
        assert secret not in combined
