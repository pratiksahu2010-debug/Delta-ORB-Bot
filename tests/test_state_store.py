from app.state_store import SignalStore


def sample_payload():
    return {
        "symbol": "BTCUSD", "side": "LONG",
        "setup_time": "2026-11-15T10:00:00+00:00",
        "entry": 100.0, "stop_loss": 98.0, "take_profit_1": 103.0,
    }


def test_sent_signal_survives_store_reopen(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    store = SignalStore(str(db_path))
    key = "BTCUSD:LONG:2026-11-15T10:00:00+00:00"
    payload = sample_payload()
    assert not store.is_sent(key)
    store.enqueue_alert(key, payload)
    claim = store.claim_due_alerts()
    assert store.mark_sent(key, payload, claim[0][2]) is True
    reopened = SignalStore(str(db_path))
    assert reopened.is_sent(key)


def test_outbox_intent_survives_reopen_and_is_idempotent(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    key = "BTCUSD:LONG:2026-11-15T10:00:00+00:00"
    store = SignalStore(str(db_path))
    payload = sample_payload()
    store.enqueue_alert(key, payload)
    store.enqueue_alert(key, {**payload, "entry": 999.0})

    reopened = SignalStore(str(db_path))
    due = reopened.due_alerts()
    assert due == [(key, payload)]
    assert {key: reopened.outbox_health()[key] for key in ("pending", "sending", "sent", "retrying")} == {"pending": 1, "sending": 0, "sent": 0, "retrying": 0}


def test_outbox_retry_schedule_survives_reopen(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    key = "BTCUSD:LONG:2026-11-15T10:00:00+00:00"
    store = SignalStore(str(db_path))
    store.enqueue_alert(key, sample_payload())
    lease = store.claim_due_alerts()[0][2]
    assert store.schedule_retry(key, lease, 60, "temporary Telegram failure")

    reopened = SignalStore(str(db_path))
    assert reopened.due_alerts() == []
    assert {key: reopened.outbox_health()[key] for key in ("pending", "sending", "sent", "retrying")} == {"pending": 1, "sending": 0, "sent": 0, "retrying": 1}


def test_mark_sent_completes_outbox_and_dedupe_atomically(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    key = "BTCUSD:LONG:2026-11-15T10:00:00+00:00"
    payload = sample_payload()
    store = SignalStore(str(db_path))
    store.enqueue_alert(key, payload)
    claim = store.claim_due_alerts()
    assert store.mark_sent(key, payload, claim[0][2]) is True

    reopened = SignalStore(str(db_path))
    assert reopened.is_sent(key)
    assert reopened.due_alerts() == []
    assert {key: reopened.outbox_health()[key] for key in ("pending", "sending", "sent", "retrying")} == {"pending": 0, "sending": 0, "sent": 1, "retrying": 0}


def test_outbox_claim_is_exclusive_and_expired_claim_recovers(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    key = "BTCUSD:LONG:2026-11-15T10:00:00+00:00"
    payload = sample_payload()
    first = SignalStore(str(db_path))
    second = SignalStore(str(db_path))
    first.enqueue_alert(key, payload)

    first_claim = first.claim_due_alerts()
    assert len(first_claim) == 1 and first_claim[0][:2] == (key, payload)
    first_lease = first_claim[0][2]
    assert {key: first.outbox_health()[key] for key in ("pending", "sending", "sent", "retrying")} == {"pending": 0, "sending": 1, "sent": 0, "retrying": 0}
    assert second.claim_due_alerts() == []

    # Simulate a worker crash by expiring the lease; the next worker recovers it.
    with first._connect() as db:
        db.execute(
            "UPDATE alert_outbox SET lease_until = ? WHERE signal_key = ?",
            ("2000-01-01T00:00:00+00:00", key),
        )
    second_claim = second.claim_due_alerts()
    assert len(second_claim) == 1 and second_claim[0][:2] == (key, payload)
    second_lease = second_claim[0][2]
    assert second_lease != first_lease
    # A stale worker cannot complete or reschedule a newer worker's lease.
    assert first.mark_sent(key, payload, first_lease) is False
    assert first.schedule_retry(key, first_lease, 10, "stale failure") is False
    assert second.mark_sent(key, payload, second_lease) is True


def test_mark_sent_completes_a_claimed_outbox_row(tmp_path):
    store = SignalStore(str(tmp_path / "state.sqlite3"))
    key = "BTCUSD:LONG:2026-11-15T10:00:00+00:00"
    payload = sample_payload()
    store.enqueue_alert(key, payload)
    claim = store.claim_due_alerts()
    assert len(claim) == 1 and claim[0][:2] == (key, payload)
    assert store.mark_sent(key, payload, claim[0][2]) is True
    assert store.due_alerts() == []
    assert store.is_sent(key)


def test_pending_setup_survives_store_reopen_and_can_be_deleted(tmp_path):
    from datetime import datetime, timedelta, timezone

    path = tmp_path / "state.sqlite3"
    store = SignalStore(str(path))
    expires = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    payload = {"side": "LONG", "candle_time": "2026-11-17T00:00:00+00:00", "expires_at": expires}
    store.save_pending_setup("BTCUSD", "MEX", payload)

    reopened = SignalStore(str(path))
    assert reopened.load_pending_setups() == [("BTCUSD", "MEX", payload)]
    assert reopened.pending_setup_count() == 1
    reopened.delete_pending_setup("BTCUSD", "MEX")
    assert SignalStore(str(path)).pending_setup_count() == 0


def test_pending_setup_store_prunes_expired_rows(tmp_path):
    from datetime import datetime, timedelta, timezone

    store = SignalStore(str(tmp_path / "state.sqlite3"))
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    store.save_pending_setup("BTCUSD", "ORB", {"expires_at": expired})
    assert store.load_pending_setups() == []
    assert store.pending_setup_count() == 0



def test_outbox_health_reports_backlog_retries_and_expired_leases(tmp_path):
    from datetime import datetime, timedelta, timezone

    store = SignalStore(str(tmp_path / "state.sqlite3"))
    now = datetime.now(timezone.utc)

    # A due pending alert is visible without being claimed by the health query.
    due_key = "ETHUSD:LONG:2026-11-15T10:00:00+00:00"
    store.enqueue_alert(due_key, {**sample_payload(), "symbol": "ETHUSD"})
    with store._connect() as db:
        db.execute(
            "UPDATE alert_outbox SET created_at = ?, next_attempt_at = ? WHERE signal_key = ?",
            ((now - timedelta(minutes=15)).isoformat(), (now + timedelta(hours=2)).isoformat(), due_key),
        )

    # A failed delivery is pending with a future retry time and an error marker.
    retry_key = "SOLUSD:LONG:2026-11-15T10:00:00+00:00"
    store.enqueue_alert(retry_key, {**sample_payload(), "symbol": "SOLUSD"})
    retry_lease = store.claim_due_alerts()[0][2]
    assert store.schedule_retry(retry_key, retry_lease, 3600, "private provider detail")

    # A worker that died while sending leaves an expired lease for recovery.
    sending_key = "XRPUSD:LONG:2026-11-15T10:00:00+00:00"
    store.enqueue_alert(sending_key, {**sample_payload(), "symbol": "XRPUSD"})
    sending_claim = store.claim_due_alerts()
    sending = next(item for item in sending_claim if item[0] == sending_key)
    with store._connect() as db:
        db.execute(
            "UPDATE alert_outbox SET lease_until = ?, updated_at = ? WHERE signal_key = ?",
            ((now - timedelta(seconds=30)).isoformat(), (now - timedelta(minutes=3)).isoformat(), sending_key),
        )

    with store._connect() as db:
        db.execute(
            "UPDATE alert_outbox SET next_attempt_at = ? WHERE signal_key = ?",
            ((now - timedelta(seconds=1)).isoformat(), due_key),
        )

    health = store.outbox_health()
    assert health["pending"] == 2
    assert health["sending"] == 1
    assert health["retrying"] == 1
    assert health["due_pending"] == 1
    assert health["scheduled_retries"] == 1
    assert health["expired_leases"] == 1
    assert health["recovery_required"] is True
    assert health["oldest_pending_age_seconds"] >= 14 * 60
    assert health["oldest_sending_age_seconds"] >= 2 * 60
    assert health["oldest_retry_age_seconds"] is not None
    assert health["max_retry_attempts"] >= 1
    assert health["max_active_attempts"] >= 1
    assert "private provider detail" not in str(health)


def test_delivery_record_reports_outbox_and_sent_signal_rows_without_payload(tmp_path):
    store = SignalStore(str(tmp_path / "delivery-audit.sqlite3"))
    key = "BTCUSD:LONG:2026-11-15T10:00:00+00:00"
    payload = sample_payload()
    store.enqueue_alert(key, payload)

    pending = store.delivery_record(key)
    assert pending["status"] == "pending"
    assert pending["sent_signal_recorded"] is False
    assert "payload" not in pending

    claim = store.claim_due_alerts()
    assert store.mark_sent(key, payload, claim[0][2]) is True
    sent = SignalStore(str(tmp_path / "delivery-audit.sqlite3")).delivery_record(key)
    assert sent["status"] == "sent"
    assert sent["sent_signal_recorded"] is True
    assert sent["outbox_sent_at"] == sent["sent_signal_at"]
    assert sent["attempts"] == 1


def test_advika_position_state_survives_reopen_and_advances_idempotently(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = SignalStore(str(path))
    key = "BTCUSD:ADVIKA:LONG:2026-11-24T08:00:00+00:00"
    position = {
        "symbol": "BTCUSD", "side": "LONG", "entry_price": 100.0,
        "stop_loss": 98.0, "take_profit_1": 103.0,
        "trailing_stop": 98.0, "partial_taken": False,
        "setup_time": "2026-11-24T08:00:00+00:00",
    }
    assert store.create_advika_position(key, position) is True
    assert store.create_advika_position(key, {**position, "entry_price": 999}) is False

    reopened = SignalStore(str(path))
    assert reopened.list_open_advika_positions("BTCUSD") == []
    assert reopened.advika_position_health()["pending"] == 1
    reopened.enqueue_alert(key, sample_payload())
    claim = reopened.claim_due_alerts()[0]
    assert reopened.mark_sent(key, sample_payload(), claim[2]) is True
    assert reopened.activate_delivered_advika_positions() == 1
    loaded = reopened.list_open_advika_positions("BTCUSD")
    assert len(loaded) == 1
    assert loaded[0]["entry_price"] == 100.0
    assert loaded[0]["status"] == "open"
    assert loaded[0]["last_processed_candle"] == position["setup_time"]

    updated = {**loaded[0], "partial_taken": True, "trailing_stop": 101.0}
    assert reopened.update_advika_position(
        key, updated, "partial", "2026-11-24T08:05:00+00:00",
    ) is True
    health = reopened.advika_position_health()
    assert health == {"pending": 0, "open": 0, "partial": 1, "closed": 0, "active": 1}

    active = reopened.list_open_advika_positions("BTCUSD")
    assert len(active) == 1
    assert active[0]["partial_taken"] is True
    assert active[0]["trailing_stop"] == 101.0
    assert active[0]["last_processed_candle"] == "2026-11-24T08:05:00+00:00"

    assert reopened.update_advika_position(
        key, updated, "closed", "2026-11-24T08:10:00+00:00",
    ) is True
    assert reopened.list_open_advika_positions("BTCUSD") == []
    assert reopened.advika_position_health()["closed"] == 1


def test_advika_entry_outbox_and_position_are_persisted_atomically(tmp_path):
    store = SignalStore(str(tmp_path / "state.sqlite3"))
    key = "BTCUSD:ADVIKA:LONG:2026-11-24T08:00:00+00:00"
    alert = {
        "symbol": "BTCUSD", "strategy": "ADVIKA", "side": "LONG",
        "entry": 100.0, "stop_loss": 98.0, "take_profit_1": 103.0,
        "setup_time": "2026-11-24T08:00:00+00:00",
    }
    position = {
        "symbol": "BTCUSD", "side": "LONG", "entry_price": 100.0,
        "stop_loss": 98.0, "take_profit_1": 103.0, "trailing_stop": 98.0,
        "partial_taken": False, "setup_time": alert["setup_time"],
    }
    assert store.enqueue_advika_signal(key, alert, position) == (True, True)
    assert store.enqueue_advika_signal(key, alert, position) == (False, False)
    assert store.due_alerts() == [(key, alert)]
    assert store.list_open_advika_positions("BTCUSD") == []
    claim = store.claim_due_alerts()[0]
    assert store.mark_sent(key, alert, claim[2]) is True
    assert store.activate_delivered_advika_positions() == 1
    active = store.list_open_advika_positions("BTCUSD")
    assert len(active) == 1
    assert active[0]["entry_price"] == 100.0
