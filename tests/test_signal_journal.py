import sqlite3

from app.signal_journal import SignalJournal
from app.state_store import SignalStore


def test_journal_records_setup_and_rejection_once(tmp_path):
    journal = SignalJournal(str(tmp_path / "journal.sqlite3"))
    stamp = "2026-11-18T09:20:00+00:00"
    journal.record_evaluation(
        "MEX", "BTCUSD", stamp, "setup",
        {"stage": "entry", "values": {"rvol": 1.8}},
        {"side": "LONG", "entry_reference": 100.0, "stop_loss": 98.0,
         "take_profit_1": 103.0},
    )
    # Same candle is idempotent even if a live evaluator calls again.
    journal.record_evaluation("MEX", "BTCUSD", stamp, "rejected", {"reason": "duplicate"})
    journal.record_evaluation(
        "RB", "ETHUSD", stamp, "rejected",
        {"stage": "volume", "reason": "rvol_below_min", "values": {"rvol": 1.1}},
    )

    rows = journal.list_records(limit=10)
    assert len(rows) == 2
    assert journal.counts() == {
        "records": 2, "reviewed": 0, "rejected_evaluations": 1, "setup_evaluations": 1
    }
    mex = journal.list_records(limit=10, strategy="MEX")[0]
    assert mex["diagnostics"]["values"]["rvol"] == 1.8
    assert mex["signal"]["stop_loss"] == 98.0


def test_journal_lifecycle_feedback_filters_and_csv(tmp_path):
    journal = SignalJournal(str(tmp_path / "journal.sqlite3"))
    stamp = "2026-11-18T09:25:00+00:00"
    journal.record_evaluation("TPR", "SOLUSD", stamp, "setup", {}, {"side": "SHORT"})
    key = journal.evaluation_id("TPR", "SOLUSD", stamp)

    journal.update_lifecycle("TPR", "SOLUSD", stamp, "trigger_confirmed", 99.5)
    assert journal.add_feedback(key, "late", "Entry was extended.",
                                {"mfe_atr": 0.7, "mae_atr": 0.2})
    row = journal.list_records(limit=5, feedback="late")[0]
    assert row["lifecycle"] == "trigger_confirmed"
    assert row["entry_price"] == 99.5
    assert row["feedback_note"] == "Entry was extended."
    assert row["outcome"]["mfe_atr"] == 0.7
    assert "evaluation_id" in journal.export_csv(feedback="late")
    assert not journal.add_feedback("missing-id", "good")



def test_journal_persists_entry_timing_and_price_distance_metrics(tmp_path):
    journal = SignalJournal(str(tmp_path / "journal.sqlite3"))
    candle_time = "2026-11-19T09:20:00+00:00"
    journal.record_evaluation(
        "MEX", "BTCUSD", candle_time, "setup", {},
        {"side": "LONG", "entry_reference": 100.0, "stop_loss": 98.0,
         "take_profit_1": 103.0},
    )
    journal.update_lifecycle(
        "MEX", "BTCUSD", candle_time, "setup_armed",
        event_at="2026-11-19T09:20:02+00:00",
    )
    journal.update_lifecycle(
        "MEX", "BTCUSD", candle_time, "trigger_confirmed", 101.2,
        event_at="2026-11-19T09:20:05+00:00",
        trigger_price=101.0,
        setup_to_trigger_seconds=3.0,
        entry_extension_atr=0.6,
        trigger_overshoot_atr=0.1,
    )
    journal.update_lifecycle(
        "MEX", "BTCUSD", candle_time, "telegram_sent",
        event_at="2026-11-19T09:20:06+00:00",
        trigger_to_telegram_seconds=1.0,
    )

    row = journal.list_records(limit=1)[0]
    assert row["setup_armed_at"] == "2026-11-19T09:20:02+00:00"
    assert row["trigger_confirmed_at"] == "2026-11-19T09:20:05+00:00"
    assert row["telegram_sent_at"] == "2026-11-19T09:20:06+00:00"
    assert row["trigger_price"] == 101.0
    assert row["setup_to_trigger_seconds"] == 3.0
    assert row["entry_extension_atr"] == 0.6
    assert row["trigger_overshoot_atr"] == 0.1
    assert row["trigger_to_telegram_seconds"] == 1.0
    csv_text = journal.export_csv()
    assert "setup_to_trigger_seconds" in csv_text
    assert "trigger_to_telegram_seconds" in csv_text



def test_journal_additively_migrates_existing_sqlite_schema(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE signal_journal (
            evaluation_id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            candle_time TEXT NOT NULL,
            evaluated_at TEXT NOT NULL,
            decision TEXT NOT NULL,
            side TEXT,
            reason TEXT,
            diagnostics_json TEXT NOT NULL,
            signal_json TEXT NOT NULL,
            lifecycle TEXT NOT NULL DEFAULT 'evaluated',
            entry_price REAL,
            stop_loss REAL,
            target_price REAL,
            feedback TEXT,
            feedback_note TEXT,
            reviewed_at TEXT,
            outcome_json TEXT NOT NULL DEFAULT '{}'
        )""")
        db.execute(
            """INSERT INTO signal_journal
            (evaluation_id,strategy,symbol,candle_time,evaluated_at,decision,
             diagnostics_json,signal_json,lifecycle)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            ("TPR:BTCUSD:legacy", "TPR", "BTCUSD", "legacy", "legacy",
             "setup", "{}", "{}", "setup_armed"),
        )

    journal = SignalJournal(str(path))
    row = journal.list_records(limit=1)[0]
    assert row["evaluation_id"] == "TPR:BTCUSD:legacy"
    assert row["setup_armed_at"] is None
    assert row["trigger_confirmed_at"] is None
    assert row["trigger_to_telegram_seconds"] is None


def test_confirmed_signal_snapshot_is_immutable_and_lifecycle_never_regresses(tmp_path):
    journal = SignalJournal(str(tmp_path / "immutable.sqlite3"))
    candle_time = "2026-11-20T10:00:00+00:00"
    original_signal = {
        "side": "LONG", "entry_reference": 100.0,
        "stop_loss": 98.0, "take_profit_1": 103.0,
    }
    journal.record_evaluation(
        "ORB", "BTCUSD", candle_time, "setup",
        {"reason": "original_setup", "values": {"rvol": 1.8}},
        original_signal,
    )
    journal.update_lifecycle(
        "ORB", "BTCUSD", candle_time, "trigger_confirmed", 101.0,
        event_at="2026-11-20T10:00:05+00:00",
        trigger_price=100.8,
    )
    journal.update_lifecycle(
        "ORB", "BTCUSD", candle_time, "telegram_sent",
        event_at="2026-11-20T10:00:06+00:00",
    )

    # A repeated evaluation for the same candle must not rewrite a confirmed alert.
    journal.record_evaluation(
        "ORB", "BTCUSD", candle_time, "setup",
        {"reason": "changed_setup", "values": {"rvol": 9.9}},
        {"side": "SHORT", "entry_reference": 111.0, "stop_loss": 112.0,
         "take_profit_1": 108.0},
    )
    # Late/out-of-order lifecycle events must not move the record backwards.
    journal.update_lifecycle(
        "ORB", "BTCUSD", candle_time, "trigger_confirmed", 999.0,
        event_at="2026-11-20T10:00:04+00:00",
        trigger_price=999.0,
    )

    row = journal.list_records(limit=1)[0]
    assert row["lifecycle"] == "telegram_sent"
    assert row["signal"] == original_signal
    assert row["side"] == "LONG"
    assert row["reason"] == "original_setup"
    assert row["diagnostics"]["values"]["rvol"] == 1.8
    assert row["entry_price"] == 101.0
    assert row["trigger_price"] == 100.8
    assert row["trigger_confirmed_at"] == "2026-11-20T10:00:05+00:00"
    assert row["telegram_sent_at"] == "2026-11-20T10:00:06+00:00"


def test_signal_journal_rejects_unknown_lifecycle(tmp_path):
    journal = SignalJournal(str(tmp_path / "invalid-lifecycle.sqlite3"))
    try:
        journal.update_lifecycle("ORB", "BTCUSD", "stamp", "unknown_state")
    except ValueError as exc:
        assert "Unknown signal lifecycle" in str(exc)
    else:
        raise AssertionError("unknown lifecycle must be rejected")


def test_market_outcome_persistence_is_separate_and_final_result_is_immutable(tmp_path):
    journal = SignalJournal(str(tmp_path / "market-outcomes.sqlite3"))
    stamp = "2026-11-20T10:00:00+00:00"
    journal.record_evaluation(
        "ORB", "BTCUSD", stamp, "setup", {},
        {"side": "LONG", "entry_reference": 100.0, "stop_loss": 98.0,
         "take_profit_1": 103.0},
    )
    journal.update_lifecycle("ORB", "BTCUSD", stamp, "trigger_confirmed", 100.0,
                              event_at="2026-11-20T10:02:00+00:00")
    evaluation_id = journal.evaluation_id("ORB", "BTCUSD", stamp)

    pending = journal.pending_market_outcomes()
    assert len(pending) == 1
    assert pending[0]["evaluation_id"] == evaluation_id
    assert journal.save_market_outcome(evaluation_id, "pending", {"result": "pending", "mfe_r": 0.4})
    assert journal.save_market_outcome(evaluation_id, "complete", {"result": "win", "mfe_r": 1.5})
    assert not journal.save_market_outcome(evaluation_id, "complete", {"result": "loss", "mfe_r": 0.0})

    row = journal.list_records(limit=1)[0]
    assert row["market_outcome_status"] == "complete"
    assert row["market_outcome"] == {"result": "win", "mfe_r": 1.5, "final_result": "win"}
    assert row["outcome"] == {}
    assert journal.pending_market_outcomes() == []
    assert "market_outcome_status" in journal.export_csv()


def test_feedback_tokens_are_single_use_and_first_review_is_immutable(tmp_path):
    journal = SignalJournal(str(tmp_path / "single-use-feedback.sqlite3"))
    stamp = "2026-11-20T12:00:00+00:00"
    journal.record_evaluation("MEX", "BTCUSD", stamp, "setup", {}, {"side": "LONG"})
    evaluation_id = journal.evaluation_id("MEX", "BTCUSD", stamp)
    first_token = journal.create_feedback_token(evaluation_id)
    second_token = journal.create_feedback_token(evaluation_id)

    assert journal.save_feedback_from_token(first_token, "good", "first review") == evaluation_id
    assert journal.save_feedback_from_token(first_token, "bad", "replay") is None
    assert journal.save_feedback_from_token(second_token, "bad", "second review") is None

    row = journal.list_records(limit=1)[0]
    assert row["feedback"] == "good"
    assert row["feedback_note"] == "first review"
    assert journal.add_feedback(evaluation_id, "late", "must not overwrite") is False

    assert journal.undo_feedback(evaluation_id) is True
    assert journal.add_feedback(evaluation_id, "late", "explicitly reviewed again") is True
    assert journal.list_records(limit=1)[0]["feedback"] == "late"


def test_feedback_token_for_missing_evaluation_is_consumed_without_creating_feedback(tmp_path):
    journal = SignalJournal(str(tmp_path / "missing-feedback.sqlite3"))
    token = journal.create_feedback_token("missing:evaluation:id")
    assert journal.save_feedback_from_token(token, "good") is None
    assert journal.save_feedback_from_token(token, "bad") is None


def test_feedback_source_and_reviewer_are_persisted_and_exported(tmp_path):
    journal = SignalJournal(str(tmp_path / "feedback-audit.sqlite3"))
    stamp = "2026-11-21T10:00:00+00:00"
    journal.record_evaluation("ORB", "ETHUSD", stamp, "setup", {}, {"side": "SHORT"})
    evaluation_id = journal.evaluation_id("ORB", "ETHUSD", stamp)

    assert journal.add_feedback(
        evaluation_id, "late", "clicked after move",
        source="telegram_command", user_id="987654",
    )
    record = journal.list_records(limit=1)[0]
    assert record["feedback"] == "late"
    assert record["feedback_source"] == "telegram_command"
    assert record["feedback_user_id"] == "987654"
    assert record["reviewed_at"]
    csv_text = journal.export_csv()
    assert "feedback_source" in csv_text
    assert "telegram_command" in csv_text

    assert journal.undo_feedback(evaluation_id)
    cleared = journal.list_records(limit=1)[0]
    assert cleared["feedback_source"] is None
    assert cleared["feedback_user_id"] is None


def test_feedback_token_persists_inline_source_and_user_id(tmp_path):
    journal = SignalJournal(str(tmp_path / "feedback-token-audit.sqlite3"))
    stamp = "2026-11-21T10:05:00+00:00"
    journal.record_evaluation("MEX", "SOLUSD", stamp, "setup", {}, {"side": "LONG"})
    evaluation_id = journal.evaluation_id("MEX", "SOLUSD", stamp)
    token = journal.create_feedback_token(evaluation_id)

    assert journal.save_feedback_from_token(
        token, "good", "great move", source="telegram_inline", user_id="42"
    ) == evaluation_id
    record = journal.list_records(limit=1)[0]
    assert record["feedback_source"] == "telegram_inline"
    assert record["feedback_user_id"] == "42"
    assert journal.save_feedback_from_token(token, "bad", "replay") is None


def test_feedback_categories_are_validated_at_storage_boundary(tmp_path):
    import pytest

    journal = SignalJournal(str(tmp_path / "feedback-category.sqlite3"))
    stamp = "2026-11-21T10:10:00+00:00"
    journal.record_evaluation("ORB", "BTCUSD", stamp, "setup", {}, {"side": "LONG"})
    evaluation_id = journal.evaluation_id("ORB", "BTCUSD", stamp)
    with pytest.raises(ValueError, match="Unknown feedback label"):
        journal.add_feedback(evaluation_id, "buy")
    token = journal.create_feedback_token(evaluation_id)
    with pytest.raises(ValueError, match="Unknown feedback label"):
        journal.save_feedback_from_token(token, "buy")


def test_confirmed_signal_outcome_context_recovers_after_restart_and_is_immutable(tmp_path):
    path = tmp_path / "outcome-recovery.sqlite3"
    journal = SignalJournal(str(path))
    store = SignalStore(str(path))
    candle_time = "2026-11-21T09:15:00+00:00"
    original_signal = {
        "side": "LONG", "entry_reference": 100.0, "stop_loss": 98.0,
        "take_profit_1": 103.0, "atr": 1.5, "reason": "original detector snapshot",
    }
    journal.record_evaluation("ORB", "BTCUSD", candle_time, "setup", {}, original_signal)
    evaluation_id = journal.evaluation_id("ORB", "BTCUSD", candle_time)
    signal_key = f"BTCUSD:ORB:LONG:{candle_time}"
    alert = {
        "symbol": "BTCUSD", "strategy": "ORB", "side": "LONG",
        "entry": 100.5, "stop_loss": 98.5,
        "take_profit_1": 103.5, "take_profit_2": 104.5,
        "atr": 1.5, "setup_time": candle_time,
        "trigger_confirmed_at": "2026-11-21T09:17:00+00:00",
        "outcome_horizon_bars": 12, "outcome_fee_pct": 0.08,
        "outcome_slippage_pct": 0.04, "outcome_tp2_r_multiple": 2.0,
    }
    store.enqueue_alert(signal_key, alert)
    lease = store.claim_due_alerts()[0][2]
    assert store.mark_sent(signal_key, alert, lease)

    # Simulate a crash after the outbox commit but before the journal lifecycle update.
    restarted_journal = SignalJournal(str(path))
    pending = restarted_journal.pending_market_outcomes()

    assert len(pending) == 1
    assert pending[0]["evaluation_id"] == evaluation_id
    assert pending[0]["tp1_price"] == 103.5
    assert pending[0]["tp2_price"] == 104.5
    assert pending[0]["atr"] == 1.5
    assert pending[0]["horizon_bars"] == 12
    assert pending[0]["fee_pct"] == 0.08
    assert pending[0]["slippage_pct"] == 0.04
    record = restarted_journal.list_records(limit=1)[0]
    assert record["lifecycle"] == "telegram_sent"
    assert record["signal"] == original_signal
    assert record["stop_loss"] == 98.0
    assert record["target_price"] == 103.0

    outcome = {
        "first_hit": "TP1", "first_hit_at": "2026-11-21T09:20:00+00:00",
        "final_result": "win", "tp1_hit": True, "tp2_hit": True,
        "stop_hit": False, "same_candle_ambiguous": False,
        "bars_to_first_hit": 1, "bars_to_resolution": 2, "bars_observed": 2,
        "elapsed_seconds": 480.0, "mfe_r": 2.0, "mae_r": 0.2,
        "mfe_pct": 4.0, "mae_pct": 0.4, "mfe_atr": 2.7, "mae_atr": 0.27,
        "exit_price": 104.5, "gross_return_pct": 3.98,
        "cost_pct": 0.12, "net_return_pct": 3.86,
        "resolution_at": "2026-11-21T09:25:00+00:00",
    }
    assert restarted_journal.save_market_outcome(evaluation_id, "complete", outcome)
    assert not restarted_journal.save_market_outcome(
        evaluation_id, "complete", {**outcome, "final_result": "loss"}
    )
    assert restarted_journal.pending_market_outcomes() == []

    with sqlite3.connect(path) as db:
        stored = db.execute(
            "SELECT status,first_hit,final_result,tp1_hit,tp2_hit,stop_hit,"
            "same_candle_ambiguous,bars_to_resolution,net_return_pct "
            "FROM signal_outcomes WHERE evaluation_id=?",
            (evaluation_id,),
        ).fetchone()
    assert stored == ("complete", "TP1", "win", 1, 1, 0, 0, 2, 3.86)


def test_pending_outbox_recovers_confirmed_outcome_before_telegram_delivery(tmp_path):
    path = tmp_path / "pending-outbox-outcome.sqlite3"
    journal = SignalJournal(str(path))
    store = SignalStore(str(path))
    candle_time = "2026-11-21T11:00:00+00:00"
    journal.record_evaluation(
        "MEX", "ETHUSD", candle_time, "setup", {},
        {"side": "SHORT", "stop_loss": 102.0, "take_profit_1": 97.0, "atr": 1.0},
    )
    evaluation_id = journal.evaluation_id("MEX", "ETHUSD", candle_time)
    alert = {
        "symbol": "ETHUSD", "strategy": "MEX", "side": "SHORT",
        "entry": 100.0, "stop_loss": 102.0, "take_profit_1": 97.0,
        "take_profit_2": 96.0, "atr": 1.0, "setup_time": candle_time,
        "trigger_confirmed_at": "2026-11-21T11:02:00+00:00",
    }
    store.enqueue_alert(f"ETHUSD:MEX:SHORT:{candle_time}", alert)

    pending = SignalJournal(str(path)).pending_market_outcomes()

    assert len(pending) == 1
    assert pending[0]["evaluation_id"] == evaluation_id
    assert SignalJournal(str(path)).list_records(limit=1)[0]["lifecycle"] == "trigger_confirmed"
    assert SignalJournal(str(path)).list_market_outcomes(status="pending")[0]["tp2_price"] == 96.0
