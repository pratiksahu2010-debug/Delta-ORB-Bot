"""Real PostgreSQL integration coverage for durable signal persistence.

CI supplies an ephemeral PostgreSQL service. The TLS URI helper is bypassed
only for this local test database; production connection validation remains
covered by test_neon_database.py.
"""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from app.migrations import apply_migrations
from app.signal_journal import SignalJournal
from app.state_store import SignalStore


@pytest.fixture
def postgres_url(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    # GitHub Actions' local postgres service has no TLS listener. Do not weaken
    # production URI validation; use the local service URI directly in tests.
    monkeypatch.setattr("app.database.postgres_conninfo", lambda value: value)
    return url


def test_postgres_state_store_outbox_claim_recovery_and_pending_setup(postgres_url):
    store = SignalStore("./unused.sqlite3", postgres_url)
    second_worker = SignalStore("./unused.sqlite3", postgres_url)
    suffix = uuid.uuid4().hex
    key = f"pg-integration:{suffix}"
    payload = {
        "symbol": f"TEST{suffix[:8]}",
        "side": "LONG",
        "setup_time": datetime.now(timezone.utc).isoformat(),
        "entry": 100.0,
        "stop_loss": 98.0,
        "take_profit_1": 103.0,
    }

    store.enqueue_alert(key, payload)
    assert store.due_alerts() == [(key, payload)]
    first_claim = store.claim_due_alerts()
    assert len(first_claim) == 1 and first_claim[0][:2] == (key, payload)
    first_lease = first_claim[0][2]
    assert second_worker.claim_due_alerts() == []

    # Simulate a crashed worker whose outbox lease has expired.
    with psycopg.connect(postgres_url) as db:
        db.execute(
            "UPDATE alert_outbox SET lease_until = %s WHERE signal_key = %s",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), key),
        )
    second_claim = second_worker.claim_due_alerts()
    assert len(second_claim) == 1 and second_claim[0][:2] == (key, payload)
    second_lease = second_claim[0][2]
    assert second_lease != first_lease
    assert store.mark_sent(key, payload, first_lease) is False
    assert store.schedule_retry(key, first_lease, 5, "stale worker") is False
    assert second_worker.mark_sent(key, payload, second_lease) is True
    assert store.is_sent(key)
    delivery = store.delivery_record(key)
    assert delivery["status"] == "sent"
    assert delivery["sent_signal_recorded"] is True
    assert store.due_alerts() == []
    assert store.health() is True

    setup_payload = {
        "side": "SHORT",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    }
    store.save_pending_setup(payload["symbol"], "MEX", setup_payload)
    assert store.load_pending_setups() == [(payload["symbol"], "MEX", setup_payload)]
    candle_time = datetime.now(timezone.utc).isoformat()
    assert store.transition_setup(payload["symbol"], "MEX", candle_time, None, "ARMED", "pg_setup_armed")
    assert store.transition_setup(payload["symbol"], "MEX", candle_time, "ARMED", "CONFIRMING", "pg_quote_crossed")
    assert store.transition_setup(payload["symbol"], "MEX", candle_time, "CONFIRMING", "TRIGGER_CONFIRMED", "pg_trigger_held")
    assert store.setup_lifecycle(payload["symbol"], "MEX", candle_time)["state"] == "TRIGGER_CONFIRMED"
    assert [event["to_state"] for event in store.setup_lifecycle_events(payload["symbol"], "MEX", candle_time)] == [
        "ARMED", "CONFIRMING", "TRIGGER_CONFIRMED"
    ]
    store.delete_pending_setup(payload["symbol"], "MEX")
    with psycopg.connect(postgres_url) as db:
        db.execute("DELETE FROM setup_lifecycle_events WHERE symbol = %s", (payload["symbol"],))
        db.execute("DELETE FROM setup_lifecycle_current WHERE symbol = %s", (payload["symbol"],))
    assert not any(row[0] == payload["symbol"] for row in store.load_pending_setups())

    with psycopg.connect(postgres_url) as db:
        db.execute("DELETE FROM sent_signals WHERE signal_key = %s", (key,))
        db.execute("DELETE FROM alert_outbox WHERE signal_key = %s", (key,))


def test_postgres_signal_journal_migrations_feedback_and_training_notes(postgres_url):
    journal = SignalJournal("./unused.sqlite3", postgres_url)
    # Reinitialization verifies the additive schema migration is repeatable.
    journal = SignalJournal("./unused.sqlite3", postgres_url)
    suffix = uuid.uuid4().hex
    symbol = f"TEST{suffix[:8]}"
    candle_time = datetime.now(timezone.utc).isoformat()

    journal.record_evaluation(
        "MEX", symbol, candle_time, "setup",
        {"stage": "entry", "values": {"rvol": 1.7}},
        {"side": "LONG", "entry_reference": 100.0, "stop_loss": 98.0,
         "take_profit_1": 103.0},
    )
    journal.update_lifecycle(
        "MEX", symbol, candle_time, "trigger_confirmed", 100.5,
        event_at=datetime.now(timezone.utc).isoformat(),
        trigger_price=100.4, setup_to_trigger_seconds=2.0,
        entry_extension_atr=0.3, trigger_overshoot_atr=0.1,
    )
    evaluation_id = journal.evaluation_id("MEX", symbol, candle_time)
    assert journal.add_feedback(
        evaluation_id, "good", "PostgreSQL integration",
        source="telegram_inline", user_id="4242",
    )
    row = journal.list_records(limit=5, strategy="MEX")[0]
    assert row["evaluation_id"] == evaluation_id
    assert row["lifecycle"] == "trigger_confirmed"
    assert row["entry_extension_atr"] == 0.3
    assert row["feedback"] == "good"
    assert row["feedback_source"] == "telegram_inline"
    assert row["feedback_user_id"] == "4242"

    token = journal.create_feedback_token(evaluation_id)
    assert journal.resolve_feedback_token(token) == evaluation_id
    # An already-reviewed journal row cannot be overwritten through a token.
    assert journal.save_feedback_from_token(token, "bad", "must not overwrite") is None

    fresh_candle_time = datetime.now(timezone.utc).isoformat()
    journal.record_evaluation("ORB", symbol, fresh_candle_time, "setup", {}, {"side": "SHORT"})
    fresh_evaluation_id = journal.evaluation_id("ORB", symbol, fresh_candle_time)
    fresh_token = journal.create_feedback_token(fresh_evaluation_id)
    assert journal.save_feedback_from_token(
        fresh_token, "good", "single-use postgres review", user_id="84"
    ) == fresh_evaluation_id
    fresh_record = journal.list_records(limit=5, strategy="ORB")[0]
    assert fresh_record["feedback_source"] == "telegram_inline"
    assert fresh_record["feedback_user_id"] == "84"
    assert journal.save_feedback_from_token(fresh_token, "bad", "replay") is None

    race_candle_time = datetime.now(timezone.utc).isoformat()
    journal.record_evaluation("MEX", symbol, race_candle_time, "setup", {}, {"side": "LONG"})
    race_evaluation_id = journal.evaluation_id("MEX", symbol, race_candle_time)
    race_token = journal.create_feedback_token(race_evaluation_id)
    with ThreadPoolExecutor(max_workers=6) as pool:
        race_results = list(pool.map(
            lambda index: journal.save_feedback_from_token(
                race_token, "good" if index == 0 else "bad",
                f"concurrent review {index}", user_id=str(index),
            ),
            range(6),
        ))
    assert race_results.count(race_evaluation_id) == 1
    assert sum(result is not None for result in race_results) == 1
    race_record = next(
        row for row in journal.list_records(limit=20, strategy="MEX")
        if row["evaluation_id"] == race_evaluation_id
    )
    assert race_record["feedback"] in {"good", "bad"}
    assert race_record["feedback_source"] == "telegram_inline"
    assert race_record["feedback_user_id"] is not None

    note_id = journal.add_training_note(f"postgres integration {suffix}")
    assert note_id > 0
    assert any(f"postgres integration {suffix}" == row["note"]
               for row in journal.list_training_notes(limit=50))

    with psycopg.connect(postgres_url) as db:
        db.execute("DELETE FROM signal_feedback_tokens WHERE evaluation_id IN (%s, %s, %s)",
                   (evaluation_id, fresh_evaluation_id, race_evaluation_id))
        db.execute("DELETE FROM signal_journal WHERE evaluation_id IN (%s, %s, %s)",
                   (evaluation_id, fresh_evaluation_id, race_evaluation_id))
        db.execute("DELETE FROM training_notes WHERE note = %s", (f"postgres integration {suffix}",))


def test_postgres_schema_migrations_are_versioned_and_idempotent(postgres_url):
    store = SignalStore("./unused.sqlite3", postgres_url)
    journal = SignalJournal("./unused.sqlite3", postgres_url)
    # Reopening both stores must not reapply already-recorded migrations.
    SignalStore("./unused.sqlite3", postgres_url)
    SignalJournal("./unused.sqlite3", postgres_url)

    with psycopg.connect(postgres_url) as db:
        versions = db.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert versions == [
        (1, "create_base_signal_schema"),
        (2, "add_lifecycle_telemetry_and_outbox_leases"),
        (3, "add_market_derived_outcome_fields"),
        (4, "add_feedback_token_consumption"),
        (5, "add_outbox_lease_fencing"),
        (6, "add_durable_setup_lifecycle"),
        (7, "add_feedback_audit_metadata"),
        (8, "create_durable_signal_outcomes"),
        (9, "create_point_in_time_shadow_ml_tables"),
        (10, "add_canonical_neon_storage_schema"),
        (11, "create_advika_signal_only_positions"),
    ]
    assert store.backend == "postgresql"
    assert journal.backend == "postgresql"



def test_postgres_concurrent_startup_serializes_versioned_migrations(postgres_url):
    schema = f"migration_race_{uuid.uuid4().hex}"
    with psycopg.connect(postgres_url) as db:
        db.execute(f'CREATE SCHEMA "{schema}"')

    def connect():
        return psycopg.connect(
            postgres_url, options=f"-c search_path={schema}", connect_timeout=5
        )

    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            applied = list(pool.map(
                lambda _: apply_migrations(connect, "postgresql"), range(6)
            ))

        assert sorted(version for batch in applied for version in batch) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        with connect() as db:
            versions = db.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
            assert versions == [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,)]
            assert db.execute("SELECT COUNT(*) FROM setup_lifecycle_events").fetchone()[0] == 0
    finally:
        with psycopg.connect(postgres_url) as db:
            db.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def test_postgres_legacy_schema_migration_preserves_existing_records(postgres_url):
    schema = f"migration_legacy_{uuid.uuid4().hex}"
    with psycopg.connect(postgres_url) as db:
        db.execute(f'CREATE SCHEMA "{schema}"')

    def connect():
        return psycopg.connect(
            postgres_url, options=f"-c search_path={schema}", connect_timeout=5
        )

    try:
        with connect() as db:
            db.execute("""CREATE TABLE alert_outbox (
                signal_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                next_attempt_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT,
                sent_at TEXT
            )""")
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
                entry_price DOUBLE PRECISION,
                stop_loss DOUBLE PRECISION,
                target_price DOUBLE PRECISION,
                feedback TEXT,
                feedback_note TEXT,
                reviewed_at TEXT,
                outcome_json TEXT NOT NULL DEFAULT '{}'
            )""")
            db.execute(
                """INSERT INTO alert_outbox
                   (signal_key,payload_json,status,attempts,created_at,next_attempt_at,updated_at)
                   VALUES ('legacy-pg-key','{}','pending',0,'a','a','a')"""
            )
            db.execute(
                """INSERT INTO signal_journal
                   (evaluation_id,strategy,symbol,candle_time,evaluated_at,decision,
                    diagnostics_json,signal_json,lifecycle)
                   VALUES ('legacy-pg-eval','ORB','BTCUSD','candle','evaluated-at',
                           'setup','{}','{}','setup_armed')"""
            )
            db.execute(
                """INSERT INTO signal_journal
                   (evaluation_id,strategy,symbol,candle_time,evaluated_at,decision,side,
                    diagnostics_json,signal_json,lifecycle,entry_price,stop_loss,target_price)
                   VALUES ('legacy-pg-confirmed','ORB','BTCUSD','candle-confirmed','0','setup','LONG',
                           '{}','{"atr":1.2}','telegram_sent',100.0,98.0,103.0)"""
            )
            db.execute(
                """INSERT INTO alert_outbox
                   (signal_key,payload_json,status,attempts,created_at,next_attempt_at,updated_at,sent_at)
                   VALUES ('BTCUSD:ORB:LONG:candle-confirmed',
                           '{"entry":100.5,"stop_loss":98.5,"take_profit_1":103.5,"take_profit_2":104.5,"atr":1.2}',
                           'sent',1,'a','a','a','b')"""
            )

        assert apply_migrations(connect, "postgresql") == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        with connect() as db:
            assert db.execute(
                "SELECT COUNT(*) FROM alert_outbox WHERE signal_key='legacy-pg-key'"
            ).fetchone()[0] == 1
            assert db.execute(
                "SELECT COUNT(*) FROM signal_journal WHERE evaluation_id='legacy-pg-eval'"
            ).fetchone()[0] == 1
            assert db.execute(
                "SELECT status,entry_price,stop_loss,tp1_price,tp2_price,atr "
                "FROM signal_outcomes WHERE evaluation_id='legacy-pg-confirmed'"
            ).fetchone() == ("pending", 100.5, 98.5, 103.5, 104.5, 1.2)
            columns = {
                row[0] for row in db.execute(
                    """SELECT column_name FROM information_schema.columns
                       WHERE table_schema = %s AND table_name = 'signal_journal'""",
                    (schema,),
                ).fetchall()
            }
            assert {"trigger_confirmed_at", "market_outcome_status", "market_outcome_json",
                    "feedback_source", "feedback_user_id"} <= columns
            assert db.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = 'signal_outcomes'",
                (schema,),
            ).fetchone()[0] == 1
            ml_table_count = db.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name LIKE 'ml_%%'",
                (schema,),
            ).fetchone()[0]
            assert ml_table_count == 6
            assert db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 11
        assert apply_migrations(connect, "postgresql") == []
    finally:
        with psycopg.connect(postgres_url) as db:
            db.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def test_postgres_normalized_outcome_rows_are_idempotent_and_restart_safe(postgres_url):
    journal = SignalJournal("./unused.sqlite3", postgres_url)
    symbol = f"OUT{uuid.uuid4().hex[:8]}"
    candle_time = datetime.now(timezone.utc).isoformat()
    journal.record_evaluation(
        "ORB", symbol, candle_time, "setup", {},
        {"side": "LONG", "stop_loss": 98.0, "take_profit_1": 103.0, "atr": 1.5},
    )
    evaluation_id = journal.evaluation_id("ORB", symbol, candle_time)
    journal.update_lifecycle(
        "ORB", symbol, candle_time, "trigger_confirmed", 100.0,
        event_at=datetime.now(timezone.utc).isoformat(),
        side="LONG", stop_loss=98.0, target_price=103.0, target_price_2=104.0,
        atr=1.5, horizon_bars=12, fee_pct=0.08, slippage_pct=0.04,
    )
    journal.update_lifecycle(
        "ORB", symbol, candle_time, "telegram_sent",
        event_at=datetime.now(timezone.utc).isoformat(),
    )

    pending = journal.pending_market_outcomes()
    assert any(row["evaluation_id"] == evaluation_id for row in pending)
    stored_context = next(
        row for row in journal.list_market_outcomes(status="pending")
        if row["evaluation_id"] == evaluation_id
    )
    assert stored_context["tp1_price"] == 103.0
    assert stored_context["tp2_price"] == 104.0
    assert stored_context["fee_pct"] == 0.08
    assert stored_context["slippage_pct"] == 0.04

    outcome = {
        "first_hit": "TP1", "final_result": "win", "tp1_hit": True,
        "tp2_hit": True, "stop_hit": False, "same_candle_ambiguous": False,
        "bars_to_first_hit": 1, "bars_to_resolution": 2, "bars_observed": 2,
        "elapsed_seconds": 600.0, "mfe_r": 2.0, "mae_r": 0.2,
        "mfe_pct": 4.0, "mae_pct": 0.4, "mfe_atr": 2.7, "mae_atr": 0.27,
        "exit_price": 104.0, "gross_return_pct": 4.0, "cost_pct": 0.12,
        "net_return_pct": 3.88, "resolution_at": datetime.now(timezone.utc).isoformat(),
    }
    assert journal.save_market_outcome(evaluation_id, "complete", outcome)
    assert not journal.save_market_outcome(
        evaluation_id, "complete", {**outcome, "final_result": "loss"}
    )
    restarted = SignalJournal("./unused.sqlite3", postgres_url)
    assert not any(
        row["evaluation_id"] == evaluation_id
        for row in restarted.pending_market_outcomes()
    )
    assert next(
        row for row in restarted.list_market_outcomes(status="complete")
        if row["evaluation_id"] == evaluation_id
    )["final_result"] == "win"

    with psycopg.connect(postgres_url) as db:
        db.execute("DELETE FROM signal_outcomes WHERE evaluation_id = %s", (evaluation_id,))
        db.execute("DELETE FROM signal_journal WHERE evaluation_id = %s", (evaluation_id,))


def test_postgres_canonical_neon_schema_uses_jsonb_timestamps_and_guards(postgres_url):
    SignalJournal("./unused.sqlite3", postgres_url)
    with psycopg.connect(postgres_url) as db:
        types = {
            (table, column): data_type
            for table, column, data_type in db.execute(
                """SELECT table_name,column_name,data_type
                   FROM information_schema.columns
                   WHERE table_schema=current_schema()
                     AND (table_name,column_name) IN (
                       ('strategy_signals','indicator_snapshot'),
                       ('strategy_signals','signal_timestamp'),
                       ('strategy_states','state_data'),
                       ('strategy_state_events','event_data'),
                       ('telegram_outbox','payload'),
                       ('ml_training_runs','metrics'),
                       ('ml_predictions','feature_snapshot')
                     )"""
            ).fetchall()
        }
        assert types[("strategy_signals", "indicator_snapshot")] == "jsonb"
        assert types[("strategy_signals", "signal_timestamp")] == "timestamp with time zone"
        assert types[("strategy_states", "state_data")] == "jsonb"
        assert types[("strategy_state_events", "event_data")] == "jsonb"
        assert types[("telegram_outbox", "payload")] == "jsonb"
        assert types[("ml_training_runs", "metrics")] == "jsonb"
        assert types[("ml_predictions", "feature_snapshot")] == "jsonb"

        triggers = {
            row[0] for row in db.execute(
                """SELECT tgname FROM pg_trigger
                   WHERE NOT tgisinternal AND tgname IN (
                     'trg_strategy_signals_immutable',
                     'trg_strategy_state_events_append_only'
                   )"""
            ).fetchall()
        }
        assert triggers == {
            "trg_strategy_signals_immutable",
            "trg_strategy_state_events_append_only",
        }
