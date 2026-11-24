import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import migrations

from app.signal_journal import SignalJournal
from app.state_store import SignalStore


def test_versioned_migrations_adopt_legacy_sqlite_tables_without_data_loss(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as db:
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
        db.execute(
            """INSERT INTO alert_outbox
               (signal_key,payload_json,status,attempts,created_at,next_attempt_at,updated_at)
               VALUES ('legacy-key','{}','pending',0,'a','a','a')"""
        )
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
               VALUES ('legacy-eval','ORB','BTCUSD','candle','evaluated-at',
                       'setup','{}','{}','setup_armed')"""
        )
        db.execute(
            """INSERT INTO signal_journal
               (evaluation_id,strategy,symbol,candle_time,evaluated_at,decision,side,
                diagnostics_json,signal_json,lifecycle,entry_price,stop_loss,target_price)
               VALUES ('legacy-confirmed','ORB','BTCUSD','candle-confirmed','0','setup','LONG',
                       '{}','{"atr":1.2}','telegram_sent',100.0,98.0,103.0)"""
        )
        db.execute(
            """INSERT INTO alert_outbox
               (signal_key,payload_json,status,attempts,created_at,next_attempt_at,updated_at,sent_at)
               VALUES ('BTCUSD:ORB:LONG:candle-confirmed',
                       '{"entry":100.5,"stop_loss":98.5,"take_profit_1":103.5,"take_profit_2":104.5,"atr":1.2,"trigger_confirmed_at":"2026-11-21T09:20:00+00:00"}',
                       'sent',1,'a','a','a','b')"""
        )

    store = SignalStore(str(path))
    journal = SignalJournal(str(path))

    with sqlite3.connect(path) as db:
        versions = db.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
        outbox_columns = {row[1] for row in db.execute("PRAGMA table_info(alert_outbox)")}
        journal_columns = {row[1] for row in db.execute("PRAGMA table_info(signal_journal)")}
        feedback_token_columns = {row[1] for row in db.execute("PRAGMA table_info(signal_feedback_tokens)")}
        lifecycle_current_exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='setup_lifecycle_current'").fetchone()
        lifecycle_events_exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='setup_lifecycle_events'").fetchone()
        signal_outcomes_exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='signal_outcomes'").fetchone()
        ml_tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ml_%'"
            ).fetchall()
        }
        outbox_count = db.execute("SELECT COUNT(*) FROM alert_outbox WHERE signal_key='legacy-key'").fetchone()[0]
        journal_count = db.execute("SELECT COUNT(*) FROM signal_journal").fetchone()[0]
        outcome_backfill = db.execute(
            "SELECT status,entry_price,stop_loss,tp1_price,tp2_price,atr "
            "FROM signal_outcomes WHERE evaluation_id='legacy-confirmed'"
        ).fetchone()

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
    assert "lease_until" in outbox_columns
    assert "trigger_confirmed_at" in journal_columns
    assert "trigger_to_telegram_seconds" in journal_columns
    assert "consumed_at" in feedback_token_columns
    assert "lease_token" in outbox_columns
    assert "feedback_source" in journal_columns
    assert "feedback_user_id" in journal_columns
    assert signal_outcomes_exists is not None
    assert ml_tables == {"ml_feature_snapshots", "ml_market_labels", "ml_training_runs", "ml_models", "ml_predictions", "ml_feedback_labels"}
    assert lifecycle_current_exists is not None
    assert lifecycle_events_exists is not None
    assert outbox_count == 1
    assert journal_count == 2
    assert outcome_backfill == ("pending", 100.5, 98.5, 103.5, 104.5, 1.2)
    assert store.outbox_health()["pending"] == 1
    assert journal.list_records(limit=1)[0]["evaluation_id"] == "legacy-eval"

    # Constructors can run repeatedly (including multiple bot instances) without
    # replaying or duplicating already-recorded schema migrations.
    SignalStore(str(path))
    SignalJournal(str(path))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 11



def test_migration_batch_rolls_back_fully_when_a_step_fails(tmp_path, monkeypatch):
    path = tmp_path / "atomic.sqlite3"

    def create_probe(db, backend):
        db.execute("CREATE TABLE migration_probe (id INTEGER PRIMARY KEY)")

    def fail_after_probe(db, backend):
        raise RuntimeError("intentional migration failure")

    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        (
            (1, "create_probe", create_probe),
            (2, "fail_after_probe", fail_after_probe),
        ),
    )

    with pytest.raises(RuntimeError, match="intentional migration failure"):
        migrations.apply_migrations(lambda: sqlite3.connect(path, timeout=5), "sqlite")

    with sqlite3.connect(path) as db:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "schema_migrations" not in tables
    assert "migration_probe" not in tables

    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        ((1, "create_probe", create_probe),),
    )
    assert migrations.apply_migrations(
        lambda: sqlite3.connect(path, timeout=5), "sqlite"
    ) == [1]
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM migration_probe").fetchone()[0] == 0


def test_concurrent_sqlite_startup_applies_each_migration_once(tmp_path):
    path = tmp_path / "concurrent-startup.sqlite3"

    def initialize():
        return SignalStore(str(path)).backend

    with ThreadPoolExecutor(max_workers=6) as pool:
        backends = list(pool.map(lambda _: initialize(), range(6)))

    assert backends == ["sqlite"] * 6
    with sqlite3.connect(path) as db:
        versions = db.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert versions == [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,)]
        assert db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='setup_lifecycle_events'"
        ).fetchone()[0] == 1
