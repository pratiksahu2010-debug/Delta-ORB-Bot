import json
import sqlite3
from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.migrations import MIGRATIONS, apply_migrations, verify_migrations
from app.signal_journal import SignalJournal
from app.state_store import SignalStore


def _now():
    return datetime.now(timezone.utc).isoformat()


def _insert_signal(db, signal_id="signal-1"):
    now = _now()
    db.execute(
        """INSERT INTO strategy_signals
           (id,signal_id,exchange,symbol,module,timeframe,direction,entry_price,
            stop_loss,target1,target2,risk_reward,signal_timestamp,candle_timestamp,
            indicator_snapshot,strategy_config,strategy_version,state_cycle_id,status,created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            signal_id, signal_id, "DELTA", "BTCUSD", "ORB", "5m", "LONG",
            100.0, 98.0, 103.0, 104.0, 1.5, now, now, "{}", "{}", "v1",
            None, "CONFIRMED", now,
        ),
    )


def test_canonical_neon_tables_and_typed_columns_are_created(tmp_path):
    journal = SignalJournal(str(tmp_path / "canonical.sqlite3"))
    with journal._connect() as db:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "strategy_signals", "strategy_states", "strategy_state_events",
            "signal_feedback", "signal_outcomes", "telegram_outbox",
            "ml_training_runs", "ml_predictions", "ml_feedback_labels",
        }.issubset(tables)

        expected_columns = {
            "strategy_signals": {
                "id", "signal_id", "exchange", "symbol", "module", "timeframe",
                "direction", "entry_price", "stop_loss", "target1", "target2",
                "risk_reward", "signal_timestamp", "candle_timestamp",
                "indicator_snapshot", "strategy_config", "strategy_version",
                "state_cycle_id", "status", "created_at",
            },
            "strategy_states": {
                "id", "exchange", "symbol", "module", "timeframe", "state",
                "direction", "setup_id", "state_data", "last_processed_candle",
                "state_version", "updated_at",
            },
            "strategy_state_events": {
                "id", "setup_id", "exchange", "symbol", "module",
                "previous_state", "new_state", "reason", "candle_timestamp",
                "event_data", "created_at",
            },
            "signal_feedback": {
                "id", "signal_id", "module", "symbol", "category", "comment",
                "source", "user_reference", "created_at",
            },
            "telegram_outbox": {
                "id", "signal_id", "idempotency_key", "payload", "status",
                "attempts", "next_attempt_at", "lease_until", "last_error_type",
                "created_at", "sent_at",
            },
        }
        for table, expected in expected_columns.items():
            columns = {
                row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()
            }
            assert expected.issubset(columns)

        outcome_columns = {
            row[1] for row in db.execute("PRAGMA table_info(signal_outcomes)").fetchall()
        }
        assert {
            "id", "signal_id", "module", "result", "target1_hit", "target2_hit",
            "stop_loss_hit", "best_favorable_pct", "worst_adverse_pct",
            "best_favorable_atr", "worst_adverse_atr", "outcome_horizon_bars",
            "cost_pct", "resolved_at", "created_at",
        }.issubset(outcome_columns)

        for table, columns in (
            ("ml_training_runs", {"id", "model_name", "model_version", "module",
             "training_started_at", "training_finished_at", "dataset_start",
             "dataset_end", "training_samples", "validation_samples",
             "test_samples", "metrics", "feature_version", "status",
             "model_artifact_reference", "created_at"}),
            ("ml_predictions", {"id", "signal_id", "module", "model_version",
             "predicted_probability", "predicted_class", "feature_snapshot",
             "prediction_timestamp", "actual_outcome", "created_at"}),
            ("ml_feedback_labels", {"id", "signal_id", "feedback_id", "module",
             "label", "label_source", "confidence", "created_at"}),
        ):
            columns_found = {
                row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()
            }
            assert columns.issubset(columns_found)


def test_feedback_categories_duplicates_foreign_keys_and_outbox_idempotency(tmp_path):
    journal = SignalJournal(str(tmp_path / "canonical-constraints.sqlite3"))
    now = _now()
    with journal._connect() as db:
        _insert_signal(db)

        feedback = (
            "feedback-1", "signal-1", "ORB", "BTCUSD", "GOOD_ENTRY",
            "clean retest", "telegram", "user-1", now,
        )
        db.execute(
            """INSERT INTO signal_feedback
               (id,signal_id,module,symbol,category,comment,source,user_reference,created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            feedback,
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO signal_feedback
                   (id,signal_id,module,symbol,category,comment,source,user_reference,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                ("feedback-2", *feedback[1:]),
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO signal_feedback
                   (id,signal_id,module,symbol,category,comment,source,user_reference,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                ("feedback-3", "signal-1", "ORB", "BTCUSD", "GOOD",
                 "invalid category", "telegram", "user-2", now),
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO signal_feedback
                   (id,signal_id,module,symbol,category,source,user_reference,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                ("feedback-4", "missing-signal", "ORB", "BTCUSD",
                 "GOOD_ENTRY", "telegram", "user-4", now),
            )

        outbox = (
            "outbox-1", "signal-1", "idempotency-1", "{}", "pending",
            0, now, None, None, now, None,
        )
        db.execute(
            """INSERT INTO telegram_outbox
               (id,signal_id,idempotency_key,payload,status,attempts,next_attempt_at,
                lease_until,last_error_type,created_at,sent_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            outbox,
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO telegram_outbox
                   (id,signal_id,idempotency_key,payload,status,attempts,next_attempt_at,
                    lease_until,last_error_type,created_at,sent_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                ("outbox-2", *outbox[1:]),
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """INSERT INTO telegram_outbox
                   (id,signal_id,idempotency_key,payload,status,attempts,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                ("outbox-3", "missing-signal", "idempotency-3", "{}", "pending", 0, now),
            )


def test_strategy_signal_snapshot_and_state_events_are_immutable(tmp_path):
    journal = SignalJournal(str(tmp_path / "append-only.sqlite3"))
    with journal._connect() as db:
        _insert_signal(db)
        db.execute(
            """INSERT INTO strategy_state_events
               (id,setup_id,exchange,symbol,module,previous_state,new_state,
                reason,candle_timestamp,event_data,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            ("event-1", "setup-1", "DELTA", "BTCUSD", "ORB",
             "ARMED", "CONFIRMING", "retest", _now(), "{}", _now()),
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(
                "UPDATE strategy_signals SET status='CLOSED' WHERE signal_id='signal-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute(
                "UPDATE strategy_state_events SET reason='changed' WHERE id='event-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM strategy_state_events WHERE id='event-1'")


def test_production_schema_verification_is_read_only_and_requires_neon(tmp_path):
    path = tmp_path / "unmigrated.sqlite3"
    with pytest.raises(RuntimeError, match="python -m app.migrate"):
        SignalJournal(str(path), migrate_on_startup=False)
    with sqlite3.connect(path) as db:
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "schema_migrations" not in tables

    with pytest.raises(RuntimeError, match="python -m app.migrate"):
        SignalStore(str(tmp_path / "unmigrated-store.sqlite3"), migrate_on_startup=False)

    with pytest.raises(ValueError, match="PostgreSQL"):
        Settings(app_env="production", database_url="sqlite:///tmp/prod.sqlite3").validate_safety()
    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        Settings(app_env="production", database_url="").validate_safety()

    Settings(
        app_env="production",
        database_url="postgresql://user:password@localhost:5432/neon",
    ).validate_safety()

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("RENDER_SERVICE_ID", "srv-test")
        render_settings = Settings(app_env="development", database_url="")
        assert render_settings.is_production is True
        with pytest.raises(ValueError, match="DATABASE_URL is required"):
            render_settings.validate_safety()
    finally:
        monkeypatch.undo()


def test_migration_10_backfills_existing_outcome_and_ml_rows(tmp_path):
    path = tmp_path / "legacy-v9.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL
        )""")
        for version, name, migration in MIGRATIONS[:9]:
            migration(db, "sqlite")
            db.execute(
                "INSERT INTO schema_migrations(version,name,applied_at) VALUES(?,?,?)",
                (version, name, _now()),
            )

        now = _now()
        db.execute(
            """INSERT INTO signal_outcomes
               (evaluation_id,strategy,symbol,candle_time,side,entry_price,stop_loss,
                tp1_price,tp2_price,horizon_bars,fee_pct,slippage_pct,status,
                final_result,tp1_hit,tp2_hit,stop_hit,mfe_pct,mae_pct,mfe_atr,
                mae_atr,resolution_at,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("eval-1","ORB","BTCUSD","candle","LONG",100,98,103,104,12,0,0,
             "complete","win",1,1,0,2.5,-0.5,1.2,-0.2,now,now,now),
        )
        db.execute(
            """INSERT INTO ml_training_runs
               (run_id,strategy,feature_version,label_version,algorithm,status,
                train_count,validation_count,test_count,train_start_at,test_end_at,
                metrics_json,split_json,created_at,completed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("run-1","ORB","features-v1","labels-v1","logistic","complete",
             60,20,20,now,now,'{"test":1}','{}',now,now),
        )
        db.execute(
            """INSERT INTO ml_feature_snapshots
               (evaluation_id,strategy,symbol,candle_time,feature_at,feature_version,
                features_json,source_hash,created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("eval-1","ORB","BTCUSD","candle",now,"features-v1",'{"rvol":1.5}',"hash",now),
        )
        db.execute(
            """INSERT INTO ml_predictions
               (evaluation_id,model_version,strategy,feature_version,
                predicted_tp1_probability,predicted_sl_probability,shadow_score,mode,created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("eval-1","model-1","ORB","features-v1",0.7,0.2,0.5,"shadow",now),
        )
        db.execute(
            """INSERT INTO ml_feedback_labels
               (evaluation_id,strategy,label_version,feedback_label,feedback_source,created_at)
               VALUES (?,?,?,?,?,?)""",
            ("eval-1","ORB","feedback-v1","GOOD_ENTRY","telegram",now),
        )

    assert apply_migrations(lambda: sqlite3.connect(path), "sqlite") == [10]
    with sqlite3.connect(path) as db:
        outcome = db.execute(
            """SELECT id,signal_id,module,result,target1_hit,target2_hit,
                      stop_loss_hit,best_favorable_pct,worst_adverse_atr,
                      outcome_horizon_bars,resolved_at
               FROM signal_outcomes WHERE evaluation_id='eval-1'"""
        ).fetchone()
        training = db.execute(
            """SELECT id,model_name,module,training_samples,validation_samples,
                      test_samples,metrics FROM ml_training_runs WHERE run_id='run-1'"""
        ).fetchone()
        prediction = db.execute(
            """SELECT id,signal_id,module,predicted_probability,predicted_class,
                      feature_snapshot FROM ml_predictions WHERE evaluation_id='eval-1'"""
        ).fetchone()
        feedback = db.execute(
            """SELECT id,signal_id,module,label,label_source
               FROM ml_feedback_labels WHERE evaluation_id='eval-1'"""
        ).fetchone()

    assert outcome[:10] == (
        "eval-1","eval-1","ORB","win",1,1,0,2.5,-0.2,12
    )
    assert outcome[10] == now
    assert training[:6] == ("run-1","logistic","ORB",60,20,20)
    assert json.loads(training[6]) == {"test": 1}
    assert prediction[:5] == ("eval-1:model-1","eval-1","ORB",0.7,None)
    assert json.loads(prediction[5]) == {"rvol": 1.5}
    assert feedback == ("eval-1:feedback-v1","eval-1","ORB","GOOD_ENTRY","telegram")
