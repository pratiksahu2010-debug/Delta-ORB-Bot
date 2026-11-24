"""Versioned, transactional schema migrations for SQLite and PostgreSQL.

Migrations are additive and safe for databases created by earlier bot versions.
The first migration adopts the existing tables in place; it never drops data.
"""
from collections.abc import Callable
from datetime import datetime, timezone
import json
import math

MIGRATION_LOCK_ID = 731_904_221


def _migration_1_create_base_schema(db, backend: str) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS sent_signals (
        signal_key TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        setup_time TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        sent_at TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS alert_outbox (
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
    db.execute("""CREATE TABLE IF NOT EXISTS pending_setups (
        symbol TEXT NOT NULL,
        strategy TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (symbol, strategy)
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS signal_journal (
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
    note_id = "INTEGER PRIMARY KEY AUTOINCREMENT" if backend == "sqlite" else "BIGSERIAL PRIMARY KEY"
    db.execute(f"""CREATE TABLE IF NOT EXISTS training_notes (
        note_id {note_id},
        created_at TEXT NOT NULL,
        note TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'telegram'
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS signal_feedback_tokens (
        token TEXT PRIMARY KEY,
        evaluation_id TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_pending_setups_expiry ON pending_setups(expires_at)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_alert_outbox_due ON alert_outbox(status, next_attempt_at)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_signal_journal_time ON signal_journal(evaluated_at)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_signal_journal_filter ON signal_journal(strategy, decision, symbol)")


def _add_column_if_missing(db, backend: str, table: str, column: str, definition: str) -> None:
    if backend == "sqlite":
        existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    else:
        db.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {definition}")


def _migration_2_add_lifecycle_and_outbox_leases(db, backend: str) -> None:
    telemetry_columns = {
        "setup_armed_at": "TEXT",
        "trigger_confirmed_at": "TEXT",
        "telegram_sent_at": "TEXT",
        "trigger_price": "REAL",
        "setup_to_trigger_seconds": "REAL",
        "entry_extension_atr": "REAL",
        "trigger_overshoot_atr": "REAL",
        "trigger_to_telegram_seconds": "REAL",
    }
    for column, definition in telemetry_columns.items():
        _add_column_if_missing(db, backend, "signal_journal", column, definition)
    _add_column_if_missing(db, backend, "alert_outbox", "lease_until", "TEXT")



def _migration_3_add_market_outcome_fields(db, backend: str) -> None:
    _add_column_if_missing(db, backend, "signal_journal", "market_outcome_status", "TEXT NOT NULL DEFAULT 'not_started'")
    _add_column_if_missing(db, backend, "signal_journal", "market_outcome_json", "TEXT NOT NULL DEFAULT '{}'")
    _add_column_if_missing(db, backend, "signal_journal", "market_outcome_checked_at", "TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS idx_signal_journal_outcome ON signal_journal(lifecycle, market_outcome_status, evaluated_at)")


def _migration_4_add_feedback_token_consumption(db, backend: str) -> None:
    _add_column_if_missing(db, backend, "signal_feedback_tokens", "consumed_at", "TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tokens_consumed ON signal_feedback_tokens(consumed_at)")


def _migration_5_add_outbox_lease_fencing(db, backend: str) -> None:
    _add_column_if_missing(db, backend, "alert_outbox", "lease_token", "TEXT")



def _migration_6_add_durable_setup_lifecycle(db, backend: str) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS setup_lifecycle_current (
        setup_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        strategy TEXT NOT NULL,
        candle_time TEXT NOT NULL,
        state TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS setup_lifecycle_events (
        setup_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        strategy TEXT NOT NULL,
        candle_time TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        reason TEXT NOT NULL,
        event_at TEXT NOT NULL,
        PRIMARY KEY (setup_id, sequence)
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_setup_lifecycle_events_time ON setup_lifecycle_events(event_at)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_setup_lifecycle_current_state ON setup_lifecycle_current(state, updated_at)")


def _migration_7_add_feedback_audit_metadata(db, backend: str) -> None:
    _add_column_if_missing(db, backend, "signal_journal", "feedback_source", "TEXT")
    _add_column_if_missing(db, backend, "signal_journal", "feedback_user_id", "TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS idx_signal_journal_feedback_source ON signal_journal(feedback_source, reviewed_at)")


def _migration_8_create_signal_outcomes(db, backend: str) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS signal_outcomes (
        evaluation_id TEXT PRIMARY KEY,
        strategy TEXT NOT NULL,
        symbol TEXT NOT NULL,
        candle_time TEXT NOT NULL,
        side TEXT NOT NULL CHECK (side IN ('LONG','SHORT')),
        trigger_confirmed_at TEXT,
        entry_price REAL NOT NULL,
        stop_loss REAL NOT NULL,
        tp1_price REAL NOT NULL,
        tp2_price REAL NOT NULL,
        atr REAL,
        horizon_bars INTEGER NOT NULL DEFAULT 12 CHECK (horizon_bars BETWEEN 1 AND 288),
        fee_pct REAL NOT NULL DEFAULT 0 CHECK (fee_pct >= 0),
        slippage_pct REAL NOT NULL DEFAULT 0 CHECK (slippage_pct >= 0),
        status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','complete','error')),
        first_hit TEXT CHECK (first_hit IS NULL OR first_hit IN ('TP1','TP2','SL')),
        first_hit_at TEXT,
        final_result TEXT CHECK (final_result IS NULL OR final_result IN ('win','loss','ambiguous','expired','pending')),
        tp1_hit INTEGER NOT NULL DEFAULT 0,
        tp2_hit INTEGER NOT NULL DEFAULT 0,
        stop_hit INTEGER NOT NULL DEFAULT 0,
        same_candle_ambiguous INTEGER NOT NULL DEFAULT 0,
        bars_to_first_hit INTEGER,
        bars_to_resolution INTEGER,
        bars_observed INTEGER NOT NULL DEFAULT 0,
        elapsed_seconds REAL,
        mfe_r REAL,
        mae_r REAL,
        mfe_pct REAL,
        mae_pct REAL,
        mfe_atr REAL,
        mae_atr REAL,
        exit_price REAL,
        gross_return_pct REAL,
        cost_pct REAL,
        net_return_pct REAL,
        resolution_at TEXT,
        outcome_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcomes_status ON signal_outcomes(status, trigger_confirmed_at)")
    placeholder = "?" if backend == "sqlite" else "%s"
    rows = db.execute(
        """SELECT evaluation_id,strategy,symbol,candle_time,side,entry_price,
                  stop_loss,target_price,trigger_confirmed_at,signal_json
           FROM signal_journal
           WHERE lifecycle IN ('trigger_confirmed','telegram_sent')
             AND entry_price IS NOT NULL AND stop_loss IS NOT NULL"""
    ).fetchall()
    now = datetime.now(timezone.utc).isoformat()
    insert_sql = f"""INSERT INTO signal_outcomes
        (evaluation_id,strategy,symbol,candle_time,side,trigger_confirmed_at,
         entry_price,stop_loss,tp1_price,tp2_price,atr,horizon_bars,fee_pct,
         slippage_pct,status,outcome_json,created_at,updated_at)
        VALUES ({','.join([placeholder] * 18)})
        ON CONFLICT(evaluation_id) DO NOTHING"""
    for row in rows:
        evaluation_id, strategy, symbol, candle_time, side, entry, stop, target, trigger_at, signal_json = row
        side = str(side or '').upper()
        if side not in {'LONG', 'SHORT'}:
            continue
        try:
            signal = json.loads(signal_json or '{}')
        except (TypeError, ValueError):
            signal = {}
        signal_key = f"{symbol}:{strategy}:{side}:{candle_time}"
        outbox = db.execute(
            f"SELECT payload_json FROM alert_outbox WHERE signal_key={placeholder}",
            (signal_key,),
        ).fetchone()
        try:
            alert = json.loads(outbox[0] or '{}') if outbox else {}
        except (TypeError, ValueError):
            alert = {}
        try:
            entry = float(alert.get("entry") or entry)
            stop = float(alert.get("stop_loss") or stop)
            risk = entry - stop if side == "LONG" else stop - entry
            if not (math.isfinite(entry) and math.isfinite(stop) and risk > 0):
                continue
            tp1 = float(alert.get("take_profit_1") or target or (
                entry + 1.5 * risk if side == "LONG" else entry - 1.5 * risk
            ))
            tp2 = float(alert.get("take_profit_2") or (
                entry + 2.0 * risk if side == "LONG" else entry - 2.0 * risk
            ))
            if side == "LONG" and not (entry < tp1 < tp2):
                tp1, tp2 = entry + 1.5 * risk, entry + 2.0 * risk
            if side == "SHORT" and not (entry > tp1 > tp2):
                tp1, tp2 = entry - 1.5 * risk, entry - 2.0 * risk
            atr_value = alert.get("atr") or signal.get("atr") or signal.get("atr_value")
            atr_value = float(atr_value) if atr_value is not None else None
            if atr_value is not None and (not math.isfinite(atr_value) or atr_value <= 0):
                atr_value = None
            horizon = int(alert.get("outcome_horizon_bars") or 12)
            fee = float(alert.get("outcome_fee_pct") or 0.0)
            slippage = float(alert.get("outcome_slippage_pct") or 0.0)
            trigger_at = alert.get("trigger_confirmed_at") or trigger_at
        except (TypeError, ValueError, OverflowError):
            continue
        db.execute(insert_sql, (
            evaluation_id, strategy, symbol, candle_time, side, trigger_at,
            entry, stop, tp1, tp2, atr_value, max(1, min(horizon, 288)),
            max(0.0, fee), max(0.0, slippage), "pending", "{}",
            now, now,
        ))


def _migration_9_create_shadow_ml_tables(db, backend: str) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS ml_feature_snapshots (
        evaluation_id TEXT NOT NULL,
        strategy TEXT NOT NULL CHECK (strategy IN ('ORB','MEX')),
        symbol TEXT NOT NULL,
        candle_time TEXT NOT NULL,
        feature_at TEXT NOT NULL,
        feature_version TEXT NOT NULL,
        features_json TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (evaluation_id, feature_version)
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ml_features_strategy_time ON ml_feature_snapshots(strategy, feature_at)")
    db.execute("""CREATE TABLE IF NOT EXISTS ml_market_labels (
        evaluation_id TEXT NOT NULL,
        strategy TEXT NOT NULL CHECK (strategy IN ('ORB','MEX')),
        label_version TEXT NOT NULL,
        label_at TEXT NOT NULL,
        horizon_bars INTEGER NOT NULL,
        tp1_hit INTEGER NOT NULL,
        stop_hit INTEGER NOT NULL,
        final_result TEXT NOT NULL,
        net_return_pct REAL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (evaluation_id, label_version)
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ml_labels_strategy_time ON ml_market_labels(strategy, label_at)")
    db.execute("""CREATE TABLE IF NOT EXISTS ml_training_runs (
        run_id TEXT PRIMARY KEY,
        strategy TEXT NOT NULL CHECK (strategy IN ('ORB','MEX')),
        feature_version TEXT NOT NULL,
        label_version TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('running','complete','insufficient_data','error')),
        train_count INTEGER NOT NULL DEFAULT 0,
        validation_count INTEGER NOT NULL DEFAULT 0,
        test_count INTEGER NOT NULL DEFAULT 0,
        train_start_at TEXT,
        train_end_at TEXT,
        validation_start_at TEXT,
        validation_end_at TEXT,
        test_start_at TEXT,
        test_end_at TEXT,
        embargo_seconds INTEGER NOT NULL DEFAULT 0,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        split_json TEXT NOT NULL DEFAULT '{}',
        error TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ml_runs_strategy_created ON ml_training_runs(strategy, created_at)")
    db.execute("""CREATE TABLE IF NOT EXISTS ml_models (
        model_version TEXT PRIMARY KEY,
        strategy TEXT NOT NULL CHECK (strategy IN ('ORB','MEX')),
        feature_version TEXT NOT NULL,
        label_version TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        train_run_id TEXT NOT NULL,
        mode TEXT NOT NULL DEFAULT 'shadow' CHECK (mode = 'shadow'),
        status TEXT NOT NULL DEFAULT 'shadow' CHECK (status IN ('shadow','retired')),
        parameters_json TEXT NOT NULL,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ml_models_strategy_status ON ml_models(strategy, status, created_at)")
    db.execute("""CREATE TABLE IF NOT EXISTS ml_predictions (
        evaluation_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        strategy TEXT NOT NULL CHECK (strategy IN ('ORB','MEX')),
        feature_version TEXT NOT NULL,
        predicted_tp1_probability REAL NOT NULL,
        predicted_sl_probability REAL NOT NULL,
        shadow_score REAL NOT NULL,
        mode TEXT NOT NULL DEFAULT 'shadow' CHECK (mode = 'shadow'),
        created_at TEXT NOT NULL,
        PRIMARY KEY (evaluation_id, model_version)
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_strategy_created ON ml_predictions(strategy, created_at)")
    db.execute("""CREATE TABLE IF NOT EXISTS ml_feedback_labels (
        evaluation_id TEXT NOT NULL,
        strategy TEXT NOT NULL CHECK (strategy IN ('ORB','MEX')),
        label_version TEXT NOT NULL,
        feedback_label TEXT NOT NULL,
        feedback_source TEXT,
        feedback_at TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (evaluation_id, label_version)
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ml_feedback_strategy_time ON ml_feedback_labels(strategy, feedback_at)")


def _migration_10_add_canonical_neon_storage(db, backend: str) -> None:
    json_type = "JSONB" if backend == "postgresql" else "TEXT"
    json_default = "'{}'::jsonb" if backend == "postgresql" else "'{}'"
    timestamp_type = "TIMESTAMPTZ" if backend == "postgresql" else "TEXT"
    numeric_type = "NUMERIC(24,10)" if backend == "postgresql" else "REAL"

    db.execute(f"""CREATE TABLE IF NOT EXISTS strategy_signals (
        id TEXT NOT NULL,
        signal_id TEXT NOT NULL UNIQUE,
        exchange TEXT NOT NULL,
        symbol TEXT NOT NULL,
        module TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        direction TEXT NOT NULL,
        entry_price {numeric_type} NOT NULL,
        stop_loss {numeric_type} NOT NULL,
        target1 {numeric_type} NOT NULL,
        target2 {numeric_type},
        risk_reward {numeric_type},
        signal_timestamp {timestamp_type} NOT NULL,
        candle_timestamp {timestamp_type} NOT NULL,
        indicator_snapshot {json_type} NOT NULL DEFAULT {json_default},
        strategy_config {json_type} NOT NULL DEFAULT {json_default},
        strategy_version TEXT,
        state_cycle_id TEXT,
        status TEXT NOT NULL,
        created_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_signals_symbol ON strategy_signals(symbol)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_signals_module ON strategy_signals(module)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_signals_timestamp ON strategy_signals(signal_timestamp)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_signals_status ON strategy_signals(status)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_signals_created ON strategy_signals(created_at)")
    if backend == "sqlite":
        db.execute("""CREATE TRIGGER IF NOT EXISTS trg_strategy_signals_no_update
            BEFORE UPDATE ON strategy_signals BEGIN
            SELECT RAISE(ABORT,'strategy_signals is immutable');
        END""")
        db.execute("""CREATE TRIGGER IF NOT EXISTS trg_strategy_signals_no_delete
            BEFORE DELETE ON strategy_signals BEGIN
            SELECT RAISE(ABORT,'strategy_signals is immutable');
        END""")

    db.execute(f"""CREATE TABLE IF NOT EXISTS strategy_states (
        id TEXT NOT NULL,
        exchange TEXT NOT NULL,
        symbol TEXT NOT NULL,
        module TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        state TEXT NOT NULL,
        direction TEXT,
        setup_id TEXT NOT NULL,
        state_data {json_type} NOT NULL DEFAULT {json_default},
        last_processed_candle {timestamp_type},
        state_version INTEGER NOT NULL DEFAULT 0,
        updated_at {timestamp_type} NOT NULL,
        UNIQUE(exchange,symbol,module,timeframe)
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_states_symbol ON strategy_states(symbol)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_states_module_state ON strategy_states(module,state)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_states_updated ON strategy_states(updated_at)")

    db.execute(f"""CREATE TABLE IF NOT EXISTS strategy_state_events (
        id TEXT NOT NULL,
        setup_id TEXT NOT NULL,
        exchange TEXT NOT NULL,
        symbol TEXT NOT NULL,
        module TEXT NOT NULL,
        previous_state TEXT,
        new_state TEXT NOT NULL,
        reason TEXT,
        candle_timestamp {timestamp_type},
        event_data {json_type} NOT NULL DEFAULT {json_default},
        created_at {timestamp_type} NOT NULL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_state_events_setup ON strategy_state_events(setup_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_state_events_symbol ON strategy_state_events(symbol)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_strategy_state_events_module_time ON strategy_state_events(module,created_at)")
    if backend == "sqlite":
        db.execute("""CREATE TRIGGER IF NOT EXISTS trg_strategy_state_events_no_update
            BEFORE UPDATE ON strategy_state_events BEGIN
            SELECT RAISE(ABORT,'strategy_state_events is append-only');
        END""")
        db.execute("""CREATE TRIGGER IF NOT EXISTS trg_strategy_state_events_no_delete
            BEFORE DELETE ON strategy_state_events BEGIN
            SELECT RAISE(ABORT,'strategy_state_events is append-only');
        END""")
    else:
        db.execute("""CREATE OR REPLACE FUNCTION reject_canonical_immutable_mutation()
            RETURNS trigger AS $BODY$
            BEGIN
                RAISE EXCEPTION '% is immutable or append-only', TG_TABLE_NAME;
            END;
            $BODY$ LANGUAGE plpgsql""")
        for table, trigger in (
            ("strategy_signals", "trg_strategy_signals_immutable"),
            ("strategy_state_events", "trg_strategy_state_events_append_only"),
        ):
            exists = db.execute(
                """SELECT 1 FROM pg_trigger t
                   JOIN pg_class c ON c.oid=t.tgrelid
                   JOIN pg_namespace n ON n.oid=c.relnamespace
                   WHERE t.tgname=%s AND n.nspname=current_schema()
                     AND NOT t.tgisinternal""",
                (trigger,),
            ).fetchone()
            if not exists:
                db.execute(
                    f"CREATE TRIGGER {trigger} BEFORE UPDATE OR DELETE ON {table} "
                    "FOR EACH ROW EXECUTE FUNCTION reject_canonical_immutable_mutation()"
                )

    db.execute(f"""CREATE TABLE IF NOT EXISTS signal_feedback (
        id TEXT NOT NULL UNIQUE,
        signal_id TEXT NOT NULL REFERENCES strategy_signals(signal_id),
        module TEXT NOT NULL,
        symbol TEXT NOT NULL,
        category TEXT NOT NULL CHECK (category IN (
            'GOOD_ENTRY','BAD_SIGNAL','LATE_ENTRY','FAKE_BREAKOUT',
            'WRONG_DIRECTION','MISSED_OPPORTUNITY','NEEDS_REVIEW',
            'STOP_LOSS_TOO_TIGHT','TARGET_TOO_CLOSE','TARGET_REACHED',
            'EXIT_TOO_EARLY','SKIP','OTHER'
        )),
        comment TEXT,
        source TEXT NOT NULL,
        user_reference TEXT NOT NULL DEFAULT '',
        created_at {timestamp_type} NOT NULL,
        UNIQUE(signal_id,category,source,user_reference)
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_signal_feedback_signal ON signal_feedback(signal_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_signal_feedback_category ON signal_feedback(category,created_at)")

    db.execute(f"""CREATE TABLE IF NOT EXISTS telegram_outbox (
        id TEXT NOT NULL UNIQUE,
        signal_id TEXT NOT NULL REFERENCES strategy_signals(signal_id),
        idempotency_key TEXT NOT NULL UNIQUE,
        payload {json_type} NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt_at {timestamp_type},
        lease_until {timestamp_type},
        last_error_type TEXT,
        created_at {timestamp_type} NOT NULL,
        sent_at {timestamp_type}
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_telegram_outbox_status_next ON telegram_outbox(status,next_attempt_at)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_telegram_outbox_created ON telegram_outbox(created_at)")

    outcome_columns = {
        "id": "TEXT", "signal_id": "TEXT", "module": "TEXT", "result": "TEXT",
        "target1_hit": "INTEGER", "target2_hit": "INTEGER", "stop_loss_hit": "INTEGER",
        "best_favorable_pct": numeric_type, "worst_adverse_pct": numeric_type,
        "best_favorable_atr": numeric_type, "worst_adverse_atr": numeric_type,
        "outcome_horizon_bars": "INTEGER", "resolved_at": timestamp_type,
    }
    for column, definition in outcome_columns.items():
        _add_column_if_missing(db, backend, "signal_outcomes", column, definition)
    resolved_copy = "resolution_at::timestamptz" if backend == "postgresql" else "resolution_at"
    db.execute(f"""UPDATE signal_outcomes SET
        id=COALESCE(id,evaluation_id), signal_id=COALESCE(signal_id,evaluation_id),
        module=COALESCE(module,strategy), result=COALESCE(result,final_result),
        target1_hit=COALESCE(target1_hit,tp1_hit),
        target2_hit=COALESCE(target2_hit,tp2_hit),
        stop_loss_hit=COALESCE(stop_loss_hit,stop_hit),
        best_favorable_pct=COALESCE(best_favorable_pct,mfe_pct),
        worst_adverse_pct=COALESCE(worst_adverse_pct,mae_pct),
        best_favorable_atr=COALESCE(best_favorable_atr,mfe_atr),
        worst_adverse_atr=COALESCE(worst_adverse_atr,mae_atr),
        outcome_horizon_bars=COALESCE(outcome_horizon_bars,horizon_bars),
        resolved_at=COALESCE(resolved_at,{resolved_copy})""")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_outcomes_signal_id ON signal_outcomes(signal_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_signal_outcomes_module_result ON signal_outcomes(strategy,final_result,resolution_at)")

    training_columns = {
        "id": "TEXT", "model_name": "TEXT", "model_version": "TEXT", "module": "TEXT",
        "training_started_at": timestamp_type, "training_finished_at": timestamp_type,
        "dataset_start": timestamp_type, "dataset_end": timestamp_type,
        "training_samples": "INTEGER", "validation_samples": "INTEGER", "test_samples": "INTEGER",
        "metrics": f"{json_type} DEFAULT {json_default}", "model_artifact_reference": "TEXT",
    }
    for column, definition in training_columns.items():
        _add_column_if_missing(db, backend, "ml_training_runs", column, definition)
    metrics_copy = "metrics_json::jsonb" if backend == "postgresql" else "metrics_json"
    empty_metrics = "'{}'::jsonb" if backend == "postgresql" else "'{}'"
    created_copy = "created_at::timestamptz" if backend == "postgresql" else "created_at"
    completed_copy = "completed_at::timestamptz" if backend == "postgresql" else "completed_at"
    train_start_copy = "train_start_at::timestamptz" if backend == "postgresql" else "train_start_at"
    test_end_copy = "test_end_at::timestamptz" if backend == "postgresql" else "test_end_at"
    db.execute(f"""UPDATE ml_training_runs SET
        id=COALESCE(id,run_id), model_name=COALESCE(model_name,algorithm),
        module=COALESCE(module,strategy),
        training_started_at=COALESCE(training_started_at,{created_copy}),
        training_finished_at=COALESCE(training_finished_at,{completed_copy}),
        dataset_start=COALESCE(dataset_start,{train_start_copy}),
        dataset_end=COALESCE(dataset_end,{test_end_copy}),
        training_samples=COALESCE(training_samples,train_count),
        validation_samples=COALESCE(validation_samples,validation_count),
        test_samples=COALESCE(test_samples,test_count),
        metrics=CASE WHEN metrics IS NULL OR metrics={empty_metrics}
                     THEN {metrics_copy} ELSE metrics END""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ml_training_runs_module_status ON ml_training_runs(strategy,status,created_at)")

    prediction_columns = {
        "id": "TEXT", "signal_id": "TEXT", "module": "TEXT",
        "predicted_probability": numeric_type, "predicted_class": "TEXT",
        "feature_snapshot": f"{json_type} DEFAULT {json_default}",
        "prediction_timestamp": timestamp_type, "actual_outcome": "TEXT",
    }
    for column, definition in prediction_columns.items():
        _add_column_if_missing(db, backend, "ml_predictions", column, definition)
    db.execute(f"""UPDATE ml_predictions SET
        id=COALESCE(id,evaluation_id || ':' || model_version),
        signal_id=COALESCE(signal_id,evaluation_id),
        module=COALESCE(module,strategy),
        predicted_probability=COALESCE(predicted_probability,predicted_tp1_probability),
        prediction_timestamp=COALESCE(prediction_timestamp,{created_copy})""")
    if backend == "postgresql":
        db.execute("""UPDATE ml_predictions p SET feature_snapshot=f.features_json::jsonb
            FROM ml_feature_snapshots f
            WHERE p.evaluation_id=f.evaluation_id AND p.feature_version=f.feature_version
              AND p.feature_snapshot='{}'::jsonb""")
    else:
        db.execute("""UPDATE ml_predictions SET feature_snapshot=(
            SELECT f.features_json FROM ml_feature_snapshots f
            WHERE f.evaluation_id=ml_predictions.evaluation_id
              AND f.feature_version=ml_predictions.feature_version
        ) WHERE feature_snapshot='{}'""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_signal_module_time ON ml_predictions(signal_id,strategy,created_at)")

    feedback_columns = {
        "id": "TEXT", "signal_id": "TEXT", "feedback_id": "TEXT",
        "module": "TEXT", "label": "TEXT", "label_source": "TEXT",
        "confidence": numeric_type,
    }
    for column, definition in feedback_columns.items():
        _add_column_if_missing(db, backend, "ml_feedback_labels", column, definition)
    db.execute("""UPDATE ml_feedback_labels SET
        id=COALESCE(id,evaluation_id || ':' || label_version),
        signal_id=COALESCE(signal_id,evaluation_id),
        module=COALESCE(module,strategy),
        label=COALESCE(label,feedback_label),
        label_source=COALESCE(label_source,feedback_source)""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_ml_feedback_signal_module ON ml_feedback_labels(signal_id,strategy,created_at)")


MIGRATIONS: tuple[tuple[int, str, Callable], ...] = (
    (1, "create_base_signal_schema", _migration_1_create_base_schema),
    (2, "add_lifecycle_telemetry_and_outbox_leases", _migration_2_add_lifecycle_and_outbox_leases),
    (3, "add_market_derived_outcome_fields", _migration_3_add_market_outcome_fields),
    (4, "add_feedback_token_consumption", _migration_4_add_feedback_token_consumption),
    (5, "add_outbox_lease_fencing", _migration_5_add_outbox_lease_fencing),
    (6, "add_durable_setup_lifecycle", _migration_6_add_durable_setup_lifecycle),
    (7, "add_feedback_audit_metadata", _migration_7_add_feedback_audit_metadata),
    (8, "create_durable_signal_outcomes", _migration_8_create_signal_outcomes),
    (9, "create_point_in_time_shadow_ml_tables", _migration_9_create_shadow_ml_tables),
    (10, "add_canonical_neon_storage_schema", _migration_10_add_canonical_neon_storage),
    (11, "create_advika_signal_only_positions", _migration_11_create_advika_signal_positions),
)



def apply_migrations(connect: Callable, backend: str) -> list[int]:
    """Apply every pending migration atomically and return applied versions.

    SQLite uses BEGIN IMMEDIATE to serialize concurrent startup. PostgreSQL uses
    a transaction-scoped advisory lock so multiple Render instances cannot race.
    """
    if backend not in {"sqlite", "postgresql"}:
        raise ValueError(f"Unsupported database backend: {backend}")

    applied_now: list[int] = []
    with connect() as db:
        if backend == "sqlite":
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
        else:
            db.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
        db.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )""")
        applied = {
            int(row[0]) for row in db.execute("SELECT version FROM schema_migrations").fetchall()
        }
        for version, name, migrate in MIGRATIONS:
            if version in applied:
                continue
            migrate(db, backend)
            placeholder = "?" if backend == "sqlite" else "%s"
            from datetime import datetime, timezone
            db.execute(
                f"INSERT INTO schema_migrations (version, name, applied_at) VALUES ({placeholder}, {placeholder}, {placeholder})",
                (version, name, datetime.now(timezone.utc).isoformat()),
            )
            applied_now.append(version)
    return applied_now


def verify_migrations(connect: Callable, backend: str) -> list[int]:
    """Verify an already-migrated database without creating or changing schema."""
    if backend not in {"sqlite", "postgresql"}:
        raise ValueError(f"Unsupported database backend: {backend}")
    try:
        with connect() as db:
            rows = db.execute("SELECT version FROM schema_migrations").fetchall()
            applied = {int(row[0]) for row in rows}
    except Exception as exc:
        raise RuntimeError(
            "Database schema is not initialized; run 'python -m app.migrate' before starting production"
        ) from exc
    expected = {version for version, _name, _migrate in MIGRATIONS}
    missing = sorted(expected - applied)
    if missing:
        raise RuntimeError(
            f"Database schema is missing migrations {missing}; run 'python -m app.migrate' before starting production"
        )
    return sorted(applied)
