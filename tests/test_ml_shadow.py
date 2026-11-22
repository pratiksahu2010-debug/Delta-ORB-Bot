from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from app.ml_shadow import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    LABEL_VERSION,
    ShadowML,
    build_point_in_time_features,
    chronological_split,
)
from app.signal_journal import SignalJournal


def _source_record(strategy="ORB", side="LONG", evaluation_id="eval-1"):
    return {
        "evaluation_id": evaluation_id,
        "strategy": strategy,
        "symbol": "BTCUSD",
        "candle_time": "2026-11-21T09:15:00+00:00",
        "feature_at": "2026-11-21T09:17:00+00:00",
        "trigger_confirmed_at": "2026-11-21T09:17:00+00:00",
        "side": side,
        "setup_to_trigger_seconds": 120.0,
        "entry_extension_atr": 0.25,
        "trigger_overshoot_atr": 0.1,
        "diagnostics": {
            "values": {"rsi": 67.0, "future_tp1_hit": 1, "mfe_r": 9.0}
        },
        "signal": {
            "side": side,
            "entry_reference": 100.0,
            "atr": 2.0,
            "risk_distance_atr": 1.5,
            "indicators": {
                "rvol": 1.8,
                "session_rvol": 1.4,
                "range_width_atr": 2.1,
                "breakout_extension_atr": 0.35,
                "ema20": 101.0,
                "ema50": 100.0,
                "session": "new_york",
                "entry_path": "ORB_RETEST",
                "retest_touch": True,
                "structure_break": True,
                "candle_quality": {
                    "body_ratio": 0.65,
                    "close_location": 0.9,
                    "opposing_wick_ratio": 0.1,
                },
                "context_checks": {"ema_alignment": True, "di_pressure": False},
            },
            "outcome": {"tp1_hit": True},
            "feedback": "bad",
            "mfe_r": 8.0,
        },
        "feedback": "bad",
        "market_outcome": {"final_result": "win"},
    }


def test_point_in_time_feature_builder_uses_whitelist_and_excludes_labels_feedback():
    record = _source_record()
    features = build_point_in_time_features(record)

    assert set(features) == set(FEATURE_COLUMNS)
    assert features["side_long"] == 1.0
    assert features["atr_pct"] == 2.0
    assert features["risk_distance_atr"] == 1.5
    assert features["rvol"] == 1.8
    assert features["rsi"] == 67.0
    assert features["candle_quality_body_ratio"] == 0.65
    assert features["context_ema_alignment"] == 1.0
    assert features["context_di_pressure"] == 0.0
    assert features["session_new_york"] == 1.0
    assert features["entry_path_retest"] == 1.0
    assert features["entry_extension_atr"] == 0.25
    assert not any("feedback" in key or "outcome" in key or "mfe" in key for key in features)


def test_point_in_time_feature_builder_rejects_features_after_confirmation():
    record = _source_record()
    record["feature_at"] = "2026-11-21T09:18:00+00:00"
    with pytest.raises(ValueError, match="after signal confirmation"):
        build_point_in_time_features(record)


def test_chronological_split_has_non_overlapping_horizon_embargoes():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "feature_at": (start + timedelta(hours=6 * index)).isoformat(),
            "horizon_bars": 12,
        }
        for index in range(100)
    ]

    split = chronological_split(rows)

    assert len(split["train"]) == 60
    assert len(split["validation"]) == 20
    assert len(split["test"]) == 20
    assert split["embargo_seconds"] == 3900
    train_end = datetime.fromisoformat(split["train"][-1]["feature_at"])
    validation_start = datetime.fromisoformat(split["validation"][0]["feature_at"])
    validation_end = datetime.fromisoformat(split["validation"][-1]["feature_at"])
    test_start = datetime.fromisoformat(split["test"][0]["feature_at"])
    assert validation_start > train_end + timedelta(seconds=split["embargo_seconds"])
    assert test_start > validation_end + timedelta(seconds=split["embargo_seconds"])


def _insert_shadow_model(journal, model_version, strategy, created_at):
    weights = [0.0] * (len(FEATURE_COLUMNS) + 1)
    zero_model = {
        "means": [0.0] * len(FEATURE_COLUMNS),
        "scales": [1.0] * len(FEATURE_COLUMNS),
        "weights": weights,
    }
    parameters = {
        "feature_columns": list(FEATURE_COLUMNS),
        "tp1_model": zero_model,
        "sl_model": zero_model,
        "strategy": strategy,
        "mode": "shadow",
    }
    with journal._connect() as db:
        db.execute(
            """INSERT INTO ml_models
               (model_version,strategy,feature_version,label_version,algorithm,
                train_run_id,mode,status,parameters_json,metrics_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                model_version, strategy, FEATURE_VERSION, LABEL_VERSION,
                "test_logistic", f"run-{model_version}", "shadow", "shadow",
                json.dumps(parameters), "{}", created_at,
            ),
        )


def test_shadow_predictions_are_versioned_and_do_not_mutate_signal_decision(tmp_path):
    journal = SignalJournal(str(tmp_path / "shadow-predictions.sqlite3"))
    candle_time = "2026-11-21T09:15:00+00:00"
    original = {
        "side": "LONG", "entry_reference": 100.0, "atr": 2.0,
        "stop_loss": 98.0, "take_profit_1": 103.0,
        "indicators": {"rvol": 1.8, "session": "london", "entry_path": "ORB_BREAKOUT"},
    }
    journal.record_evaluation("ORB", "BTCUSD", candle_time, "setup", {}, original)
    evaluation_id = journal.evaluation_id("ORB", "BTCUSD", candle_time)
    journal.update_lifecycle(
        "ORB", "BTCUSD", candle_time, "trigger_confirmed", 100.5,
        event_at="2026-11-21T09:17:00+00:00",
        side="LONG", stop_loss=98.0, target_price=103.5,
        target_price_2=104.5, atr=2.0,
    )
    service = ShadowML(journal)
    first = service.capture_and_predict(evaluation_id)
    assert first["captured"] is True
    assert first["predicted"] is False

    _insert_shadow_model(journal, "orb-shadow-v1", "ORB", "2026-11-21T09:20:00+00:00")
    second = service.capture_and_predict(evaluation_id)
    assert second["predicted"] is True
    assert second["mode"] == "shadow"
    assert second["model_version"] == "orb-shadow-v1"
    assert second["tp1_probability"] == 0.5
    assert second["sl_probability"] == 0.5

    _insert_shadow_model(journal, "orb-shadow-v2", "ORB", "2026-11-21T09:21:00+00:00")
    third = service.capture_and_predict(evaluation_id)
    assert third["model_version"] == "orb-shadow-v2"

    record = journal.list_records(limit=1)[0]
    assert record["decision"] == "setup"
    assert record["signal"] == original
    assert record["lifecycle"] == "trigger_confirmed"
    with sqlite3.connect(tmp_path / "shadow-predictions.sqlite3") as db:
        predictions = db.execute(
            "SELECT model_version,mode FROM ml_predictions WHERE evaluation_id=? ORDER BY model_version",
            (evaluation_id,),
        ).fetchall()
    assert predictions == [("orb-shadow-v1", "shadow"), ("orb-shadow-v2", "shadow")]


def test_market_outcome_labels_are_separate_from_human_feedback(tmp_path):
    journal = SignalJournal(str(tmp_path / "separate-labels.sqlite3"))
    candle_time = "2026-11-21T10:00:00+00:00"
    signal = {
        "side": "LONG", "entry_reference": 100.0, "atr": 1.5,
        "stop_loss": 98.0, "take_profit_1": 103.0,
        "indicators": {"rvol": 1.7, "body_atr": 1.2},
    }
    journal.record_evaluation("ORB", "BTCUSD", candle_time, "setup", {}, signal)
    evaluation_id = journal.evaluation_id("ORB", "BTCUSD", candle_time)
    journal.update_lifecycle(
        "ORB", "BTCUSD", candle_time, "trigger_confirmed", 100.0,
        event_at="2026-11-21T10:02:00+00:00",
        side="LONG", stop_loss=98.0, target_price=103.0,
        target_price_2=104.0, atr=1.5,
    )
    assert journal.add_feedback(evaluation_id, "bad", note="manual label only")
    outcome = {
        "method": "completed_5m_ohlc_path",
        "first_hit": "TP1", "final_result": "win", "result": "win",
        "tp1_hit": True, "tp2_hit": True, "stop_hit": False,
        "same_candle_ambiguous": False, "bars_observed": 2,
        "bars_to_resolution": 2, "net_return_pct": 3.2,
        "resolution_at": "2026-11-21T10:15:00+00:00",
    }
    assert journal.save_market_outcome(evaluation_id, "complete", outcome)

    service = ShadowML(journal)
    assert service.backfill_features("ORB") == 1
    assert service.sync_market_labels("ORB") == 1
    assert service.sync_human_feedback_labels("ORB") == 1
    with sqlite3.connect(tmp_path / "separate-labels.sqlite3") as db:
        label = db.execute(
            "SELECT tp1_hit,stop_hit,final_result,net_return_pct "
            "FROM ml_market_labels WHERE evaluation_id=?",
            (evaluation_id,),
        ).fetchone()
        feature_json = db.execute(
            "SELECT features_json FROM ml_feature_snapshots WHERE evaluation_id=?",
            (evaluation_id,),
        ).fetchone()[0]
        label_columns = {
            row[1] for row in db.execute("PRAGMA table_info(ml_market_labels)").fetchall()
        }
        feedback_label = db.execute(
            "SELECT feedback_label FROM ml_feedback_labels WHERE evaluation_id=?",
            (evaluation_id,),
        ).fetchone()[0]
    assert label == (1, 0, "win", 3.2)
    assert feedback_label == "bad"
    assert "feedback" not in label_columns
    assert "bad" not in feature_json
    assert "outcome" not in feature_json
    assert "mfe" not in feature_json


def test_orb_and_mex_train_separate_shadow_models_with_oos_metrics(tmp_path):
    journal = SignalJournal(str(tmp_path / "separate-models.sqlite3"))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with journal._connect() as db:
        for strategy in ("ORB", "MEX"):
            for index in range(100):
                feature_at = start + timedelta(hours=6 * index)
                label_at = feature_at + timedelta(hours=1)
                evaluation_id = f"{strategy}-sample-{index}"
                features = {key: 0.0 for key in FEATURE_COLUMNS}
                features["rvol"] = 1.0 + (index % 5) / 2.0
                features["rsi"] = float(30 + index % 60)
                features["side_long"] = float(index % 2 == 0)
                tp1_hit = int(index % 2 == 0)
                stop_hit = int(index % 3 == 0)
                net_return = 1.5 if tp1_hit and not stop_hit else -0.8 if stop_hit else 0.2
                db.execute(
                    """INSERT INTO ml_feature_snapshots
                       (evaluation_id,strategy,symbol,candle_time,feature_at,
                        feature_version,features_json,source_hash,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        evaluation_id, strategy, "BTCUSD", feature_at.isoformat(),
                        feature_at.isoformat(), FEATURE_VERSION,
                        json.dumps(features), f"hash-{evaluation_id}", feature_at.isoformat(),
                    ),
                )
                db.execute(
                    """INSERT INTO ml_market_labels
                       (evaluation_id,strategy,label_version,label_at,horizon_bars,
                        tp1_hit,stop_hit,final_result,net_return_pct,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        evaluation_id, strategy, LABEL_VERSION, label_at.isoformat(), 12,
                        tp1_hit, stop_hit, "win" if tp1_hit else "loss",
                        net_return, label_at.isoformat(),
                    ),
                )

    service = ShadowML(journal)
    orb = service.train_shadow_model("ORB")
    mex = service.train_shadow_model("MEX")

    assert orb["status"] == "complete"
    assert mex["status"] == "complete"
    assert orb["strategy"] == "ORB"
    assert mex["strategy"] == "MEX"
    assert orb["model_version"].startswith("orb-")
    assert mex["model_version"].startswith("mex-")
    for result in (orb, mex):
        assert result["train"] == 60
        assert result["validation"] == 20
        assert result["test"] == 20
        assert result["metrics"]["test"]["tp1"]["sample_size"] == 20
        assert "precision_at_0_5" in result["metrics"]["test"]["tp1"]
        assert "recall_at_0_5" in result["metrics"]["test"]["sl"]
        assert "calibration_bins" in result["metrics"]["test"]["tp1"]
        assert result["metrics"]["baseline_strategy_expectancy_pct"] is not None
        assert result["metrics"]["promotion"] == "shadow_only_no_automatic_promotion"

    status = service.status()
    assert status["mode"] == "shadow_only"
    assert status["live_signal_decisions_modified"] is False
    assert status["automatic_promotion"] is False
    assert status["promotion_requires_explicit_approval"] is True
    assert {model["strategy"] for model in status["models"]} == {"ORB", "MEX"}
    assert status["counts"]["ml_predictions"] == 0
