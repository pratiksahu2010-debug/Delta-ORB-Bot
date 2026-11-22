"""Leakage-safe, strategy-isolated shadow ML research.

Features are immutable point-in-time snapshots derived only from the signal
and trigger telemetry available when the signal was confirmed. Market outcomes
are copied into a separate label table. Human feedback is never joined into
features or labels. Models only produce versioned shadow probabilities.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import uuid

import numpy as np


FEATURE_VERSION = "pit_orb_mex_v1"
LABEL_VERSION = "market_outcome_v1"
FEEDBACK_LABEL_VERSION = "human_feedback_v1"
ALGORITHM = "logistic_regression_l2_v1"
MIN_TRAIN_SAMPLES = 30
MIN_VALIDATION_SAMPLES = 10
MIN_TEST_SAMPLES = 10
MIN_TOTAL_SAMPLES = MIN_TRAIN_SAMPLES + MIN_VALIDATION_SAMPLES + MIN_TEST_SAMPLES

FEATURE_COLUMNS = (
    "side_long",
    "atr_pct",
    "risk_distance_atr",
    "rvol",
    "session_rvol",
    "atr_expansion",
    "rsi",
    "roc3",
    "range_atr",
    "body_atr",
    "impulse_atr",
    "roc3_atr",
    "extension_atr",
    "range_width_atr",
    "breakout_extension_atr",
    "ema_spread_atr",
    "ema21_slope_atr_3bars",
    "ema20_ema50_spread_pct",
    "context_score",
    "candle_quality_body_ratio",
    "candle_quality_close_location",
    "candle_quality_opposing_wick_ratio",
    "retest_touch",
    "structure_break",
    "context_ema_alignment",
    "context_di_pressure",
    "context_rsi_momentum",
    "context_vwap_position",
    "session_asia",
    "session_london",
    "session_new_york",
    "entry_path_retest",
    "entry_path_breakout",
    "mex_path_explosion",
    "entry_extension_atr",
    "trigger_overshoot_atr",
    "setup_to_trigger_seconds",
    "utc_hour",
    "utc_weekday",
)

_NUMERIC_INDICATOR_KEYS = {
    "risk_distance_atr", "rvol", "session_rvol", "atr_expansion", "rsi",
    "roc3", "range_atr", "body_atr", "impulse_atr", "roc3_atr",
    "extension_atr", "range_width_atr", "breakout_extension_atr",
    "ema_spread_atr", "ema21_slope_atr_3bars", "context_score",
}
_FORBIDDEN_FEATURE_FRAGMENTS = (
    "outcome", "future", "mfe", "mae", "feedback", "review", "pnl",
    "return", "label", "target_hit", "stop_hit", "result", "profit",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc(value) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value):
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def build_point_in_time_features(record: dict) -> dict:
    """Build a fixed, whitelisted feature vector without reading labels/feedback."""
    strategy = str(record.get("strategy") or "").upper()
    if strategy not in {"ORB", "MEX"}:
        raise ValueError("Only ORB and MEX have separate shadow models")

    signal = record.get("signal") or {}
    diagnostics = record.get("diagnostics") or {}
    if not isinstance(signal, dict):
        signal = {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    indicators = signal.get("indicators") or {}
    diagnostic_values = diagnostics.get("values") or {}
    if not isinstance(indicators, dict):
        indicators = {}
    if not isinstance(diagnostic_values, dict):
        diagnostic_values = {}
    features = {key: 0.0 for key in FEATURE_COLUMNS}

    for key in _NUMERIC_INDICATOR_KEYS:
        value = indicators.get(key)
        if value is None:
            value = diagnostic_values.get(key)
        number = _number(value)
        if number is not None:
            features[key] = number

    side = str(record.get("side") or signal.get("side") or "").upper()
    features["side_long"] = 1.0 if side == "LONG" else 0.0

    risk_distance = _number(signal.get("risk_distance_atr") or indicators.get("risk_distance_atr"))
    if risk_distance is not None:
        features["risk_distance_atr"] = risk_distance
    for key in ("retest_touch", "structure_break"):
        number = _number(indicators.get(key))
        if number is not None:
            features[key] = number

    atr_value = _number(signal.get("atr"))
    entry_reference = _number(signal.get("entry_reference") or record.get("entry_price"))
    if atr_value and entry_reference and entry_reference > 0:
        features["atr_pct"] = atr_value / entry_reference * 100.0

    ema20 = _number(indicators.get("ema20"))
    ema50 = _number(indicators.get("ema50"))
    if ema20 is not None and ema50 is not None and entry_reference and entry_reference > 0:
        features["ema20_ema50_spread_pct"] = (ema20 - ema50) / entry_reference * 100.0

    quality = indicators.get("candle_quality") or diagnostic_values.get("candle_quality") or {}
    if isinstance(quality, dict):
        for source, target in (
            ("body_ratio", "candle_quality_body_ratio"),
            ("close_location", "candle_quality_close_location"),
            ("opposing_wick_ratio", "candle_quality_opposing_wick_ratio"),
        ):
            number = _number(quality.get(source))
            if number is not None:
                features[target] = number

    context = indicators.get("context_checks") or {}
    if isinstance(context, dict):
        for source, target in (
            ("ema_alignment", "context_ema_alignment"),
            ("di_pressure", "context_di_pressure"),
            ("rsi_momentum", "context_rsi_momentum"),
            ("vwap_position", "context_vwap_position"),
        ):
            number = _number(context.get(source))
            if number is not None:
                features[target] = number

    session = str(indicators.get("session") or "").lower().replace(" ", "_").replace("-", "_")
    for name in ("asia", "london", "new_york"):
        features[f"session_{name}"] = float(session == name)
    entry_path = str(indicators.get("entry_path") or "").lower()
    features["entry_path_retest"] = float("retest" in entry_path)
    features["entry_path_breakout"] = float("breakout" in entry_path)
    features["mex_path_explosion"] = float(
        str(indicators.get("mex_path") or "").upper() == "MOMENTUM_EXPLOSION"
    )

    for source, target in (
        ("entry_extension_atr", "entry_extension_atr"),
        ("trigger_overshoot_atr", "trigger_overshoot_atr"),
        ("setup_to_trigger_seconds", "setup_to_trigger_seconds"),
    ):
        number = _number(record.get(source))
        if number is not None:
            features[target] = number

    candle_time = _utc(record.get("candle_time"))
    feature_at = _utc(record.get("feature_at") or record.get("trigger_confirmed_at") or candle_time)
    trigger_at = record.get("trigger_confirmed_at")
    if feature_at < candle_time:
        raise ValueError("point-in-time feature timestamp precedes its source candle")
    if trigger_at and feature_at > _utc(trigger_at):
        raise ValueError("point-in-time features contain information after signal confirmation")
    features["utc_hour"] = feature_at.hour + feature_at.minute / 60.0
    features["utc_weekday"] = float(feature_at.weekday())

    clean = {}
    for key in FEATURE_COLUMNS:
        if any(fragment in key.lower() for fragment in _FORBIDDEN_FEATURE_FRAGMENTS):
            raise ValueError(f"Forbidden target-like feature name: {key}")
        number = _number(features.get(key))
        clean[key] = number if number is not None else 0.0
    return clean


def chronological_split(rows: list[dict]) -> dict:
    """Chronological 60/20/20 split with a max-label-horizon embargo."""
    ordered = sorted(rows, key=lambda row: _utc(row["feature_at"]))
    if len(ordered) < MIN_TOTAL_SAMPLES:
        return {
            "train": [], "validation": [], "test": [],
            "embargo_seconds": 0, "reason": "insufficient_total_samples",
        }
    train_cut_index = max(1, int(len(ordered) * 0.60))
    validation_cut_index = max(train_cut_index + 1, int(len(ordered) * 0.80))
    validation_cut_index = min(validation_cut_index, len(ordered) - 1)
    train_end = _utc(ordered[train_cut_index - 1]["feature_at"])
    validation_end = _utc(ordered[validation_cut_index - 1]["feature_at"])
    max_horizon = max(max(1, int(row.get("horizon_bars") or 12)) for row in ordered)
    embargo_seconds = (max_horizon + 1) * 300
    validation_start = train_end + timedelta(seconds=embargo_seconds)
    test_start = validation_end + timedelta(seconds=embargo_seconds)

    train = [row for row in ordered if _utc(row["feature_at"]) <= train_end]
    validation = [
        row for row in ordered
        if validation_start < _utc(row["feature_at"]) <= validation_end
    ]
    test = [row for row in ordered if _utc(row["feature_at"]) > test_start]
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "embargo_seconds": embargo_seconds,
        "train_start_at": train[0]["feature_at"] if train else None,
        "train_end_at": train[-1]["feature_at"] if train else None,
        "validation_start_at": validation[0]["feature_at"] if validation else None,
        "validation_end_at": validation[-1]["feature_at"] if validation else None,
        "test_start_at": test[0]["feature_at"] if test else None,
        "test_end_at": test[-1]["feature_at"] if test else None,
    }


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _fit_logistic(x_train, y_train):
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales[scales < 1e-9] = 1.0
    normalized = (x_train - means) / scales
    design = np.column_stack([np.ones(len(normalized)), normalized])
    prevalence = float(np.mean(y_train))
    clipped = min(1.0 - 1e-5, max(1e-5, prevalence))
    weights = np.zeros(design.shape[1], dtype=float)
    weights[0] = math.log(clipped / (1.0 - clipped))
    regularization = 0.01
    learning_rate = 0.05
    for _ in range(400):
        probabilities = _sigmoid(design @ weights)
        gradient = (design.T @ (probabilities - y_train)) / len(y_train)
        gradient[1:] += regularization * weights[1:]
        weights -= learning_rate * gradient
    return {
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": weights.tolist(),
        "train_prevalence": prevalence,
    }


def _predict_probability(features: dict, model: dict) -> float:
    vector = np.asarray([features[name] for name in FEATURE_COLUMNS], dtype=float)
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    normalized = (vector - means) / scales
    design = np.concatenate(([1.0], normalized))
    return float(_sigmoid(np.asarray([design @ weights]))[0])


def _binary_metrics(y_true, probabilities, baseline_probability):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    predicted = p >= 0.5
    actual = y >= 0.5
    true_positive = int(np.sum(predicted & actual))
    false_positive = int(np.sum(predicted & ~actual))
    false_negative = int(np.sum(~predicted & actual))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    bins = []
    for index in range(5):
        lower, upper = index / 5.0, (index + 1) / 5.0
        mask = (p >= lower) & ((p < upper) if index < 4 else (p <= upper))
        if np.any(mask):
            bins.append({
                "lower": lower, "upper": upper, "count": int(np.sum(mask)),
                "mean_probability": float(np.mean(p[mask])),
                "observed_rate": float(np.mean(y[mask])),
            })
    return {
        "sample_size": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else None,
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "brier_score": float(np.mean((p - y) ** 2)) if len(y) else None,
        "baseline_brier_score": float(np.mean((baseline_probability - y) ** 2)) if len(y) else None,
        "calibration_bins": bins,
    }


class ShadowML:
    """Separate ORB/MEX point-in-time research and shadow inference service."""

    def __init__(self, journal):
        self.journal = journal
        self.backend = journal.backend
        self.placeholder = "?" if self.backend == "sqlite" else "%s"

    def _store_feature_snapshot(self, record: dict) -> bool:
        strategy = str(record.get("strategy") or "").upper()
        if strategy not in {"ORB", "MEX"}:
            return False
        features = build_point_in_time_features(record)
        feature_at = _utc(record.get("feature_at") or record.get("trigger_confirmed_at") or record["candle_time"]).isoformat()
        source = {
            "features": features,
            "feature_at": feature_at,
            "feature_version": FEATURE_VERSION,
        }
        source_hash = hashlib.sha256(
            json.dumps(source, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        ph = self.placeholder
        with self.journal._connect() as db:
            cursor = db.execute(
                f"""INSERT INTO ml_feature_snapshots
                    (evaluation_id,strategy,symbol,candle_time,feature_at,feature_version,
                     features_json,source_hash,created_at)
                    VALUES ({','.join([ph] * 9)})
                    ON CONFLICT(evaluation_id,feature_version) DO NOTHING""",
                (
                    record["evaluation_id"], strategy, record["symbol"],
                    record["candle_time"], feature_at, FEATURE_VERSION,
                    json.dumps(features, sort_keys=True, separators=(",", ":")),
                    source_hash, _now(),
                ),
            )
        return cursor.rowcount == 1

    def backfill_features(self, strategy: str) -> int:
        if strategy not in {"ORB", "MEX"}:
            raise ValueError("strategy must be ORB or MEX")
        records = self.journal.list_ml_source_records(strategy=strategy, limit=5000)
        return sum(1 for record in records if self._store_feature_snapshot(record))

    def sync_market_labels(self, strategy: str) -> int:
        if strategy not in {"ORB", "MEX"}:
            raise ValueError("strategy must be ORB or MEX")
        ph = self.placeholder
        with self.journal._connect() as db:
            rows = db.execute(
                f"""SELECT o.evaluation_id,o.strategy,o.horizon_bars,o.tp1_hit,o.stop_hit,
                           o.final_result,o.net_return_pct,o.resolution_at,o.updated_at,
                           o.outcome_json
                    FROM signal_outcomes o
                    JOIN signal_journal j
                      ON j.evaluation_id=o.evaluation_id AND j.strategy=o.strategy
                    WHERE o.strategy={ph} AND o.status='complete'
                      AND j.decision='setup' AND o.resolution_at IS NOT NULL
                      AND o.bars_observed > 0
                    ORDER BY o.resolution_at ASC""",
                (strategy,),
            ).fetchall()
            inserted = 0
            for row in rows:
                try:
                    outcome = json.loads(row[9] or "{}")
                except (TypeError, ValueError):
                    continue
                if outcome.get("method") != "completed_5m_ohlc_path":
                    continue
                label_at = row[7] or row[8]
                cursor = db.execute(
                    f"""INSERT INTO ml_market_labels
                        (evaluation_id,strategy,label_version,label_at,horizon_bars,
                         tp1_hit,stop_hit,final_result,net_return_pct,created_at)
                        VALUES ({','.join([ph] * 10)})
                        ON CONFLICT(evaluation_id,label_version) DO NOTHING""",
                    (
                        row[0], row[1], LABEL_VERSION, label_at, row[2],
                        int(bool(row[3])), int(bool(row[4])), row[5], row[6], _now(),
                    ),
                )
                inserted += int(cursor.rowcount == 1)
        return inserted

    def sync_human_feedback_labels(self, strategy: str) -> int:
        """Copy human reviews into their own table; never join them into market labels."""
        if strategy not in {"ORB", "MEX"}:
            raise ValueError("strategy must be ORB or MEX")
        ph = self.placeholder
        with self.journal._connect() as db:
            rows = db.execute(
                f"""SELECT evaluation_id,strategy,feedback,feedback_source,reviewed_at
                    FROM signal_journal
                    WHERE strategy={ph} AND decision='setup'
                      AND feedback IS NOT NULL AND TRIM(feedback) != ''
                    ORDER BY reviewed_at ASC""",
                (strategy,),
            ).fetchall()
            inserted = 0
            for row in rows:
                cursor = db.execute(
                    f"""INSERT INTO ml_feedback_labels
                        (evaluation_id,strategy,label_version,feedback_label,
                         feedback_source,feedback_at,created_at)
                        VALUES ({','.join([ph] * 7)})
                        ON CONFLICT(evaluation_id,label_version) DO NOTHING""",
                    (
                        row[0], row[1], FEEDBACK_LABEL_VERSION, row[2],
                        row[3], row[4], _now(),
                    ),
                )
                inserted += int(cursor.rowcount == 1)
        return inserted

    def capture_and_predict(self, evaluation_id: str):
        record = self.journal.get_ml_source_record(evaluation_id)
        if not record:
            return {"captured": False, "predicted": False, "reason": "confirmed_ORB_or_MEX_signal_not_found"}
        self._store_feature_snapshot(record)
        ph = self.placeholder
        with self.journal._connect() as db:
            snapshot = db.execute(
                f"""SELECT features_json FROM ml_feature_snapshots
                    WHERE evaluation_id={ph} AND feature_version={ph}""",
                (evaluation_id, FEATURE_VERSION),
            ).fetchone()
            model_row = db.execute(
                f"""SELECT model_version,parameters_json FROM ml_models
                    WHERE strategy={ph} AND feature_version={ph}
                      AND status='shadow' AND mode='shadow'
                    ORDER BY created_at DESC LIMIT 1""",
                (record["strategy"], FEATURE_VERSION),
            ).fetchone()
        if not snapshot or not model_row:
            return {"captured": bool(snapshot), "predicted": False, "reason": "no_shadow_model"}
        features = json.loads(snapshot[0] or "{}")
        parameters = json.loads(model_row[1] or "{}")
        p_tp1 = _predict_probability(features, parameters["tp1_model"])
        p_sl = _predict_probability(features, parameters["sl_model"])
        score = p_tp1 - p_sl
        model_version = model_row[0]
        with self.journal._connect() as db:
            db.execute(
                f"""INSERT INTO ml_predictions
                    (evaluation_id,model_version,strategy,feature_version,
                     predicted_tp1_probability,predicted_sl_probability,
                     shadow_score,mode,created_at)
                    VALUES ({','.join([ph] * 9)})
                    ON CONFLICT(evaluation_id,model_version) DO NOTHING""",
                (
                    evaluation_id, model_version, record["strategy"], FEATURE_VERSION,
                    p_tp1, p_sl, score, "shadow", _now(),
                ),
            )
        return {
            "captured": True, "predicted": True, "mode": "shadow",
            "evaluation_id": evaluation_id, "model_version": model_version,
            "tp1_probability": p_tp1, "sl_probability": p_sl, "shadow_score": score,
        }

    def _load_training_rows(self, strategy: str):
        ph = self.placeholder
        with self.journal._connect() as db:
            rows = db.execute(
                f"""SELECT f.evaluation_id,f.feature_at,f.features_json,
                           l.tp1_hit,l.stop_hit,l.net_return_pct,l.horizon_bars,l.label_at
                    FROM ml_feature_snapshots f
                    JOIN ml_market_labels l
                      ON l.evaluation_id=f.evaluation_id
                     AND l.strategy=f.strategy
                     AND l.label_version={ph}
                    WHERE f.strategy={ph} AND f.feature_version={ph}
                    ORDER BY f.feature_at ASC""",
                (LABEL_VERSION, strategy, FEATURE_VERSION),
            ).fetchall()
        result = []
        for row in rows:
            try:
                feature_at = _utc(row[1])
                label_at = _utc(row[7])
                features = json.loads(row[2] or "{}")
                vector = [_number(features.get(key)) for key in FEATURE_COLUMNS]
                if label_at <= feature_at or any(value is None for value in vector):
                    continue
                result.append({
                    "evaluation_id": row[0], "feature_at": feature_at.isoformat(),
                    "features": features, "vector": vector,
                    "tp1_hit": int(bool(row[3])), "stop_hit": int(bool(row[4])),
                    "net_return_pct": _number(row[5]),
                    "horizon_bars": int(row[6] or 12),
                })
            except (TypeError, ValueError, KeyError):
                continue
        return result

    def _update_training_run(self, run_id, status, *, train_count=0,
                             validation_count=0, test_count=0, split=None,
                             metrics=None, error=None, model_version=None):
        ph = self.placeholder
        split = split or {}
        with self.journal._connect() as db:
            db.execute(
                f"""UPDATE ml_training_runs SET status={ph},train_count={ph},
                    validation_count={ph},test_count={ph},train_start_at={ph},
                    train_end_at={ph},validation_start_at={ph},validation_end_at={ph},
                    test_start_at={ph},test_end_at={ph},embargo_seconds={ph},
                    split_json={ph},metrics_json={ph},error={ph},completed_at={ph}
                    WHERE run_id={ph}""",
                (
                    status, train_count, validation_count, test_count,
                    split.get("train_start_at"), split.get("train_end_at"),
                    split.get("validation_start_at"), split.get("validation_end_at"),
                    split.get("test_start_at"), split.get("test_end_at"),
                    int(split.get("embargo_seconds") or 0),
                    json.dumps(split, sort_keys=True, default=str),
                    json.dumps(metrics or {}, sort_keys=True, default=str),
                    error, _now(), run_id,
                ),
            )

    def train_shadow_model(self, strategy: str):
        strategy = str(strategy).upper()
        if strategy not in {"ORB", "MEX"}:
            raise ValueError("strategy must be ORB or MEX")
        run_id = uuid.uuid4().hex
        created_at = _now()
        ph = self.placeholder
        with self.journal._connect() as db:
            db.execute(
                f"""INSERT INTO ml_training_runs
                    (run_id,strategy,feature_version,label_version,algorithm,status,
                     metrics_json,split_json,created_at)
                    VALUES ({','.join([ph] * 9)})""",
                (
                    run_id, strategy, FEATURE_VERSION, LABEL_VERSION, ALGORITHM,
                    "running", "{}", "{}", created_at,
                ),
            )
        try:
            feature_count = self.backfill_features(strategy)
            label_count = self.sync_market_labels(strategy)
            feedback_label_count = self.sync_human_feedback_labels(strategy)
            rows = self._load_training_rows(strategy)
            split = chronological_split(rows)
            train, validation, test = split["train"], split["validation"], split["test"]
            split_metadata = {
                key: value for key, value in split.items()
                if key not in {"train", "validation", "test"}
            }
            counts = {
                "feature_snapshots_added": feature_count,
                "market_labels_added": label_count,
                "human_feedback_labels_added": feedback_label_count,
                "dataset_samples": len(rows),
                "train": len(train), "validation": len(validation), "test": len(test),
            }
            if (
                len(train) < MIN_TRAIN_SAMPLES
                or len(validation) < MIN_VALIDATION_SAMPLES
                or len(test) < MIN_TEST_SAMPLES
            ):
                self._update_training_run(
                    run_id, "insufficient_data", train_count=len(train),
                    validation_count=len(validation), test_count=len(test),
                    split=split_metadata, metrics={"sample_counts": counts},
                    error="Not enough chronologically separated labeled samples for all splits",
                )
                return {"run_id": run_id, "strategy": strategy, "status": "insufficient_data", **counts}

            x_train = np.asarray([row["vector"] for row in train], dtype=float)
            x_validation = np.asarray([row["vector"] for row in validation], dtype=float)
            x_test = np.asarray([row["vector"] for row in test], dtype=float)
            y_train_tp1 = np.asarray([row["tp1_hit"] for row in train], dtype=float)
            y_train_sl = np.asarray([row["stop_hit"] for row in train], dtype=float)
            y_validation_tp1 = np.asarray([row["tp1_hit"] for row in validation], dtype=float)
            y_validation_sl = np.asarray([row["stop_hit"] for row in validation], dtype=float)
            y_test_tp1 = np.asarray([row["tp1_hit"] for row in test], dtype=float)
            y_test_sl = np.asarray([row["stop_hit"] for row in test], dtype=float)

            tp1_model = _fit_logistic(x_train, y_train_tp1)
            sl_model = _fit_logistic(x_train, y_train_sl)
            p_validation_tp1 = np.asarray([
                _predict_probability(dict(zip(FEATURE_COLUMNS, vector)), tp1_model)
                for vector in x_validation
            ])
            p_validation_sl = np.asarray([
                _predict_probability(dict(zip(FEATURE_COLUMNS, vector)), sl_model)
                for vector in x_validation
            ])
            p_test_tp1 = np.asarray([
                _predict_probability(dict(zip(FEATURE_COLUMNS, vector)), tp1_model)
                for vector in x_test
            ])
            p_test_sl = np.asarray([
                _predict_probability(dict(zip(FEATURE_COLUMNS, vector)), sl_model)
                for vector in x_test
            ])
            tp1_base = float(np.mean(y_train_tp1))
            sl_base = float(np.mean(y_train_sl))
            validation_metrics = {
                "tp1": _binary_metrics(y_validation_tp1, p_validation_tp1, tp1_base),
                "sl": _binary_metrics(y_validation_sl, p_validation_sl, sl_base),
            }
            test_metrics = {
                "tp1": _binary_metrics(y_test_tp1, p_test_tp1, tp1_base),
                "sl": _binary_metrics(y_test_sl, p_test_sl, sl_base),
            }
            net_returns = np.asarray([
                row["net_return_pct"] if row["net_return_pct"] is not None else np.nan
                for row in test
            ], dtype=float)
            valid_returns = np.isfinite(net_returns)
            selected = (p_test_tp1 >= p_test_sl) & valid_returns
            baseline_expectancy = (
                float(np.mean(net_returns[valid_returns])) if np.any(valid_returns) else None
            )
            selected_expectancy = (
                float(np.mean(net_returns[selected])) if np.any(selected) else None
            )
            metrics = {
                "sample_counts": counts,
                "validation": validation_metrics,
                "test": test_metrics,
                "baseline_strategy_expectancy_pct": baseline_expectancy,
                "shadow_selected_expectancy_pct": selected_expectancy,
                "shadow_selected_samples": int(np.sum(selected)),
                "shadow_selected_coverage": (
                    float(np.sum(selected)) / int(np.sum(valid_returns))
                    if np.any(valid_returns) else None
                ),
                "promotion": "shadow_only_no_automatic_promotion",
            }
            model_version = f"{strategy.lower()}-{FEATURE_VERSION}-{run_id[:12]}"
            parameters = {
                "feature_columns": list(FEATURE_COLUMNS),
                "tp1_model": tp1_model,
                "sl_model": sl_model,
                "threshold": 0.5,
                "feature_version": FEATURE_VERSION,
                "label_version": LABEL_VERSION,
                "strategy": strategy,
                "mode": "shadow",
            }
            with self.journal._connect() as db:
                db.execute(
                    f"""INSERT INTO ml_models
                        (model_version,strategy,feature_version,label_version,algorithm,
                         train_run_id,mode,status,parameters_json,metrics_json,created_at)
                        VALUES ({','.join([ph] * 11)})""",
                    (
                        model_version, strategy, FEATURE_VERSION, LABEL_VERSION, ALGORITHM,
                        run_id, "shadow", "shadow",
                        json.dumps(parameters, sort_keys=True, separators=(",", ":")),
                        json.dumps(metrics, sort_keys=True, default=str), _now(),
                    ),
                )
            self._update_training_run(
                run_id, "complete", train_count=len(train),
                validation_count=len(validation), test_count=len(test),
                split=split_metadata, metrics=metrics,
            )
            return {
                "run_id": run_id, "model_version": model_version,
                "strategy": strategy, "status": "complete", **counts, "metrics": metrics,
            }
        except Exception as exc:
            self._update_training_run(
                run_id, "error", error=f"{type(exc).__name__}: {str(exc)[:240]}"
            )
            raise

    def status(self, limit: int = 20):
        ph = self.placeholder
        with self.journal._connect() as db:
            models = db.execute(
                f"""SELECT model_version,strategy,feature_version,label_version,
                           algorithm,train_run_id,mode,status,metrics_json,created_at
                    FROM ml_models ORDER BY created_at DESC LIMIT {ph}""",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
            runs = db.execute(
                f"""SELECT run_id,strategy,feature_version,algorithm,status,
                           train_count,validation_count,test_count,metrics_json,split_json,error,
                           created_at,completed_at
                    FROM ml_training_runs ORDER BY created_at DESC LIMIT {ph}""",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
            counts = {}
            for table in ("ml_feature_snapshots","ml_market_labels","ml_feedback_labels","ml_predictions"):
                counts[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            predictions = db.execute(
                f"""SELECT evaluation_id,model_version,strategy,feature_version,
                           predicted_tp1_probability,predicted_sl_probability,
                           shadow_score,mode,created_at
                    FROM ml_predictions ORDER BY created_at DESC LIMIT {ph}""",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        model_keys = [
            "model_version","strategy","feature_version","label_version","algorithm",
            "train_run_id","mode","status","metrics_json","created_at",
        ]
        run_keys = [
            "run_id","strategy","feature_version","algorithm","status","train_count",
            "validation_count","test_count","metrics_json","split_json","error","created_at","completed_at",
        ]
        def decode(rows, keys):
            result = []
            for row in rows:
                item = dict(zip(keys, row))
                try:
                    item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
                except (TypeError, ValueError):
                    item["metrics"] = {}
                    item.pop("metrics_json", None)
                if "split_json" in item:
                    try:
                        item["split"] = json.loads(item.pop("split_json") or "{}")
                    except (TypeError, ValueError):
                        item["split"] = {}
                        item.pop("split_json", None)
                result.append(item)
            return result
        prediction_keys = [
            "evaluation_id","model_version","strategy","feature_version",
            "predicted_tp1_probability","predicted_sl_probability","shadow_score",
            "mode","created_at",
        ]
        return {
            "mode": "shadow_only",
            "live_signal_decisions_modified": False,
            "automatic_promotion": False,
            "promotion_requires_explicit_approval": True,
            "counts": counts,
            "models": decode(models, model_keys),
            "training_runs": decode(runs, run_keys),
            "predictions": [dict(zip(prediction_keys, row)) for row in predictions],
            "feedback_used_as_market_label": False,
        }
