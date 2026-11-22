from datetime import datetime, timezone

import pytest

from app.outcomes import evaluate_market_outcome


def candle(stamp, open_, high, low, close):
    return {
        "timestamp": datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc),
        "open": open_, "high": high, "low": low, "close": close,
    }


def record(side="LONG", **overrides):
    values = {
        "side": side,
        "entry_price": 100.0,
        "stop_loss": 98.0 if side == "LONG" else 102.0,
        "tp1_price": 103.0 if side == "LONG" else 97.0,
        "tp2_price": 104.0 if side == "LONG" else 96.0,
        "atr": 1.0,
        "fee_pct": 0.10,
        "slippage_pct": 0.05,
        "horizon_bars": 3,
        "trigger_confirmed_at": "2026-11-20T10:02:00+00:00",
        "candle_time": "2026-11-20T10:00:00+00:00",
    }
    values.update(overrides)
    return values


def test_tp1_first_then_tp2_records_both_hits_and_cost_adjusted_metrics():
    bars = [
        candle("2026-11-20T10:05:00", 100, 103.5, 99.0, 103.0),
        candle("2026-11-20T10:10:00", 103.2, 104.2, 102.5, 104.0),
    ]

    result = evaluate_market_outcome(record(), bars)

    assert result["first_hit"] == "TP1"
    assert result["final_result"] == "win"
    assert result["tp1_hit"] is True
    assert result["tp2_hit"] is True
    assert result["stop_hit"] is False
    assert result["bars_to_first_hit"] == 1
    assert result["bars_to_resolution"] == 2
    assert result["elapsed_seconds"] == 780.0
    assert result["exit_price"] == 104.0
    assert result["mfe_r"] == 2.1
    assert result["mae_r"] == 0.5
    assert result["mfe_pct"] == 4.2
    assert result["mae_pct"] == 1.0
    assert result["mfe_atr"] == 4.2
    assert result["mae_atr"] == 1.0
    assert result["gross_return_pct"] == 4.0
    assert result["cost_pct"] == 0.15
    assert result["net_return_pct"] == 3.85
    assert result["complete"] is True


def test_tp2_gap_is_first_hit_and_immediately_resolves_as_win():
    bars = [candle("2026-11-20T10:05:00", 104.2, 105.0, 103.5, 104.5)]

    result = evaluate_market_outcome(record(), bars)

    assert result["first_hit"] == "TP2"
    assert result["tp1_hit"] is True
    assert result["tp2_hit"] is True
    assert result["final_result"] == "win"
    assert result["exit_reason"] == "tp2_gap"


def test_stop_first_records_loss_and_round_trip_costs():
    bars = [candle("2026-11-20T10:05:00", 100, 101, 97, 98)]

    result = evaluate_market_outcome(record(), bars)

    assert result["first_hit"] == "SL"
    assert result["stop_hit"] is True
    assert result["final_result"] == "loss"
    assert result["exit_price"] == 98.0
    assert result["gross_return_pct"] == -2.0
    assert result["net_return_pct"] == -2.15


def test_same_candle_stop_and_target_uses_conservative_stop_first():
    bars = [candle("2026-11-20T10:05:00", 100, 104.5, 97, 101)]

    result = evaluate_market_outcome(record(), bars)

    assert result["first_hit"] == "SL"
    assert result["tp1_hit"] is True
    assert result["tp2_hit"] is True
    assert result["stop_hit"] is True
    assert result["same_candle_ambiguous"] is True
    assert result["final_result"] == "loss"
    assert result["exit_price"] == 98.0
    assert "stop-first" in result["ambiguity_rule"]


def test_same_candle_stop_vs_tp2_after_tp1_is_flagged_ambiguous():
    bars = [
        candle("2026-11-20T10:05:00", 103.1, 103.5, 102.5, 103.2),
        candle("2026-11-20T10:10:00", 103.2, 104.5, 97, 100),
    ]

    result = evaluate_market_outcome(record(), bars)

    assert result["first_hit"] == "TP1"
    assert result["tp1_hit"] is True
    assert result["tp2_hit"] is True
    assert result["stop_hit"] is True
    assert result["same_candle_ambiguous"] is True
    assert result["final_result"] == "loss"
    assert result["exit_price"] == 98.0


def test_horizon_expiry_and_incomplete_history_are_distinguished():
    quiet = [
        candle("2026-11-20T10:05:00", 100, 100.5, 99.5, 100.1),
        candle("2026-11-20T10:10:00", 100.1, 100.4, 99.6, 100.0),
        candle("2026-11-20T10:15:00", 100.0, 100.3, 99.7, 100.1),
    ]
    expired = evaluate_market_outcome(record(), quiet)
    assert expired["final_result"] == "expired"
    assert expired["bars_to_resolution"] == 3
    assert expired["exit_price"] == 100.1
    assert expired["complete"] is True

    pending = evaluate_market_outcome(record(), quiet[:1])
    assert pending["final_result"] == "pending"
    assert pending["complete"] is False
    assert pending["resolution_at"] is None


def test_short_outcome_tracks_target_hits_and_directional_excursions():
    bars = [candle("2026-11-20T10:05:00", 100, 101, 95.8, 96.0)]

    result = evaluate_market_outcome(record("SHORT"), bars)

    assert result["first_hit"] == "TP1"
    assert result["tp1_hit"] is True
    assert result["tp2_hit"] is True
    assert result["final_result"] == "win"
    assert result["exit_price"] == 96.0
    assert result["mfe_r"] == 2.1
    assert result["mae_r"] == 0.5


def test_trigger_candle_is_excluded_and_legacy_record_starts_after_setup_candle():
    old_record = record()
    old_record["trigger_confirmed_at"] = None
    bars = [
        candle("2026-11-20T10:00:00", 100, 105, 97, 102),
        candle("2026-11-20T10:05:00", 100, 100.5, 99.5, 100),
        candle("2026-11-20T10:10:00", 100, 104.5, 99.5, 104),
    ]

    result = evaluate_market_outcome(old_record, bars, horizon_bars=2)

    assert result["bars_observed"] == 2
    assert result["first_hit"] == "TP1"
    assert result["final_result"] == "win"


def test_market_outcome_rejects_invalid_prices_horizon_and_costs():
    with pytest.raises(ValueError, match="directionally valid"):
        evaluate_market_outcome({**record(), "stop_loss": 101}, [])
    with pytest.raises(ValueError, match="horizon_bars"):
        evaluate_market_outcome(record(), [], horizon_bars=0)
    with pytest.raises(ValueError, match="fee_pct"):
        evaluate_market_outcome(record(), [], fee_pct=-0.1)
    with pytest.raises(ValueError, match="tp2_r_multiple"):
        evaluate_market_outcome(record(tp2_price=None), [], tp2_r_multiple=1.5)


def test_unclosed_five_minute_candle_is_not_used():
    bars = [candle("2026-11-20T10:05:00", 100, 104.5, 99.5, 104)]
    result = evaluate_market_outcome(
        record(), bars, as_of="2026-11-20T10:07:00+00:00"
    )
    assert result["bars_observed"] == 0
    assert result["final_result"] == "pending"
    assert result["complete"] is False
