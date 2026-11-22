import numpy as np
import pandas as pd
import pytest

from app.tpr_rb import evaluate_rb, evaluate_tpr


def flat_frame(count=80):
    stamps = pd.date_range("2026-11-15T18:00:00Z", periods=count, freq="5min")
    return pd.DataFrame({
        "timestamp": stamps,
        "open": np.full(count, 100.0),
        "high": np.full(count, 100.2),
        "low": np.full(count, 99.8),
        "close": np.full(count, 100.0),
        "volume": np.full(count, 100.0),
    })


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_rb_accepts_clean_range_breakout_in_both_directions(monkeypatch, side):
    import app.tpr_rb as module

    frame = flat_frame()
    if side == "LONG":
        frame.loc[68:78, ["open", "high", "low", "close"]] = [100.0, 101.2, 98.8, 100.0]
        frame.loc[79, ["open", "high", "low", "close", "volume"]] = [101.4, 103.0, 101.3, 102.9, 220.0]
        values = {"ema9": 102.0, "ema21": 101.5, "rsi": 60.0,
                  "plus_di": 28.0, "minus_di": 12.0, "vwap": 101.0}
    else:
        frame.loc[68:78, ["open", "high", "low", "close"]] = [100.0, 101.2, 98.8, 100.0]
        frame.loc[79, ["open", "high", "low", "close", "volume"]] = [98.6, 98.7, 97.0, 97.1, 220.0]
        values = {"ema9": 98.0, "ema21": 98.5, "rsi": 40.0,
                  "plus_di": 12.0, "minus_di": 28.0, "vwap": 99.0}

    def indicators(data):
        out = data.copy()
        for key, value in {
            **values, "atr": 2.0, "rvol": 2.0, "atr_expansion": 1.2,
            "adx": 20.0, "roc3": 0.8,
        }.items():
            out[key] = value
        return out

    monkeypatch.setattr(module, "_indicators", indicators)
    result = evaluate_rb(frame, min_rvol=1.35)
    assert result is not None
    assert result["strategy"] == "RB"
    assert result["side"] == side
    assert result["indicators"]["setup"] == "RANGE_BREAKOUT"


def test_rb_rejects_flat_market_without_range_breakout():
    assert evaluate_rb(flat_frame()) is None


def test_rb_rejects_invalid_ohlc():
    frame = flat_frame()
    frame.loc[79, "high"] = 98.0
    diagnostics = {}
    assert evaluate_rb(frame, diagnostics=diagnostics) is None
    assert diagnostics["stage"] == "data_validation"


def test_tpr_wraps_existing_pullback_detector_and_labels_strategy(monkeypatch):
    import app.tpr_rb as module

    expected = {
        "side": "LONG", "entry_reference": 101.0, "stop_loss": 99.0,
        "take_profit_1": 104.0, "atr": 2.0, "reason": "test",
        "ema_spread_atr": 0.2, "ema21_slope_atr_3bars": 0.1,
        "efficiency_ratio": 0.4, "trend_consistency": 0.8,
    }
    monkeypatch.setattr(module, "evaluate_tpr_setup", lambda *_a, **_k: dict(expected))
    result = evaluate_tpr(flat_frame())
    assert result["strategy"] == "TPR"
    assert result["trigger_price"] == pytest.approx(100.2)
    assert result["indicators"]["setup"] == "TREND_PULLBACK_REJECTION"

def test_direct_detector_defaults_match_final_balanced_profile():
    import inspect
    from app.tpr_rb import evaluate_tpr, evaluate_rb

    assert inspect.signature(evaluate_tpr).parameters["adx_min"].default == 17.0
    assert inspect.signature(evaluate_rb).parameters["min_rvol"].default == 1.45


def test_rb_range_rejection_exposes_source_range_and_atr(monkeypatch):
    import app.tpr_rb as module

    frame = flat_frame()

    def indicators(data):
        out = data.copy()
        for key, value in {
            "atr": 0.01, "rvol": 2.0, "ema9": 100.0, "ema21": 99.5,
            "plus_di": 20.0, "minus_di": 10.0, "rsi": 55.0, "vwap": 99.0,
        }.items():
            out[key] = value
        return out

    monkeypatch.setattr(module, "_indicators", indicators)
    diagnostics = {}
    assert evaluate_rb(frame, diagnostics=diagnostics) is None
    assert diagnostics["reason"] == "range_width_out_of_bounds"
    values = diagnostics["values"]
    assert values["range_high"] == pytest.approx(100.2)
    assert values["range_low"] == pytest.approx(99.8)
    assert values["current_atr"] == pytest.approx(0.01)
    assert values["range_lookback"] == 12
    assert values["range_bar_count"] == 12
