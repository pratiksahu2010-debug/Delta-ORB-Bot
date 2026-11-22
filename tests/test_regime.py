import pandas as pd

from app.regime import classify_regime


def test_accepts_directional_trend_with_small_pullback():
    close = pd.Series([100, 101, 102, 103, 102.7, 103.4, 104.0, 103.8, 105, 106, 107], dtype=float)
    ema21 = pd.Series([99.5] * 11, dtype=float)
    result = classify_regime(close, ema21, "LONG")
    assert result is not None
    assert result["regime"] == "TRENDING_UP"
    assert result["efficiency_ratio"] >= 0.20
    assert result["trend_consistency"] >= 4 / 6


def test_rejects_sideways_whipsaw():
    close = pd.Series([100, 101, 99, 101, 99, 101, 99, 101, 99, 101, 100], dtype=float)
    ema21 = pd.Series([100] * 11, dtype=float)
    assert classify_regime(close, ema21, "LONG") is None
    assert classify_regime(close, ema21, "SHORT") is None


def test_rejects_wrong_trend_side_consistency():
    close = pd.Series([100, 99, 98, 97, 98, 97, 96, 95, 94, 93, 92], dtype=float)
    ema21 = pd.Series([100] * 11, dtype=float)
    assert classify_regime(close, ema21, "LONG") is None


def test_rejects_non_finite_inputs():
    close = pd.Series([100.0] * 10 + [float("nan")])
    ema21 = pd.Series([99.0] * 11)
    assert classify_regime(close, ema21, "LONG") is None


def test_rejected_regime_exposes_efficiency_and_consistency_metrics():
    close = pd.Series([100, 101, 99, 101, 99, 101, 99, 101, 99, 101, 100], dtype=float)
    ema21 = pd.Series([100] * 11, dtype=float)
    diagnostics = {}
    assert classify_regime(close, ema21, "LONG", diagnostics=diagnostics) is None
    assert diagnostics["reason"] == "regime_quality_below_minimum"
    values = diagnostics["values"]
    assert values["side"] == "LONG"
    assert values["efficiency_ratio"] < values["min_efficiency"]
    assert values["trend_consistency"] < values["min_consistency"]
    assert values["lookback"] == 10
    assert values["consistency_bars"] == 6
