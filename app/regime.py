"""Lightweight 5m market-regime classification for pullback setup quality.

This is a setup-quality filter, not a direction generator. It deliberately
uses only completed 5m candles and never uses 15m context as a veto.
"""
import numpy as np
import pandas as pd


def classify_regime(
    close: pd.Series,
    ema21: pd.Series,
    side: str,
    lookback: int = 10,
    consistency_bars: int = 6,
    min_efficiency: float = 0.20,
    min_consistency: float = 4 / 6,
    diagnostics: dict | None = None,
) -> dict | None:
    """Return regime metrics when the tape is directional enough for a pullback.

    Efficiency ratio = absolute net displacement / sum of absolute bar moves.
    Consistency measures the share of recent closes holding on the trend side
    of EMA21. A small countertrend pullback is allowed; persistent chop is not.
    """
    if side not in {"LONG", "SHORT"}:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({"reason": "invalid_side", "values": {"side": side}})
        return None
    if len(close) < max(lookback + 1, consistency_bars):
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({"reason": "insufficient_history", "values": {
                "bars": len(close), "required_bars": max(lookback + 1, consistency_bars),
                "lookback": lookback, "consistency_bars": consistency_bars,
            }})
        return None

    closes = pd.to_numeric(close, errors="coerce").astype(float)
    averages = pd.to_numeric(ema21, errors="coerce").astype(float)
    window = closes.iloc[-(lookback + 1):]
    avg_window = averages.iloc[-consistency_bars:]
    if window.isna().any() or avg_window.isna().any():
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({"reason": "invalid_inputs", "values": {"side": side}})
        return None
    values = window.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(avg_window.to_numpy(dtype=float)).all():
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({"reason": "non_finite_inputs", "values": {"side": side}})
        return None

    path_length = float(np.abs(np.diff(values)).sum())
    net_move = abs(float(values[-1] - values[0]))
    efficiency = net_move / path_length if path_length > 0 else 0.0

    recent_close = closes.iloc[-consistency_bars:]
    if side == "LONG":
        consistency = float((recent_close > avg_window).mean())
    else:
        consistency = float((recent_close < avg_window).mean())

    if efficiency < min_efficiency or consistency < min_consistency:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({
                "reason": "regime_quality_below_minimum",
                "values": {
                    "side": side, "efficiency_ratio": efficiency,
                    "min_efficiency": min_efficiency,
                    "trend_consistency": consistency,
                    "min_consistency": min_consistency,
                    "lookback": lookback, "consistency_bars": consistency_bars,
                    "path_length": path_length, "net_move": net_move,
                },
            })
        return None

    return {
        "regime": "TRENDING_UP" if side == "LONG" else "TRENDING_DOWN",
        "efficiency_ratio": efficiency,
        "trend_consistency": consistency,
    }
