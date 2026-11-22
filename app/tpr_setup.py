"""TPR trend-pullback setup evaluator (not a separate NORMAL strategy).

This detector is exclusively owned by the TPR strategy. The live engine
separately waits for a fresh L1 ask/bid to break the rejection candle extreme.
"""
import math

import pandas as pd

from .indicators import ema, rsi, atr, adx, vwap
from .regime import classify_regime
from .risk import LIVE_TARGET_R_MULTIPLE


def _reject(diagnostics, stage: str, reason: str, values=None):
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"stage": stage, "reason": reason})
        if values:
            clean = {}
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    number = float(value)
                    clean[key] = round(number, 6) if math.isfinite(number) else None
                else:
                    clean[key] = value
            diagnostics["values"] = clean
    return None


def evaluate_tpr_setup(frame: pd.DataFrame, adx_min: float = 17.0, diagnostics: dict | None = None) -> dict | None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns) or len(frame) < 60:
        return _reject(diagnostics, "history", "insufficient_or_missing_ohlcv", {"bars": len(frame)})

    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
    if f["timestamp"].isna().any():
        return _reject(diagnostics, "data_validation", "invalid_timestamp")
    f = f.sort_values("timestamp").reset_index(drop=True)
    if f.timestamp.duplicated().any():
        return _reject(diagnostics, "data_validation", "duplicate_timestamps")
    for col in ("open", "high", "low", "close", "volume"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
    prices = f[["open", "high", "low", "close", "volume"]]
    finite_prices = prices.apply(lambda column: column.map(math.isfinite))
    if prices.isna().any().any() or not finite_prices.all().all():
        return _reject(diagnostics, "data_validation", "invalid_ohlcv_values")
    if (f.volume < 0).any() or (f.high < f[["open", "close", "low"]].max(axis=1)).any():
        return _reject(diagnostics, "data_validation", "invalid_ohlc_structure")
    if (f.low > f[["open", "close", "high"]].min(axis=1)).any():
        return _reject(diagnostics, "data_validation", "invalid_ohlc_structure")

    f["ema9"] = ema(f.close, 9)
    f["ema21"] = ema(f.close, 21)
    f["rsi"] = rsi(f.close, 14)
    f["atr"] = atr(f, 14)
    f["adx"], f["plus_di"], f["minus_di"] = adx(f, 14)

    # Session VWAP is calculated only from the latest UTC date.
    f["vwap"] = float("nan")
    latest_session = f["timestamp"].iloc[-1].date()
    session_mask = f["timestamp"].dt.date == latest_session
    f.loc[session_mask, "vwap"] = vwap(f.loc[session_mask])

    c, prev, trend_ref = f.iloc[-1], f.iloc[-2], f.iloc[-4]
    vals = [
        c.ema9, c.ema21, c.rsi, c.atr, c.adx, c.plus_di,
        c.minus_di, c.vwap, trend_ref.ema21,
    ]
    if any(pd.isna(x) for x in vals) or c.atr <= 0:
        return _reject(diagnostics, "indicators", "indicator_warmup_or_invalid", {"atr": c.atr, "adx": c.adx})
    if c.adx < adx_min:
        return _reject(diagnostics, "adx", "adx_below_minimum", {"adx": c.adx, "adx_min": adx_min})

    atr_now = float(c.atr)
    # ADX is a TPR-only gate (default minimum 17). Independent
    # slope/separation and regime checks continue to screen sideways tape.
    ema_spread = abs(float(c.ema9) - float(c.ema21)) / atr_now
    ema21_slope = (float(c.ema21) - float(trend_ref.ema21)) / atr_now
    if ema_spread < 0.05:
        return _reject(diagnostics, "trend_quality", "ema_spread_too_narrow", {"ema_spread_atr": ema_spread, "ema21_slope_atr": ema21_slope})

    regime_side = "LONG" if c.ema9 > c.ema21 else "SHORT"
    regime_diagnostics = {}
    regime = classify_regime(f.close, f.ema21, regime_side, diagnostics=regime_diagnostics)
    if regime is None:
        regime_values = dict(regime_diagnostics.get("values", {}))
        regime_values.update({
            "regime_side": regime_side,
            "ema_spread_atr": ema_spread,
            "ema21_slope_atr": ema21_slope,
            "adx": float(c.adx),
        })
        if regime_diagnostics.get("reason"):
            regime_values["regime_rejection_reason"] = regime_diagnostics["reason"]
        return _reject(diagnostics, "regime", "trend_regime_rejected", regime_values)

    body = abs(float(c.close) - float(c.open))
    candle_range = float(c.high) - float(c.low)
    if candle_range <= 0:
        # A valid flat OHLC bar can occur when the market has no price movement.
        # Keep rejecting it, but retain enough raw evidence to distinguish a
        # genuine flat bar from an upstream candle-construction/data defect.
        return _reject(diagnostics, "candle_quality", "zero_range_candle", {
            "timestamp": c.timestamp.isoformat(),
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "candle_range": candle_range,
            "atr": atr_now,
            "volume": float(c.volume),
            "source_candle_quality": ("flat_no_trades" if float(c.volume) <= 0
                                      else "flat_price_with_volume"),
        })
    lower_wick = min(float(c.open), float(c.close)) - float(c.low)
    upper_wick = float(c.high) - max(float(c.open), float(c.close))
    close_location = (float(c.close) - float(c.low)) / candle_range

    # The setup candle must genuinely test the EMA zone, then reclaim the
    # immediately preceding candle's extreme. The live trigger is still the
    # setup candle high/low and must occur after this candle has completed.
    long_touch = min(
        abs(float(c.low) - float(c.ema9)),
        abs(float(c.low) - float(c.ema21)),
    ) <= 0.25 * atr_now
    short_touch = min(
        abs(float(c.high) - float(c.ema9)),
        abs(float(c.high) - float(c.ema21)),
    ) <= 0.25 * atr_now

    # Ignore tiny doji-like candles, weak wick flickers, and oversized
    # exhaustion bars. The upper bound avoids arming a trigger after a
    # rejection candle has already travelled an unusually large distance.
    meaningful_rejection = (
        0.50 * atr_now <= candle_range <= 2.00 * atr_now
        and 0.12 * atr_now <= body <= 1.25 * atr_now
    )
    long_rejection = (
        meaningful_rejection
        and lower_wick >= max(0.80 * body, 0.10 * atr_now)
        and close_location >= 0.70
        and float(c.close) > float(prev.high)
    )
    short_rejection = (
        meaningful_rejection
        and upper_wick >= max(0.80 * body, 0.10 * atr_now)
        and close_location <= 0.30
        and float(c.close) < float(prev.low)
    )

    # Avoid entering after the rejection candle has already stretched far
    # away from its fast EMA; the separate live quote gate caps trigger chase.
    not_extended_long = (float(c.close) - float(c.ema9)) <= 0.75 * atr_now
    not_extended_short = (float(c.ema9) - float(c.close)) <= 0.75 * atr_now

    long_ok = (
        c.ema9 > c.ema21
        and ema21_slope >= 0.03
        and c.close > c.vwap
        and c.plus_di > c.minus_di
        and 52 <= c.rsi <= 68
        and long_touch and long_rejection and not_extended_long
    )
    short_ok = (
        c.ema9 < c.ema21
        and ema21_slope <= -0.03
        and c.close < c.vwap
        and c.minus_di > c.plus_di
        and 32 <= c.rsi <= 48
        and short_touch and short_rejection and not_extended_short
    )
    if not long_ok and not short_ok:
        return _reject(diagnostics, "entry_setup", "no_valid_pullback_rejection", {
            "ema9": float(c.ema9), "ema21": float(c.ema21), "vwap": float(c.vwap),
            "rsi": float(c.rsi), "adx": float(c.adx),
            "plus_di": float(c.plus_di), "minus_di": float(c.minus_di),
            "ema_spread_atr": ema_spread, "ema21_slope_atr": ema21_slope,
            "long_touch": long_touch, "short_touch": short_touch,
            "long_rejection": long_rejection, "short_rejection": short_rejection,
            "not_extended_long": not_extended_long, "not_extended_short": not_extended_short,
            "long_trend_alignment": bool(c.ema9 > c.ema21 and ema21_slope >= 0.03
                                         and c.close > c.vwap and c.plus_di > c.minus_di),
            "short_trend_alignment": bool(c.ema9 < c.ema21 and ema21_slope <= -0.03
                                          and c.close < c.vwap and c.minus_di > c.plus_di),
            "long_rsi_in_range": bool(52 <= c.rsi <= 68),
            "short_rsi_in_range": bool(32 <= c.rsi <= 48),
            "close": float(c.close), "previous_high": float(prev.high),
            "previous_low": float(prev.low), "candle_range": candle_range,
            "candle_body": body, "atr": atr_now,
        })

    side = "LONG" if long_ok else "SHORT"
    entry_reference = float(c.close)
    recent = f.iloc[-5:]
    buffer = 0.15 * atr_now
    stop = (
        float(recent.low.min()) - buffer
        if side == "LONG"
        else float(recent.high.max()) + buffer
    )
    risk = abs(entry_reference - stop)
    risk_atr = risk / atr_now
    if risk <= 0 or risk_atr > 3.0:
        return _reject(diagnostics, "risk", "structural_risk_out_of_bounds", {"risk_atr": risk_atr})

    target = (
        entry_reference + LIVE_TARGET_R_MULTIPLE * risk
        if side == "LONG"
        else entry_reference - LIVE_TARGET_R_MULTIPLE * risk
    )
    return {
        "side": side,
        "entry_reference": entry_reference,
        "stop_loss": stop,
        "take_profit_1": target,
        "atr": atr_now,
        "risk_distance_atr": risk_atr,
        "ema_spread_atr": ema_spread,
        "ema21_slope_atr_3bars": ema21_slope,
        "market_regime": regime["regime"],
        "efficiency_ratio": regime["efficiency_ratio"],
        "trend_consistency": regime["trend_consistency"],
        "reason": (
            f"5m {regime['regime']} pullback rejection; ER={regime['efficiency_ratio']:.2f}; "
            f"trend consistency={regime['trend_consistency']:.0%}; "
            f"EMA spread={ema_spread:.2f} ATR; EMA21 slope={ema21_slope:.2f} ATR; "
            f"ADX={float(c.adx):.1f}"
        ),
        "candle_time": c.timestamp,
    }
