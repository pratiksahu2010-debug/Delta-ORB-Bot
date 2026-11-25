"""Advika Choice: independent, completed-5m signal strategy."""
from __future__ import annotations

import math
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .indicators import adx, atr, ema, rsi

IST = ZoneInfo("Asia/Kolkata")
FUNDING_TIMES_IST = (time(5, 30), time(13, 30), time(21, 30))
HISTORY_BARS = 240
PULLBACK_LOOKBACK_BARS = 5
PULLBACK_SCORE_LOOKBACK_BARS = 3
STOP_BUFFER_ATR = 0.1
MIN_STOP_ATR = 0.5
MAX_STOP_ATR = 2.0
BAR_SECONDS = 300


def parse_dead_hours(value: str | None) -> tuple[tuple[time, time], ...]:
    """Parse comma-separated IST intervals, e.g. '00:00-05:00,12:00-13:00'."""
    intervals = []
    if not str(value or "").strip():
        return ()
    for item in str(value).split(","):
        parts = item.strip().split("-")
        if len(parts) != 2:
            raise ValueError("ADVIKA_DEAD_HOURS_IST must contain HH:MM-HH:MM ranges")
        try:
            start, end = (time.fromisoformat(part.strip()) for part in parts)
        except ValueError:
            raise ValueError("ADVIKA_DEAD_HOURS_IST contains an invalid HH:MM time") from None
        if start == end:
            raise ValueError("Advika dead-hour interval cannot have identical start and end")
        intervals.append((start, end))
    return tuple(intervals)


def _in_interval(current: time, start: time, end: time) -> bool:
    return start <= current < end if start < end else current >= start or current < end


def _reject(diagnostics, reason, values=None):
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"reason": reason})
        if values:
            diagnostics["values"] = {
                k: round(float(v), 8) if isinstance(v, (int, float)) and math.isfinite(float(v)) else v
                for k, v in values.items()
            }
    return None


def _prepare(frame: pd.DataFrame, diagnostics: dict | None = None) -> pd.DataFrame | None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}

    def reject(reason: str, values: dict | None = None):
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({"reason": reason})
            if values:
                diagnostics["values"] = values
        return None

    if not isinstance(frame, pd.DataFrame):
        return reject("invalid_candle_frame")
    missing = sorted(required - set(frame.columns))
    if missing:
        return reject("missing_ohlcv_columns", {"missing": ",".join(missing)})
    if len(frame) < HISTORY_BARS:
        return reject("insufficient_history", {
            "bars_available": len(frame), "required_bars": HISTORY_BARS,
        })
    data = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if data.timestamp.isna().any():
        return reject("invalid_candle_timestamp")
    for col in ("open", "high", "low", "close", "volume"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[["open", "high", "low", "close", "volume"]].isna().any().any():
        return reject("invalid_numeric_ohlcv")
    prices = data[["open", "high", "low", "close"]]
    numeric = data[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        return reject("non_finite_ohlcv")
    if (prices <= 0).any().any() or (data.volume < 0).any():
        return reject("invalid_price_or_volume")
    if data.timestamp.duplicated().any():
        return reject("duplicate_candle_timestamp")
    data = data.sort_values("timestamp").reset_index(drop=True)
    if ((data.high < data[["open", "close", "low"]].max(axis=1)).any()
            or (data.low > data[["open", "close", "high"]].min(axis=1)).any()):
        return reject("invalid_ohlc_relationship")
    stamps = data.timestamp.tolist()
    if any((b - a).total_seconds() != BAR_SECONDS for a, b in zip(stamps, stamps[1:])):
        return reject("noncontiguous_5m_history")
    return data


def _with_indicators(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["ema20"] = ema(out.close, 20)
    out["ema50"] = ema(out.close, 50)
    out["ema200"] = ema(out.close, 200)
    out["rsi14"] = rsi(out.close, 14)
    out["atr14"] = atr(out, 14)
    out["adx14"], _, _ = adx(out, 14)
    fast = out.close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = out.close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = fast - slow
    out["macd_signal"] = out.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out.macd - out.macd_signal
    out["volume_avg20"] = out.volume.shift(1).rolling(20, min_periods=20).mean()
    return out


def _funding_blackout(close_time: datetime) -> bool:
    local = close_time.astimezone(IST)
    current = datetime.combine(local.date(), local.time().replace(tzinfo=None))
    return any(abs((current - datetime.combine(local.date(), anchor)).total_seconds()) <= 300
               for anchor in FUNDING_TIMES_IST)


def _btc_above_ema50(frame: pd.DataFrame | None, candle_time: pd.Timestamp) -> bool:
    btc = _prepare(frame) if frame is not None else None
    if btc is None or btc.timestamp.iloc[-1] != candle_time:
        return False
    value = ema(btc.close, 50).iloc[-1]
    return bool(pd.notna(value) and btc.close.iloc[-1] > value)


def evaluate_advika_choice(
    frame: pd.DataFrame, *, symbol: str, btc_frame: pd.DataFrame | None = None,
    dead_hours_ist: str = "", min_relative_volume: float = 1.2,
    min_score: int = 75, diagnostics: dict | None = None,
) -> dict | None:
    """Evaluate one symbol's newest fully closed 5m candle."""
    data = _prepare(frame, diagnostics)
    if data is None:
        reason = str((diagnostics or {}).get("reason", "history_or_ohlcv_invalid"))
        values = (diagnostics or {}).get("values", {"required_bars": HISTORY_BARS})
        return _reject(diagnostics, reason, values)
    data = _with_indicators(data)
    current, previous = data.iloc[-1], data.iloc[-2]
    close_time = current.timestamp.to_pydatetime() + timedelta(minutes=5)
    try:
        dead_hours = parse_dead_hours(dead_hours_ist)
    except ValueError:
        return _reject(diagnostics, "invalid_dead_hours_configuration")
    local_clock = close_time.astimezone(IST).time().replace(tzinfo=None)
    if _funding_blackout(close_time):
        return _reject(diagnostics, "funding_time_blackout", {"candle_close_ist": close_time.astimezone(IST).isoformat()})
    if any(_in_interval(local_clock, start, end) for start, end in dead_hours):
        return _reject(diagnostics, "configured_dead_hour", {"candle_close_ist": close_time.astimezone(IST).isoformat()})

    # EMA50/EMA200, RSI and ADX are quality inputs, not mandatory gates.
    # Missing optional quality indicators therefore contribute zero points.
    fields = ("ema20", "atr14", "macd", "macd_signal", "macd_hist", "volume_avg20")
    prior = data.iloc[-3]
    if any(pd.isna(current[k]) for k in fields) or any(
        pd.isna(previous[k]) for k in ("macd", "macd_signal", "macd_hist")
    ) or any(pd.isna(prior[k]) for k in ("macd", "macd_signal", "macd_hist")):
        return _reject(diagnostics, "indicator_warmup_or_invalid")
    if not math.isfinite(float(min_relative_volume)) or min_relative_volume < 1.0:
        return _reject(diagnostics, "invalid_min_relative_volume_configuration")
    if not 50 <= int(min_score) <= 100:
        return _reject(diagnostics, "invalid_min_score_configuration")
    if current.volume_avg20 <= 0 or current.volume < min_relative_volume * current.volume_avg20:
        return _reject(diagnostics, "signal_volume_below_minimum", {
            "volume": current.volume, "average_volume_20": current.volume_avg20,
            "minimum_relative_volume": min_relative_volume,
        })
    if current.atr14 <= 0 or not math.isfinite(float(current.atr14)):
        return _reject(diagnostics, "invalid_atr")

    pullback_low = float(data.low.iloc[-PULLBACK_LOOKBACK_BARS:].min())
    pullback_high = float(data.high.iloc[-PULLBACK_LOOKBACK_BARS:].max())
    candle_range = max(float(current.high - current.low), 1e-12)
    long_candle_confirm = (
        current.close > current.open
        and (current.close - current.low) / candle_range >= 0.60
    )
    short_candle_confirm = (
        current.close < current.open
        and (current.high - current.close) / candle_range >= 0.60
    )
    long_cross = (
        previous.macd <= previous.macd_signal
        and current.macd > current.macd_signal
        and current.macd_hist > 0
        and current.macd > previous.macd
        and current.macd_hist > previous.macd_hist
        and long_candle_confirm
    )
    short_cross = (
        previous.macd >= previous.macd_signal
        and current.macd < current.macd_signal
        and current.macd_hist < 0
        and current.macd < previous.macd
        and current.macd_hist < previous.macd_hist
        and short_candle_confirm
    )
    # Earlier continuation entry: require a genuine counter-direction candle,
    # renewed MACD histogram momentum, and a reclaim of its body (not its full
    # wick extreme). This avoids waiting for a late breakout while retaining a
    # completed directional candle as confirmation.
    long_continuation = (
        previous.macd > previous.macd_signal
        and current.macd > current.macd_signal
        and previous.macd_hist <= prior.macd_hist
        and current.macd_hist > previous.macd_hist > 0
        and previous.close < previous.open
        and current.close > current.open
        and current.close > previous.open
    )
    short_continuation = (
        previous.macd < previous.macd_signal
        and current.macd < current.macd_signal
        and previous.macd_hist >= prior.macd_hist
        and current.macd_hist < previous.macd_hist < 0
        and previous.close > previous.open
        and current.close < current.open
        and current.close < previous.open
    )
    side = (
        "LONG" if long_cross or long_continuation
        else "SHORT" if short_cross or short_continuation
        else None
    )
    if side is None:
        return _reject(diagnostics, "mandatory_macd_trigger_failed", {
            "macd": current.macd, "macd_signal": current.macd_signal,
            "macd_hist": current.macd_hist,
            "previous_macd_hist": previous.macd_hist,
            "relative_volume": current.volume / current.volume_avg20,
        })
    entry, atr_value = float(current.close), float(current.atr14)
    # Avoid buying/selling after price has already stretched too far from EMA20.
    entry_extension_atr = (
        max(0.0, (entry - float(current.ema20)) / atr_value)
        if side == "LONG" else
        max(0.0, (float(current.ema20) - entry) / atr_value)
    )
    if entry_extension_atr > 1.2:
        return _reject(diagnostics, "entry_too_extended_from_ema20", {
            "extension_atr": entry_extension_atr, "maximum_extension_atr": 1.2,
        })
    setup_type = (
        "MACD_CROSSOVER" if (long_cross or short_cross)
        else "MACD_PULLBACK_CONTINUATION"
    )
    if side == "LONG" and symbol.upper() != "BTCUSD" and not _btc_above_ema50(btc_frame, current.timestamp):
        return _reject(diagnostics, "btc_ema50_filter_failed_for_alt_long")

    # Anchor risk beyond a genuine counter-direction pullback swing, not on
    # its wick. If no such candle exists, use the 2-ATR fallback. Reject a
    # swing that would require more than 2 ATR rather than silently moving the
    # stop inside the swing; also prevent microscopic (<0.5 ATR) risk.
    recent_pullback = data.iloc[-(PULLBACK_SCORE_LOOKBACK_BARS + 1):-1]
    counter_pullback = (
        recent_pullback.loc[recent_pullback.close < recent_pullback.open]
        if side == "LONG" else
        recent_pullback.loc[recent_pullback.close > recent_pullback.open]
    )
    if side == "LONG":
        swing_extreme = float(counter_pullback.low.min()) if not counter_pullback.empty else entry - 2 * atr_value
        buffered_stop = swing_extreme - STOP_BUFFER_ATR * atr_value
        risk = max(entry - buffered_stop, MIN_STOP_ATR * atr_value)
        if risk > MAX_STOP_ATR * atr_value:
            return _reject(diagnostics, "pullback_stop_exceeds_maximum_atr", {
                "stop_distance_atr": risk / atr_value,
                "maximum_stop_atr": MAX_STOP_ATR,
            })
        stop = entry - risk
    else:
        swing_extreme = float(counter_pullback.high.max()) if not counter_pullback.empty else entry + 2 * atr_value
        buffered_stop = swing_extreme + STOP_BUFFER_ATR * atr_value
        risk = max(buffered_stop - entry, MIN_STOP_ATR * atr_value)
        if risk > MAX_STOP_ATR * atr_value:
            return _reject(diagnostics, "pullback_stop_exceeds_maximum_atr", {
                "stop_distance_atr": risk / atr_value,
                "maximum_stop_atr": MAX_STOP_ATR,
            })
        stop = entry + risk
    if not math.isfinite(risk) or risk <= 0:
        return _reject(diagnostics, "invalid_pullback_stop_distance")
    target = entry + 1.5 * risk if side == "LONG" else entry - 1.5 * risk
    relative_volume = float(current.volume / current.volume_avg20)

    # MACD trigger and RVOL are mandatory. EMA200/EMA50 are quality factors,
    # never hard gates; ADX contributes at most five points and never blocks.
    macd_points = 25
    rvol_points = 8 if relative_volume < 1.3 else 12 if relative_volume < 1.5 else 15
    histogram_points = 10 if (
        (side == "LONG" and current.macd_hist > previous.macd_hist)
        or (side == "SHORT" and current.macd_hist < previous.macd_hist)
    ) else 0
    trend_aligned = pd.notna(current.ema200) and (
        current.close > current.ema200 if side == "LONG"
        else current.close < current.ema200
    )
    trend_points = 15 if trend_aligned else 0
    rsi_available = pd.notna(current.rsi14) and pd.notna(previous.rsi14)

    # EMA50 quality belongs to the actual recent counter-direction pullback,
    # never to the signal candle or an unrelated older candle in a wide window.
    if side == "LONG":
        pullback_distance = (
            float((counter_pullback.low - current.ema50).abs().min())
            if pd.notna(current.ema50) and not counter_pullback.empty else math.inf
        )
        rsi_points = 10 if rsi_available and 40 <= current.rsi14 <= 65 and current.rsi14 > previous.rsi14 else (
            6 if rsi_available and current.rsi14 > previous.rsi14 else 0
        )
        candle_close_position = (current.close - current.low) / max(float(current.high - current.low), 1e-12)
        candle_direction_ok = current.close > current.open
    else:
        pullback_distance = (
            float((counter_pullback.high - current.ema50).abs().min())
            if pd.notna(current.ema50) and not counter_pullback.empty else math.inf
        )
        rsi_points = 10 if rsi_available and 35 <= current.rsi14 <= 60 and current.rsi14 < previous.rsi14 else (
            6 if rsi_available and current.rsi14 < previous.rsi14 else 0
        )
        candle_close_position = (current.high - current.close) / max(float(current.high - current.low), 1e-12)
        candle_direction_ok = current.close < current.open
    pullback_points = 10 if pullback_distance <= 0.5 * atr_value else 6 if pullback_distance <= atr_value else 0
    adx_value = (
        float(current.adx14)
        if pd.notna(current.adx14) and math.isfinite(float(current.adx14)) else 0.0
    )
    adx_points = 0 if adx_value < 15 else 1 if adx_value < 20 else 3 if adx_value <= 25 else 5
    body_ratio = abs(float(current.close - current.open)) / max(float(current.high - current.low), 1e-12)
    candle_points = 5 if candle_direction_ok and body_ratio >= 0.5 and candle_close_position >= 0.75 else (
        3 if candle_direction_ok and body_ratio >= 0.3 and candle_close_position >= 0.6 else 0
    )
    rr_points = 5
    score = int(macd_points + rvol_points + histogram_points + trend_points +
                pullback_points + rsi_points + adx_points + candle_points + rr_points)
    if score < int(min_score):
        return _reject(diagnostics, "advika_score_below_minimum", {
            "score": score, "minimum_score": int(min_score), "adx": adx_value,
            "relative_volume": relative_volume, "setup_type": setup_type,
        })
    result = {
        "side": side, "entry_reference": entry, "trigger_price": entry,
        "stop_loss": stop, "take_profit_1": target, "atr": atr_value,
        "risk_distance": risk, "risk_reward": 1.5, "risk_percent_cap": 1.0,
        "position_size_formula": "equity * 0.01 / abs(entry - stop)",
        "partial_exit_percent": 50, "trailing_stop": "EMA20 on each completed 5m candle",
        "early_exit": "close remainder if MACD crosses back or histogram magnitude shrinks for 3 consecutive bars",
        "setup_type": setup_type,
        "reason": f"Advika Choice {side}: {setup_type} MACD(12,26,9) + relative volume {relative_volume:.2f}x; quality {score}/100",
        "setup_time": current.timestamp.isoformat(), "candle_close_time": close_time.isoformat(),
        "ema20": float(current.ema20), "entry_extension_atr": entry_extension_atr,
        "macd": float(current.macd), "macd_signal": float(current.macd_signal),
        "macd_histogram": float(current.macd_hist),
        "adx": adx_value,
        "rsi": float(current.rsi14) if pd.notna(current.rsi14) else None,
        "relative_volume": relative_volume,
        "quality_score": score, "minimum_quality_score": int(min_score),
        "quality_score_breakdown": {
            "macd_trigger": macd_points, "relative_volume": rvol_points,
            "histogram_expansion": histogram_points, "ema200_alignment": trend_points,
            "ema50_pullback_quality": pullback_points, "rsi_momentum": rsi_points,
            "adx_strength": adx_points, "signal_candle": candle_points,
            "risk_reward": rr_points,
        },
        "pullback_extreme": pullback_low if side == "LONG" else pullback_high,
    }
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"reason": "advika_signal", "side": side})
    return result


def evaluate_advika_exit(position: dict, frame: pd.DataFrame) -> dict | None:
    """Return closed-candle paper-management instructions; never submits orders."""
    data = _prepare(frame)
    if data is None:
        return None
    data = _with_indicators(data)
    current, previous = data.iloc[-1], data.iloc[-2]
    side = str(position.get("side", "")).upper()
    if side not in {"LONG", "SHORT"}:
        return None
    required = ("ema20", "macd", "macd_signal", "macd_hist")
    if any(pd.isna(current[name]) for name in required) or any(
        pd.isna(previous[name]) for name in ("macd", "macd_signal")
    ):
        return None

    candle_time = current.timestamp.isoformat()
    stop = float(position.get("stop_loss", 0) or 0)
    target = float(position.get("take_profit_1", 0) or 0)
    partial_taken = bool(position.get("partial_taken", False))
    trailing = float(position.get("trailing_stop", stop) or stop)
    effective_stop = (
        max(stop, trailing) if side == "LONG" else min(stop, trailing)
    )
    # A completed candle touching both stop and target is treated conservatively
    # as a stop-out because intrabar event order cannot be reconstructed. After
    # the partial target, the more protective EMA20 trail is an active stop.
    stop_hit = (
        effective_stop > 0
        and (current.low <= effective_stop if side == "LONG" else current.high >= effective_stop)
    )
    if stop_hit:
        reason = (
            "trailing_stop_touched"
            if partial_taken and effective_stop != stop else "stop_loss_touched"
        )
        return {"action": "EXIT_REMAINDER", "reason": reason,
                "candle_time": candle_time}
    target_hit = (
        target > 0 and (current.high >= target if side == "LONG" else current.low <= target)
    )
    if target_hit and not partial_taken:
        ema_stop = float(current.ema20)
        initial_trail = max(stop, ema_stop) if side == "LONG" else min(stop, ema_stop)
        return {"action": "TAKE_PARTIAL_50", "reason": "target_1_5r_reached",
                "trailing_stop": initial_trail, "candle_time": candle_time}

    hist = data.macd_hist.iloc[-4:]
    shrinking = len(hist) == 4 and all(
        abs(float(hist.iloc[i])) < abs(float(hist.iloc[i - 1])) for i in range(1, 4)
    )
    crossed_back = (
        previous.macd >= previous.macd_signal and current.macd < current.macd_signal
        if side == "LONG" else
        previous.macd <= previous.macd_signal and current.macd > current.macd_signal
    )
    if shrinking or crossed_back:
        return {"action": "EXIT_REMAINDER",
                "reason": "histogram_shrunk_3_bars" if shrinking else "macd_crossed_back",
                "candle_time": candle_time}

    if not partial_taken:
        return {"action": "HOLD", "reason": "waiting_for_1_5r_partial_target",
                "candle_time": candle_time}
    ema_stop = float(current.ema20)
    old_trailing = float(position.get("trailing_stop", stop) or stop)
    trailing = max(old_trailing, ema_stop) if side == "LONG" else min(old_trailing, ema_stop)
    return {"action": "UPDATE_TRAILING_STOP", "trailing_stop": trailing,
            "candle_time": candle_time}
