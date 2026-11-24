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


def _prepare(frame: pd.DataFrame) -> pd.DataFrame | None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns) or len(frame) < HISTORY_BARS:
        return None
    data = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data.isna().any().any():
        return None
    prices = data[["open", "high", "low", "close"]]
    numeric = data[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (prices <= 0).any().any() or (data.volume < 0).any():
        return None
    if data.timestamp.duplicated().any():
        return None
    data = data.sort_values("timestamp").reset_index(drop=True)
    if ((data.high < data[["open", "close", "low"]].max(axis=1)).any()
            or (data.low > data[["open", "close", "high"]].min(axis=1)).any()):
        return None
    stamps = data.timestamp.tolist()
    if any((b - a).total_seconds() != BAR_SECONDS for a, b in zip(stamps, stamps[1:])):
        return None
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
    dead_hours_ist: str = "", diagnostics: dict | None = None,
) -> dict | None:
    """Evaluate one symbol's newest fully closed 5m candle."""
    data = _prepare(frame)
    if data is None:
        return _reject(diagnostics, "history_or_ohlcv_invalid", {"required_bars": HISTORY_BARS})
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

    fields = ("ema20", "ema50", "ema200", "rsi14", "atr14", "adx14",
              "macd", "macd_signal", "macd_hist", "volume_avg20")
    if any(pd.isna(current[k]) for k in fields) or any(
        pd.isna(previous[k]) for k in ("rsi14", "macd", "macd_signal", "macd_hist")
    ):
        return _reject(diagnostics, "indicator_warmup_or_invalid")
    if current.adx14 <= 25:
        return _reject(diagnostics, "adx_not_above_25", {"adx": current.adx14})
    if current.volume_avg20 <= 0 or current.volume < 1.5 * current.volume_avg20:
        return _reject(diagnostics, "signal_volume_below_1_5x", {
            "volume": current.volume, "average_volume_20": current.volume_avg20,
        })
    if current.atr14 <= 0 or not math.isfinite(float(current.atr14)):
        return _reject(diagnostics, "invalid_atr")

    long_ok = (
        current.close > current.ema200 and current.ema50 > current.ema200
        and current.low <= current.ema50 < current.close
        and previous.macd <= previous.macd_signal and current.macd > current.macd_signal
        and current.macd < 0 and previous.macd_hist <= 0 < current.macd_hist
        and 40 <= previous.rsi14 <= 50 and current.rsi14 > previous.rsi14
        and current.close > current.open
    )
    short_ok = (
        current.close < current.ema200 and current.ema50 < current.ema200
        and current.high >= current.ema50 > current.close
        and previous.macd >= previous.macd_signal and current.macd < current.macd_signal
        and current.macd > 0 and previous.macd_hist >= 0 > current.macd_hist
        and 50 <= previous.rsi14 <= 60 and current.rsi14 < previous.rsi14
        and current.close < current.open
    )
    side = "LONG" if long_ok else "SHORT" if short_ok else None
    if side is None:
        return _reject(diagnostics, "entry_conditions_not_met", {
            "adx": current.adx14, "ema50": current.ema50, "ema200": current.ema200,
            "rsi": current.rsi14, "macd": current.macd, "macd_signal": current.macd_signal,
            "macd_hist": current.macd_hist, "relative_volume": current.volume / current.volume_avg20,
        })
    if side == "LONG" and symbol.upper() != "BTCUSD" and not _btc_above_ema50(btc_frame, current.timestamp):
        return _reject(diagnostics, "btc_ema50_filter_failed_for_alt_long")

    entry, atr_value = float(current.close), float(current.atr14)
    stop = max(entry - 2 * atr_value, float(current.low)) if side == "LONG" else min(entry + 2 * atr_value, float(current.high))
    risk = entry - stop if side == "LONG" else stop - entry
    if not math.isfinite(risk) or risk <= 0:
        return _reject(diagnostics, "invalid_pullback_stop_distance")
    target = entry + 1.5 * risk if side == "LONG" else entry - 1.5 * risk
    result = {
        "side": side, "entry_reference": entry, "trigger_price": entry,
        "stop_loss": stop, "take_profit_1": target, "atr": atr_value,
        "risk_distance": risk, "risk_reward": 1.5, "risk_percent_cap": 1.0,
        "position_size_formula": "equity * 0.01 / abs(entry - stop)",
        "partial_exit_percent": 50, "trailing_stop": "EMA20 on each completed 5m candle",
        "early_exit": "close remainder if MACD crosses back or histogram magnitude shrinks for 3 consecutive bars",
        "reason": f"Advika Choice {side}: EMA200 trend, EMA50 pullback, MACD(12,26,9) cross, RSI(14) turn, ADX>25, volume>=1.5x",
        "setup_time": current.timestamp.isoformat(), "candle_close_time": close_time.isoformat(),
        "ema20": float(current.ema20), "macd": float(current.macd),
        "macd_signal": float(current.macd_signal), "macd_histogram": float(current.macd_hist),
        "adx": float(current.adx14), "rsi": float(current.rsi14),
        "relative_volume": float(current.volume / current.volume_avg20),
        "pullback_extreme": float(current.low if side == "LONG" else current.high),
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
    # A completed candle touching both stop and target is treated conservatively
    # as a stop-out because intrabar event order cannot be reconstructed.
    stop_hit = (
        stop > 0 and (current.low <= stop if side == "LONG" else current.high >= stop)
    )
    if stop_hit:
        return {"action": "EXIT_REMAINDER", "reason": "stop_loss_touched",
                "candle_time": candle_time}
    target_hit = (
        target > 0 and (current.high >= target if side == "LONG" else current.low <= target)
    )
    if target_hit and not partial_taken:
        return {"action": "TAKE_PARTIAL_50", "reason": "target_1_5r_reached",
                "candle_time": candle_time}

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

    ema_stop = float(current.ema20)
    old_trailing = float(position.get("trailing_stop", stop) or stop)
    trailing = max(old_trailing, ema_stop) if side == "LONG" else min(old_trailing, ema_stop)
    return {"action": "UPDATE_TRAILING_STOP", "trailing_stop": trailing,
            "candle_time": candle_time}
