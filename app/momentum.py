"""Independent completed-5m ORB and MEX setup detectors.

Both detectors return setup intents only. SignalEngine performs fresh L1
quote confirmation, extension/risk validation, durable deduplication and
Telegram delivery. No forming candle or order API is used here.
"""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import math
import numpy as np
import pandas as pd
from .risk import LIVE_TARGET_R_MULTIPLE
from .indicators import adx, atr, ema, rsi, vwap
from .candle_quality import directional_candle_quality

SESSIONS = {
    'asia': ('Asia/Tokyo', '09:00', 120),
    'london': ('Europe/London', '08:00', 120),
    'new_york': ('America/New_York', '09:30', 120),
}

def _session_window(stamp, session):
    zone_name, open_time, active_minutes = session
    zone = ZoneInfo(zone_name)
    local = stamp.astimezone(zone)
    start = datetime.combine(local.date(), time.fromisoformat(open_time), tzinfo=zone)
    range_end = start + timedelta(minutes=5)
    return start, range_end, range_end + timedelta(minutes=int(active_minutes))

def _opening_range_candle(frame, start_utc):
    matches = frame[frame.timestamp == pd.Timestamp(start_utc)]
    return matches if len(matches) == 1 else None

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


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _frame(frame, diagnostics=None):
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns) or len(frame) < 60:
        return _reject(diagnostics, "history", "insufficient_or_missing_ohlcv", {"bars": len(frame) if isinstance(frame, pd.DataFrame) else 0})
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    if f[["timestamp", *numeric_columns]].isna().any().any():
        return _reject(diagnostics, "data_validation", "invalid_timestamp_or_ohlcv")
    if not np.isfinite(f[numeric_columns].to_numpy(dtype=float)).all():
        return _reject(diagnostics, "data_validation", "non_finite_ohlcv")
    f = f.sort_values("timestamp").reset_index(drop=True)
    if f.timestamp.duplicated().any() or (f.volume < 0).any():
        return _reject(diagnostics, "data_validation", "duplicate_timestamps_or_negative_volume")
    if (f.high < f[["open", "close", "low"]].max(axis=1)).any():
        return _reject(diagnostics, "data_validation", "invalid_ohlc_structure")
    if (f.low > f[["open", "close", "high"]].min(axis=1)).any():
        return _reject(diagnostics, "data_validation", "invalid_ohlc_structure")
    return f


def _indicators(f):
    f = f.copy()
    f["ema9"] = ema(f.close, 9)
    f["ema21"] = ema(f.close, 21)
    f["ema20"] = ema(f.close, 20)
    f["ema50"] = ema(f.close, 50)
    f["rsi"] = rsi(f.close, 14)
    f["atr"] = atr(f, 14)
    f["adx"], f["plus_di"], f["minus_di"] = adx(f, 14)
    f["vwap"] = np.nan
    session_date = f.timestamp.iloc[-1].date()
    mask = f.timestamp.dt.date == session_date
    f.loc[mask, "vwap"] = vwap(f.loc[mask])
    f["volume_base"] = f.volume.shift(1).rolling(20, min_periods=15).mean()
    f["rvol"] = f.volume / f.volume_base.replace(0, np.nan)
    f["atr_base"] = f.atr.shift(1).rolling(20, min_periods=15).mean()
    f["atr_expansion"] = f.atr / f.atr_base.replace(0, np.nan)
    f["roc3"] = f.close.pct_change(3) * 100.0
    fast = f.close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = f.close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_hist"] = macd_line - signal
    f["bb_mid"] = f.close.rolling(20, min_periods=20).mean()
    f["bb_std"] = f.close.rolling(20, min_periods=20).std(ddof=0)
    f["bb_width"] = (4 * f.bb_std) / f.bb_mid.replace(0, np.nan)
    return f


def _risk(f, side, reference, atr_value):
    recent = f.iloc[-5:]
    buffer = 0.15 * atr_value
    stop = float(recent.low.min()) - buffer if side == "LONG" else float(recent.high.max()) + buffer
    risk = (reference - stop) if side == "LONG" else (stop - reference)
    risk_atr = risk / atr_value if atr_value > 0 else float("inf")
    if not math.isfinite(risk) or risk <= 0 or risk_atr > 3.0:
        return None
    target = reference + LIVE_TARGET_R_MULTIPLE * risk if side == "LONG" else reference - LIVE_TARGET_R_MULTIPLE * risk
    return {"stop_loss": stop, "take_profit_1": target, "risk_distance_atr": risk_atr}


def _base_result(f, side, strategy, reason, trigger_price, reference, atr_value, extra, diagnostics=None):
    risk = _risk(f, side, reference, atr_value)
    if risk is None:
        return _reject(diagnostics, "risk", "structural_risk_out_of_bounds", {"atr": atr_value, "reference": reference})
    candle = f.iloc[-1]
    return {
        "strategy": strategy, "side": side,
        "entry_reference": float(reference), "trigger_price": float(trigger_price),
        "stop_loss": risk["stop_loss"], "take_profit_1": risk["take_profit_1"],
        "atr": float(atr_value), "risk_distance_atr": round(risk["risk_distance_atr"], 4),
        "reason": reason, "candle_time": candle.timestamp,
        "indicators": extra,
    }


def evaluate_orb(frame, *, adx_min=None, min_rvol=1.10,
                 enabled_sessions=("asia", "london", "new_york"), sessions=None,
                 diagnostics=None, prepared=False):
    """Detect a completed 5m close beyond a valid session opening range."""
    if prepared:
        f = frame
    else:
        f = _frame(frame, diagnostics)
        if f is None:
            return None
        f = _indicators(f)
    c = f.iloc[-1]
    # ADX is intentionally not an ORB gate; DI direction remains required.
    vals = [c.atr, c.plus_di, c.minus_di, c.ema20, c.ema50, c.rsi, c.vwap, c.rvol]
    if any(pd.isna(x) for x in vals) or c.atr <= 0:
        return _reject(diagnostics, "indicators", "indicator_warmup_or_invalid", {"atr": c.atr, "rvol": c.rvol})
    stamp = c.timestamp.to_pydatetime()
    candidates = []
    session_map = SESSIONS if sessions is None else sessions
    for name in enabled_sessions:
        if name not in session_map:
            continue
        zone_name, open_time, active_minutes = session_map[name]
        start, range_end, expiry = _session_window(stamp, session_map[name])
        if not (range_end <= stamp < expiry):
            continue
        candidates.append((start, name, zone_name, range_end, expiry))
    if not candidates:
        return _reject(diagnostics, "session", "outside_enabled_orb_session", {"enabled_sessions": list(enabled_sessions)})
    start, name, zone_name, range_end, expiry = max(candidates, key=lambda x: x[0])
    start_utc = start.astimezone(timezone.utc)
    opening = _opening_range_candle(f, start_utc)
    if opening is None:
        return _reject(diagnostics, "opening_range", "opening_range_candle_missing", {"session": name, "opening_candle_matches": 0 if opening is None else len(opening)})
    rh, rl = float(opening.high.iloc[0]), float(opening.low.iloc[0])
    width_atr = (rh - rl) / float(c.atr)
    if not 0.05 <= width_atr <= 3.0:
        opening_volume = float(opening.volume.iloc[0])
        opening_quality = ("flat_no_trades" if rh == rl and opening_volume <= 0
                           else "flat_price_with_volume" if rh == rl else "non_flat")
        return _reject(diagnostics, "opening_range", "opening_range_width_out_of_bounds", {"session": name, "session_timezone": zone_name, "opening_candle_at": opening.timestamp.iloc[0].isoformat(), "opening_open": float(opening.open.iloc[0]), "opening_high": rh, "opening_low": rl, "opening_close": float(opening.close.iloc[0]), "opening_volume": opening_volume, "opening_source_quality": opening_quality, "current_atr": float(c.atr), "range_width_atr": float(width_atr), "minimum_range_atr": 0.05, "maximum_range_atr": 3.0})
    side = "LONG" if c.close > rh else "SHORT" if c.close < rl else None
    if side is None:
        return _reject(diagnostics, "breakout", "no_close_outside_opening_range", {"close": c.close, "range_high": rh, "range_low": rl})
    edge = rh if side == "LONG" else rl
    buffer = 0.03 * float(c.atr)
    if (side == "LONG" and c.close < edge + buffer) or (side == "SHORT" and c.close > edge - buffer):
        return _reject(diagnostics, "breakout", "breakout_buffer_not_met", {"close": c.close, "edge": edge, "buffer": buffer})
    # Only the first two completed closes beyond the edge are eligible.
    age = 0
    for _, bar in f.iloc[::-1].iterrows():
        if (bar.timestamp.to_pydatetime() < range_end.astimezone(timezone.utc)):
            break
        beyond = bar.close > edge if side == "LONG" else bar.close < edge
        if not beyond:
            break
        age += 1
    if age > 2:
        return _reject(diagnostics, "timing", "breakout_confirmation_too_late", {"breakout_close_age": age})
    previous_close = float(f.close.iloc[-2])
    retest_tolerance = 0.15 * float(c.atr)
    retest_touch = bool(
        (previous_close > edge and float(c.low) <= edge + retest_tolerance)
        if side == "LONG" else
        (previous_close < edge and float(c.high) >= edge - retest_tolerance)
    )
    di_ok = c.plus_di > c.minus_di if side == "LONG" else c.minus_di > c.plus_di
    trend_ok = c.ema20 > c.ema50 if side == "LONG" else c.ema20 < c.ema50
    vwap_ok = c.close > c.vwap if side == "LONG" else c.close < c.vwap
    rsi_ok = c.rsi >= 52 if side == "LONG" else c.rsi <= 48
    if not di_ok or not trend_ok or not vwap_ok or not rsi_ok:
        return _reject(diagnostics, "trend_confirmation", "di_ema20_50_vwap_or_rsi_filter", {"plus_di": c.plus_di, "minus_di": c.minus_di, "ema20": c.ema20, "ema50": c.ema50, "vwap": c.vwap, "rsi": c.rsi, "di_ok": bool(di_ok), "trend_ok": bool(trend_ok), "vwap_ok": bool(vwap_ok), "rsi_ok": bool(rsi_ok)})
    session_bars = f[(f.timestamp >= pd.Timestamp(start_utc)) & (f.timestamp < c.timestamp)]
    if len(session_bars) < 2:
        return _reject(diagnostics, "session_volume", "insufficient_session_volume_history", {"session": name, "session_open_utc": start_utc.isoformat(), "current_candle_at": c.timestamp.isoformat(), "session_bars": len(session_bars), "required_session_bars": 2})
    session_base = float(session_bars.volume.mean())
    session_rvol = float(c.volume) / session_base if session_base > 0 else 0.0
    if float(c.rvol) < min_rvol or session_rvol < min_rvol:
        return _reject(diagnostics, "volume", "relative_volume_below_minimum", {"rvol": c.rvol, "session_rvol": session_rvol, "min_rvol": min_rvol})
    body_atr = abs(float(c.close) - float(c.open)) / float(c.atr)
    if body_atr < 0.20:
        return _reject(diagnostics, "candle_body_strength", "breakout_body_below_minimum", {"body_atr": body_atr, "minimum_body_atr": 0.20})
    quality = directional_candle_quality(
        c.open, c.high, c.low, c.close, side,
        min_body_ratio=0.35, min_close_location=0.65,
        max_opposing_wick_ratio=0.35,
    )
    if not quality["passed"]:
        return _reject(diagnostics, "candle_quality", "directional_candle_quality_failed", {"quality": quality})
    extension = abs(float(c.close) - edge) / float(c.atr)
    if extension > 0.80:
        return _reject(diagnostics, "entry_quality", "breakout_extension_too_large", {"extension_atr": extension, "max_extension_atr": 0.80})
    # ORB trigger is anchored to the range edge, not the breakout candle high/low.
    trigger = edge + buffer if side == "LONG" else edge - buffer
    entry_path = "ORB_RETEST" if retest_touch else "ORB_BREAKOUT"
    reason = (f"{name.upper()} {entry_path} close beyond opening range; "
              f"RVOL={float(c.rvol):.2f}, session_RVOL={session_rvol:.2f}, "
              f"extension={extension:.2f} ATR")
    return _base_result(
        f, side, "ORB", reason, trigger, edge, float(c.atr),
        {"session": name, "session_timezone": zone_name,
         "session_open_utc": start_utc.isoformat(), "range_high": rh,
         "range_low": rl, "range_width_atr": round(width_atr, 4),
         "breakout_extension_atr": round(extension, 4),
         "entry_path": entry_path, "retest_touch": retest_touch,
         "ema20": round(float(c.ema20), 8), "ema50": round(float(c.ema50), 8),
         "body_atr": round(body_atr, 4),
         "rvol": round(float(c.rvol), 4), "session_rvol": round(session_rvol, 4),
         "candle_quality": quality}, diagnostics,
    )

