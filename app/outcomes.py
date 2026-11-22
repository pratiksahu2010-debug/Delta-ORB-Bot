"""Conservative, restart-safe forward-market outcome measurements.

Uses completed 5-minute OHLC candles only. Outcomes are research observations,
not fills or instructions to execute trades. If stop and target levels both
trade in one candle and the open does not establish their order, stop is
assumed first and the ambiguity is recorded explicitly.
"""
from datetime import datetime, timedelta, timezone
import math


BAR_SECONDS = 300


def _utc(value) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _value(candle, name):
    return float(getattr(candle, name) if hasattr(candle, name) else candle[name])


def _rounded(value):
    return round(float(value), 6) if value is not None and math.isfinite(float(value)) else None


def evaluate_market_outcome(
    record: dict,
    candles,
    horizon_bars: int | None = None,
    tp2_r_multiple: float = 2.0,
    fee_pct: float | None = None,
    slippage_pct: float | None = None,
    as_of=None,
) -> dict:
    """Measure TP1/TP2/SL path, excursions and estimated net return.

    TP1 and TP2 are observation levels only. TP1 defaults to the confirmed
    signal's stored target (normally 1.5R); TP2 defaults to the configured
    R multiple (2R by default). The live alert and its target are unchanged.

    After TP1, observation continues until TP2, SL or the configured horizon.
    This is a full-position path measurement, not a partial-exit simulator.
    """
    horizon = horizon_bars if horizon_bars is not None else int(record.get("horizon_bars") or 12)
    if not isinstance(horizon, int) or not 1 <= horizon <= 288:
        raise ValueError("horizon_bars must be an integer from 1 to 288")
    if not math.isfinite(float(tp2_r_multiple)) or float(tp2_r_multiple) <= 1.5:
        raise ValueError("tp2_r_multiple must be greater than the TP1 1.5R level")

    side = str(record.get("side") or "").upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("market outcome requires LONG or SHORT side")
    entry = float(record.get("entry_price") or 0)
    stop = float(record.get("stop_loss") or 0)
    if not all(math.isfinite(v) and v > 0 for v in (entry, stop)):
        raise ValueError("entry and stop must be finite positive prices")
    risk = entry - stop if side == "LONG" else stop - entry
    if risk <= 0:
        raise ValueError("signal prices are not directionally valid")

    tp1 = float(
        record.get("tp1_price")
        or record.get("target_price")
        or (record.get("signal") or {}).get("take_profit_1")
        or (entry + 1.5 * risk if side == "LONG" else entry - 1.5 * risk)
    )
    tp2_value = record.get("tp2_price") or (record.get("signal") or {}).get("take_profit_2")
    tp2 = float(tp2_value) if tp2_value is not None else (
        entry + float(tp2_r_multiple) * risk
        if side == "LONG" else entry - float(tp2_r_multiple) * risk
    )
    if side == "LONG" and not (entry < tp1 < tp2):
        raise ValueError("LONG TP1 and TP2 must be above entry and TP2 above TP1")
    if side == "SHORT" and not (entry > tp1 > tp2):
        raise ValueError("SHORT TP1 and TP2 must be below entry and TP2 below TP1")

    signal = record.get("signal") or {}
    atr_value = record.get("atr") or signal.get("atr") or signal.get("atr_value")
    try:
        atr_value = float(atr_value) if atr_value is not None else None
        if atr_value is not None and (not math.isfinite(atr_value) or atr_value <= 0):
            atr_value = None
    except (TypeError, ValueError):
        atr_value = None

    fees = float(record.get("fee_pct", 0.0) if fee_pct is None else fee_pct)
    slippage = float(record.get("slippage_pct", 0.0) if slippage_pct is None else slippage_pct)
    if not all(math.isfinite(v) and v >= 0 for v in (fees, slippage)):
        raise ValueError("fee_pct and slippage_pct must be finite non-negative round-trip percentages")
    cost_pct = fees + slippage

    trigger_value = record.get("trigger_confirmed_at")
    if trigger_value:
        trigger_at = _utc(trigger_value)
        first_full_bar_epoch = math.ceil(trigger_at.timestamp() / BAR_SECONDS) * BAR_SECONDS
    else:
        trigger_at = _utc(record.get("candle_time"))
        first_full_bar_epoch = math.ceil(
            (trigger_at.timestamp() + BAR_SECONDS) / BAR_SECONDS
        ) * BAR_SECONDS

    as_of_time = _utc(as_of) if as_of is not None else datetime.now(timezone.utc)
    eligible = []
    seen = set()
    for candle in candles or []:
        stamp = _utc(
            getattr(candle, "timestamp") if hasattr(candle, "timestamp") else candle["timestamp"]
        )
        epoch = int(stamp.timestamp())
        if epoch < first_full_bar_epoch or epoch % BAR_SECONDS != 0 or epoch in seen:
            continue
        if stamp + timedelta(seconds=BAR_SECONDS) > as_of_time:
            continue
        seen.add(epoch)
        high, low, opn, close = (
            _value(candle, key) for key in ("high", "low", "open", "close")
        )
        if not all(math.isfinite(v) and v > 0 for v in (high, low, opn, close)):
            continue
        if high < max(opn, close, low) or low > min(opn, close, high):
            continue
        eligible.append((stamp, opn, high, low, close))
    eligible.sort(key=lambda row: row[0])
    eligible = eligible[:horizon]

    mfe_abs = 0.0
    mae_abs = 0.0
    tp1_hit = False
    tp2_hit = False
    stop_hit = False
    same_candle_ambiguous = False
    first_hit = None
    first_hit_at = None
    bars_to_first_hit = None
    final_result = "pending"
    resolution_at = None
    bars_to_resolution = None
    exit_price = None
    exit_reason = None
    bars_observed = 0
    last_close = None

    for stamp, opn, high, low, close in eligible:
        bars_observed += 1
        bar_closed_at = stamp + timedelta(seconds=BAR_SECONDS)
        last_close = close
        if side == "LONG":
            mfe_abs = max(mfe_abs, high - entry)
            mae_abs = max(mae_abs, entry - low)
            this_stop = low <= stop
            this_tp1 = high >= tp1
            this_tp2 = high >= tp2
            open_stop = opn <= stop
            open_tp1 = opn >= tp1
            open_tp2 = opn >= tp2
        else:
            mfe_abs = max(mfe_abs, entry - low)
            mae_abs = max(mae_abs, high - entry)
            this_stop = high >= stop
            this_tp1 = low <= tp1
            this_tp2 = low <= tp2
            open_stop = opn >= stop
            open_tp1 = opn <= tp1
            open_tp2 = opn <= tp2

        tp1_hit = tp1_hit or this_tp1
        tp2_hit = tp2_hit or this_tp2
        stop_hit = stop_hit or this_stop

        # An opening gap establishes the first event. Otherwise, when both
        # protective stop and a target trade in one OHLC bar, ordering is
        # unknowable; use stop-first and retain an explicit ambiguity flag.
        if this_stop and (this_tp1 or this_tp2) and not (open_stop or open_tp1 or open_tp2):
            same_candle_ambiguous = True
            if first_hit is None:
                first_hit = "SL"
                first_hit_at = bar_closed_at.isoformat()
                bars_to_first_hit = bars_observed
            final_result = "loss"
            exit_price = stop
            exit_reason = "same_candle_stop_first"
        elif open_stop or (this_stop and not (this_tp1 or this_tp2)):
            if first_hit is None:
                first_hit = "SL"
                first_hit_at = bar_closed_at.isoformat()
                bars_to_first_hit = bars_observed
            final_result = "loss"
            exit_price = stop
            exit_reason = "stop"
        elif open_tp2:
            if first_hit is None:
                first_hit = "TP2"
                first_hit_at = bar_closed_at.isoformat()
                bars_to_first_hit = bars_observed
            final_result = "win"
            exit_price = tp2
            exit_reason = "tp2_gap"
        elif open_tp1:
            if first_hit is None:
                first_hit = "TP1"
                first_hit_at = bar_closed_at.isoformat()
                bars_to_first_hit = bars_observed
            if this_stop and this_tp2:
                same_candle_ambiguous = True
                final_result = "loss"
                exit_price = stop
                exit_reason = "same_candle_stop_first_after_tp1"
            elif this_stop:
                final_result = "loss"
                exit_price = stop
                exit_reason = "stop_after_open_tp1"
            elif this_tp2:
                final_result = "win"
                exit_price = tp2
                exit_reason = "tp2"
        elif this_stop:
            if first_hit is None:
                first_hit = "SL"
                first_hit_at = bar_closed_at.isoformat()
                bars_to_first_hit = bars_observed
            final_result = "loss"
            exit_price = stop
            exit_reason = "stop"
        elif this_tp2:
            if first_hit is None:
                first_hit = "TP1" if this_tp1 else "TP2"
                first_hit_at = bar_closed_at.isoformat()
                bars_to_first_hit = bars_observed
            final_result = "win"
            exit_price = tp2
            exit_reason = "tp2"
        elif this_tp1 and first_hit is None:
            first_hit = "TP1"
            first_hit_at = bar_closed_at.isoformat()
            bars_to_first_hit = bars_observed

        if final_result in {"win", "loss"}:
            resolution_at = bar_closed_at.isoformat()
            bars_to_resolution = bars_observed
            break

    complete = final_result in {"win", "loss"} or bars_observed >= horizon
    if complete and final_result == "pending":
        final_result = "expired"
        exit_price = last_close
        exit_reason = "horizon_close"
        if eligible:
            resolution_at = (eligible[-1][0] + timedelta(seconds=BAR_SECONDS)).isoformat()
            bars_to_resolution = bars_observed

    elapsed_seconds = None
    if resolution_at:
        elapsed_seconds = max(0.0, (_utc(resolution_at) - trigger_at).total_seconds())

    gross_return_pct = None
    net_return_pct = None
    if exit_price is not None:
        direction = 1.0 if side == "LONG" else -1.0
        gross_return_pct = direction * (float(exit_price) - entry) / entry * 100.0
        net_return_pct = gross_return_pct - cost_pct

    mfe_pct = mfe_abs / entry * 100.0
    mae_pct = mae_abs / entry * 100.0
    result = {
        "method": "completed_5m_ohlc_path",
        "horizon_bars": horizon,
        "bars_observed": bars_observed,
        "first_hit": first_hit,
        "first_hit_at": first_hit_at,
        "first_event": first_hit.lower() if first_hit else None,
        "first_event_at": first_hit_at,
        "final_result": final_result,
        "result": final_result,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "stop_hit": stop_hit,
        "same_candle_ambiguous": same_candle_ambiguous,
        "ambiguity_rule": (
            "If stop and any target trade in one candle and the open does not "
            "establish order, record stop-first as the conservative outcome."
        ),
        "ambiguity_reason": (
            "Stop and target both traded in the same OHLC candle without a "
            "determinable intrabar order."
            if same_candle_ambiguous else None
        ),
        "bars_to_first_hit": bars_to_first_hit,
        "bars_to_resolution": bars_to_resolution,
        "elapsed_seconds": _rounded(elapsed_seconds),
        "resolution_at": resolution_at,
        "exit_price": _rounded(exit_price),
        "exit_reason": exit_reason,
        "mfe_r": _rounded(mfe_abs / risk),
        "mae_r": _rounded(mae_abs / risk),
        "mfe_pct": _rounded(mfe_pct),
        "mae_pct": _rounded(mae_pct),
        "mfe_atr": _rounded(mfe_abs / atr_value) if atr_value else None,
        "mae_atr": _rounded(mae_abs / atr_value) if atr_value else None,
        "gross_return_pct": _rounded(gross_return_pct),
        "fee_pct": _rounded(fees),
        "slippage_pct": _rounded(slippage),
        "cost_pct": _rounded(cost_pct),
        "net_return_pct": _rounded(net_return_pct),
        "complete": complete,
        "limitations": [
            "Completed 5m OHLC bars cannot establish intrabar order when the open does not resolve a stop/target collision.",
            "Event timestamps are the completed candle close, not an inferred intrabar execution timestamp.",
            "TP1 is recorded as a path event; the measurement continues to TP2, SL or expiry and does not simulate partial fills.",
            "Fees and slippage are configurable round-trip estimates; funding, spread, liquidity and actual fills are not modeled.",
            "The trigger-containing candle is excluded unless it begins exactly at the trigger timestamp.",
        ],
    }
    return result
