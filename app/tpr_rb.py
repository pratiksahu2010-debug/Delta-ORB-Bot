"""TPR + Range Breakout setup detectors for the second 100-symbol universe.

TPR reuses the established trend-pullback/rejection model. RB is a distinct
range-expansion setup. Both return setup intents only; SignalEngine owns live
quote confirmation, dedupe, persistence and Telegram delivery.
"""
import math
import pandas as pd

from .momentum import _frame, _indicators, _base_result, _reject
from .tpr_setup import evaluate_tpr_setup


def evaluate_tpr(frame: pd.DataFrame, *, adx_min: float = 17.0, diagnostics=None):
    """Trend Pullback Reversal: completed-candle pullback/rejection setup."""
    result = evaluate_tpr_setup(frame, adx_min=adx_min, diagnostics=diagnostics)
    if result:
        result["strategy"] = "TPR"
        result["indicators"] = {
            "setup": "TREND_PULLBACK_REJECTION",
            "ema_spread_atr": round(float(result.get("ema_spread_atr", 0.0)), 4),
            "ema21_slope_atr_3bars": round(float(result.get("ema21_slope_atr_3bars", 0.0)), 4),
            "efficiency_ratio": round(float(result.get("efficiency_ratio", 0.0)), 4),
            "trend_consistency": round(float(result.get("trend_consistency", 0.0)), 4),
        }
        result["trigger_price"] = (
            float(frame.iloc[-1]["high"]) if result["side"] == "LONG"
            else float(frame.iloc[-1]["low"])
        )
    return result


def evaluate_rb(
    frame: pd.DataFrame,
    *,
    min_rvol: float = 1.45,
    range_lookback: int = 12,
    min_range_atr: float = 0.60,
    max_range_atr: float = 4.50,
    diagnostics=None,
):
    """Range Breakout: break a mature 12-bar range with volume and clean close.

    The breakout candle is excluded from the range calculation. A small ATR
    buffer is used for the quote trigger, while the close must already prove
    acceptance beyond the range. This is not an ORB/session-opening strategy.
    """
    f = _frame(frame, diagnostics)
    if f is None:
        return None
    if range_lookback < 8 or range_lookback > 30:
        return _reject(diagnostics, "configuration", "invalid_range_lookback")
    f = _indicators(f)
    c = f.iloc[-1]
    vals = [c.atr, c.rvol, c.ema9, c.ema21, c.plus_di, c.minus_di, c.rsi, c.vwap]
    if any(pd.isna(x) for x in vals) or float(c.atr) <= 0:
        return _reject(diagnostics, "indicators", "indicator_warmup_or_invalid")
    prior = f.iloc[-(range_lookback + 1):-1]
    range_high = float(prior.high.max())
    range_low = float(prior.low.min())
    range_width_atr = (range_high - range_low) / float(c.atr)
    prior_ranges = prior.high.astype(float) - prior.low.astype(float)
    flat_bars = prior_ranges <= 0
    flat_volume = prior.loc[flat_bars, "volume"].astype(float)
    source_quality = {
        "range_zero_range_bar_count": int(flat_bars.sum()),
        "range_zero_range_zero_volume_count": int((flat_volume <= 0).sum()),
        "range_zero_range_positive_volume_count": int((flat_volume > 0).sum()),
        "range_bar_count": int(len(prior)),
    }
    if not math.isfinite(range_width_atr) or not (min_range_atr <= range_width_atr <= max_range_atr):
        return _reject(diagnostics, "range_quality", "range_width_out_of_bounds", {
            "range_width_atr": range_width_atr,
            "min_range_atr": min_range_atr,
            "max_range_atr": max_range_atr,
            "range_high": range_high,
            "range_low": range_low,
            "current_atr": float(c.atr),
            "range_lookback": range_lookback,
            **source_quality,
        })
    if float(c.rvol) < min_rvol:
        return _reject(diagnostics, "volume", "rvol_below_minimum", {
            "rvol": float(c.rvol), "min_rvol": min_rvol,
        })

    candle_range = float(c.high) - float(c.low)
    body = abs(float(c.close) - float(c.open))
    if candle_range <= 0 or body / float(c.atr) < 0.30:
        return _reject(diagnostics, "candle_quality", "breakout_body_too_weak", {
            "candle_range": candle_range, "candle_body": body,
            "body_atr": body / float(c.atr), "minimum_body_atr": 0.30,
            "atr": float(c.atr), "volume": float(c.volume),
            "source_candle_quality": ("flat_no_trades" if candle_range <= 0 and float(c.volume) <= 0
                                      else "flat_price_with_volume" if candle_range <= 0
                                      else "non_flat"),
        })
    close_location = (float(c.close) - float(c.low)) / candle_range
    long_break = float(c.close) > range_high + 0.03 * float(c.atr)
    short_break = float(c.close) < range_low - 0.03 * float(c.atr)

    # A light 2-of-4 context score avoids a brittle all-indicators checklist.
    if long_break:
        checks = {
            "ema_alignment": bool(c.ema9 > c.ema21),
            "di_pressure": bool(c.plus_di > c.minus_di),
            "rsi_pressure": bool(c.rsi >= 50),
            "vwap_position": bool(c.close >= c.vwap),
        }
        side = "LONG"
        clean_close = close_location >= 0.72
        trigger = range_high + 0.03 * float(c.atr)
    elif short_break:
        checks = {
            "ema_alignment": bool(c.ema9 < c.ema21),
            "di_pressure": bool(c.minus_di > c.plus_di),
            "rsi_pressure": bool(c.rsi <= 50),
            "vwap_position": bool(c.close <= c.vwap),
        }
        side = "SHORT"
        clean_close = close_location <= 0.28
        trigger = range_low - 0.03 * float(c.atr)
    else:
        return _reject(diagnostics, "breakout", "no_close_beyond_range", {
            "range_high": range_high, "range_low": range_low,
            "close": float(c.close),
        })
    context_score = sum(checks.values())
    if not clean_close:
        return _reject(diagnostics, "candle_quality", "close_not_accepted_beyond_range", {
            "close_location": close_location, "side": side,
        })
    if context_score < 2:
        return _reject(diagnostics, "directional_context", "insufficient_directional_context", {
            "context_score": context_score, "checks": checks,
        })

    reference = float(c.close)
    extension = abs(reference - float(c.ema9)) / float(c.atr)
    if extension > 1.50:
        return _reject(diagnostics, "entry_quality", "breakout_too_extended", {
            "extension_atr": extension,
        })
    return _base_result(
        f, side, "RB",
        f"5m range breakout accepted; RVOL={float(c.rvol):.2f}; "
        f"range={range_width_atr:.2f} ATR; context={context_score}/4",
        trigger, reference, float(c.atr),
        {
            "setup": "RANGE_BREAKOUT",
            "range_high": range_high,
            "range_low": range_low,
            "range_width_atr": round(range_width_atr, 4),
            "rvol": round(float(c.rvol), 4),
            "context_score": context_score,
            "context_checks": checks,
            "extension_atr": round(extension, 4),
            "close_location": round(close_location, 4),
        },
        diagnostics,
    )
