"""Directional candle-quality checks for momentum entries.

The thresholds are deliberately isolated from the existing pullback strategy.
"""
import math


def directional_candle_quality(open_price, high, low, close, direction,
                               *, min_body_ratio=0.20,
                               min_close_location=0.60,
                               max_opposing_wick_ratio=0.45):
    try:
        o, h, l, c = (float(x) for x in (open_price, high, low, close))
        body_floor = float(min_body_ratio)
        close_floor = float(min_close_location)
        wick_ceiling = float(max_opposing_wick_ratio)
    except (TypeError, ValueError, OverflowError):
        return {"passed": False, "reason": "invalid_candle_quality_input",
                "failed_checks": ["invalid_candle_quality_input"]}
    if (direction not in ("LONG", "SHORT")
            or not all(math.isfinite(x) for x in (o, h, l, c, body_floor, close_floor, wick_ceiling))
            or min(o, h, l, c) <= 0 or h < l or h < max(o, c) or l > min(o, c)
            or not 0 <= body_floor <= 1 or not 0 <= close_floor <= 1
            or not 0 <= wick_ceiling <= 1):
        return {"passed": False, "reason": "invalid_candle_quality_input",
                "failed_checks": ["invalid_candle_quality_input"]}
    span = h - l
    if span <= max(abs(c) * 1e-12, 1e-12):
        return {"passed": False, "reason": "zero_range", "failed_checks": ["zero_range"]}
    body_ratio = abs(c - o) / span
    close_location = (c - l) / span if direction == "LONG" else (h - c) / span
    opposing_wick = ((h - max(o, c)) / span if direction == "LONG"
                     else (min(o, c) - l) / span)
    aligned = c > o if direction == "LONG" else c < o
    failed = []
    if not aligned:
        failed.append("body_not_directional")
    if body_ratio + 1e-9 < body_floor:
        failed.append("body_too_small")
    if close_location + 1e-9 < close_floor:
        failed.append("close_not_near_directional_extreme")
    if opposing_wick > wick_ceiling + 1e-9:
        failed.append("opposing_wick_too_large")
    return {
        "passed": not failed, "reason": "accepted" if not failed else "poor_directional_candle",
        "failed_checks": failed, "body_aligned": bool(aligned),
        "body_ratio": round(body_ratio, 4),
        "close_location": round(close_location, 4),
        "opposing_wick_ratio": round(opposing_wick, 4),
        "minimum_body_ratio": body_floor,
        "minimum_close_location": close_floor,
        "maximum_opposing_wick_ratio": wick_ceiling,
    }
