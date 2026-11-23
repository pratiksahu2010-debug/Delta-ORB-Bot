from datetime import datetime, timezone

import pandas as pd
import pytest

from app.config import Settings
from app.momentum import SESSIONS, _opening_range_candle, _session_window


def test_default_orb_sessions_are_configured_and_validate():
    settings = Settings()
    settings.validate_safety()
    assert settings.orb_session_config == SESSIONS


def test_orb_sessions_reject_invalid_timezone_and_window():
    settings = Settings(orb_sessions='{"asia":["Not/A_Zone","09:00",120]}')
    with pytest.raises(ValueError, match="ORB_SESSIONS"):
        settings.validate_safety()


def test_new_york_session_open_tracks_daylight_saving_time():
    winter = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)
    summer = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)

    winter_start, winter_end, winter_expiry = _session_window(winter, SESSIONS["new_york"])
    summer_start, summer_end, summer_expiry = _session_window(summer, SESSIONS["new_york"])

    assert winter_start.isoformat() == "2026-01-15T09:30:00-05:00"
    assert summer_start.isoformat() == "2026-07-15T09:30:00-04:00"
    assert (winter_end - winter_start).total_seconds() == 300
    assert (summer_expiry - summer_end).total_seconds() == 7200


def test_london_session_open_tracks_daylight_saving_time():
    winter = datetime(2026, 1, 15, 8, 0, tzinfo=timezone.utc)
    summer = datetime(2026, 7, 15, 7, 0, tzinfo=timezone.utc)

    winter_start, _, _ = _session_window(winter, SESSIONS["london"])
    summer_start, _, _ = _session_window(summer, SESSIONS["london"])

    assert winter_start.utcoffset().total_seconds() == 0
    assert summer_start.utcoffset().total_seconds() == 3600


def test_missing_or_duplicate_opening_range_candle_is_not_accepted():
    start = pd.Timestamp("2026-11-23T00:00:00Z")
    frame = pd.DataFrame({"timestamp": [start + pd.Timedelta(minutes=5)]})
    assert _opening_range_candle(frame, start) is None

    duplicate = pd.DataFrame({"timestamp": [start, start]})
    assert _opening_range_candle(duplicate, start) is None

    exact = pd.DataFrame({"timestamp": [start]})
    assert len(_opening_range_candle(exact, start)) == 1


def _orb_test_frame():
    stamps = pd.date_range("2026-11-22T19:10:00Z", periods=61, freq="5min")
    return pd.DataFrame({
        "timestamp": stamps,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 100.0,
    })


def _patch_ready_orb_indicators(monkeypatch):
    import app.momentum as module

    def indicators(frame):
        result = frame.copy()
        for name, value in {
            "atr": 1.0, "plus_di": 20.0, "minus_di": 10.0,
            "ema20": 101.0, "ema50": 100.0, "rsi": 55.0,
            "vwap": 99.0, "rvol": 2.0,
        }.items():
            result[name] = value
        return result

    monkeypatch.setattr(module, "_indicators", indicators)


def test_evaluate_orb_uses_defined_session_map(monkeypatch):
    from app.momentum import evaluate_orb

    _patch_ready_orb_indicators(monkeypatch)
    diagnostics = {}
    assert evaluate_orb(_orb_test_frame(), diagnostics=diagnostics) is None
    assert diagnostics["stage"] == "breakout"
    assert diagnostics["reason"] == "no_close_outside_opening_range"


def test_evaluate_orb_rejects_missing_opening_candle(monkeypatch):
    from app.momentum import evaluate_orb

    _patch_ready_orb_indicators(monkeypatch)
    frame = _orb_test_frame().drop(index=58).reset_index(drop=True)
    diagnostics = {}
    assert evaluate_orb(frame, diagnostics=diagnostics) is None
    assert diagnostics["stage"] == "opening_range"
    assert diagnostics["reason"] == "opening_range_candle_missing"
