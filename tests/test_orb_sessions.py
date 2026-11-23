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
