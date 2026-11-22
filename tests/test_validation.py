from datetime import datetime, timezone, timedelta
from app.models import Quote
from app.validation import validate_quote

def test_fresh_quote_passes():
    now = datetime.now(timezone.utc)
    ok, reason = validate_quote(Quote(100, now), now, 3, 100, 2, 0.8)
    assert ok and reason == "ok"

def test_stale_quote_rejected():
    now = datetime.now(timezone.utc)
    ok, reason = validate_quote(Quote(100, now-timedelta(seconds=5)), now, 3, 100, 2, 0.8)
    assert not ok and reason == "stale_or_future_quote"

def test_extended_entry_rejected():
    now = datetime.now(timezone.utc)
    ok, reason = validate_quote(Quote(102, now), now, 3, 100, 2, 0.8)
    assert not ok and reason == "entry_too_extended"


def test_nonfinite_quote_is_rejected():
    now = datetime.now(timezone.utc)
    ok, reason = validate_quote(Quote(float("nan"), now), now, 3, 100, 2, 0.8)
    assert not ok and reason == "non_finite_quote_or_threshold"



def test_confirmation_max_gap_must_cover_confirmation_hold():
    from app.config import Settings

    settings = Settings(
        trigger_confirmation_seconds=0.5,
        trigger_confirmation_max_gap_seconds=0.4,
    )
    try:
        settings.validate_safety()
    except ValueError as exc:
        assert "Invalid strategy thresholds" in str(exc)
    else:
        raise AssertionError("max quote gap shorter than hold duration must be rejected")


def test_confirmation_max_gap_accepts_valid_timing():
    from app.config import Settings

    settings = Settings(
        trigger_confirmation_seconds=0.5,
        trigger_confirmation_max_gap_seconds=1.5,
    )
    settings.validate_safety()



def test_non_finite_runtime_thresholds_are_rejected():
    from app.config import Settings

    fields = (
        "adx_min",
        "orb_min_rvol",
        "rb_min_rvol",
        "mex_min_rvol",
        "mex_min_atr_expansion",
        "mex_live_min_eval_interval_seconds",
        "quote_max_age_seconds",
        "max_entry_extension_atr",
        "trigger_confirmation_seconds",
        "trigger_confirmation_max_gap_seconds",
        "candle_poll_interval_seconds",
    )
    for field in fields:
        for value in (float("nan"), float("inf"), float("-inf")):
            settings = Settings(**{field: value})
            try:
                settings.validate_safety()
            except ValueError as exc:
                assert "Invalid strategy thresholds" in str(exc)
            else:
                raise AssertionError(f"{field} accepted non-finite value {value}")


def test_quote_age_and_entry_extension_must_be_positive():
    from app.config import Settings

    for field in ("quote_max_age_seconds", "max_entry_extension_atr"):
        for value in (0, -0.1):
            settings = Settings(**{field: value})
            try:
                settings.validate_safety()
            except ValueError as exc:
                assert "Invalid strategy thresholds" in str(exc)
            else:
                raise AssertionError(f"{field} accepted non-positive value {value}")
