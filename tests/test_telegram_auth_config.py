from app.config import Settings


def test_private_chat_defaults_feedback_allowlist_to_chat_owner():
    settings = Settings(_env_file=None, telegram_chat_id="123456789")
    assert settings.telegram_feedback_user_ids == ["123456789"]


def test_group_chat_requires_explicit_feedback_user_allowlist():
    settings = Settings(_env_file=None, telegram_chat_id="-100123456")
    assert settings.telegram_feedback_user_ids == []


def test_group_chat_feedback_allowlist_parses_comma_separated_user_ids():
    settings = Settings(
        _env_file=None,
        telegram_chat_id="-100123456",
        telegram_allowed_user_ids="42, 84,42",
    )
    assert settings.telegram_feedback_user_ids == ["42", "84"]


def test_outcome_measurement_settings_have_safe_research_only_defaults():
    settings = Settings(_env_file=None)
    assert settings.outcome_horizon_bars == 12
    assert settings.outcome_tp2_r_multiple == 2.0
    assert settings.outcome_fee_pct == 0.0
    assert settings.outcome_slippage_pct == 0.0


def test_outcome_measurement_settings_reject_invalid_ranges():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, outcome_horizon_bars=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, outcome_tp2_r_multiple=1.5)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, outcome_fee_pct=-0.01)
