from app.config import Settings


def test_combined_mode_remains_backward_compatible():
    settings = Settings(bot_module="ALL")
    assert settings.enabled_modules == ["ORB", "MEX"]
    assert settings.strategy_enabled("ORB")
    assert settings.strategy_enabled("RB")
    assert settings.strategy_enabled("TPR")
    assert settings.strategy_enabled("MEX")


def test_mex_mode_enables_only_mex():
    settings = Settings(bot_module="MEX")
    assert settings.enabled_modules == ["MEX"]
    assert settings.module_enabled("MEX")
    assert not settings.module_enabled("ORB")
    assert settings.strategy_enabled("MEX")
    assert not settings.strategy_enabled("RB")
    assert not settings.strategy_enabled("TPR")
    assert not settings.strategy_enabled("ORB")


def test_orb_mode_enables_orb_family_only():
    settings = Settings(bot_module="ORB")
    assert settings.enabled_modules == ["ORB"]
    assert settings.module_enabled("ORB")
    assert not settings.module_enabled("MEX")
    assert settings.strategy_enabled("ORB")
    assert settings.strategy_enabled("RB")
    assert settings.strategy_enabled("TPR")
    assert not settings.strategy_enabled("MEX")


def test_invalid_bot_module_fails_safety_validation():
    import pytest
    settings = Settings(bot_module="BOTH_BUT_NOT_ALL")
    with pytest.raises(ValueError, match="BOT_MODULE"):
        settings.validate_safety()
