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


def test_orb_health_strategy_count_includes_enabled_setup_types(monkeypatch):
    from types import SimpleNamespace
    import app.main as main

    monkeypatch.setattr(main, "settings", SimpleNamespace(
        enabled_modules=["ORB"],
        module_enabled=lambda module: module == "ORB",
        orb_enabled=True, rb_enabled=True, tpr_enabled=True, mex_enabled=True,
    ))
    assert main._enabled_strategy_count() == 3


def test_mex_health_strategy_count_counts_only_mex(monkeypatch):
    from types import SimpleNamespace
    import app.main as main

    monkeypatch.setattr(main, "settings", SimpleNamespace(
        enabled_modules=["MEX"],
        module_enabled=lambda module: module == "MEX",
        orb_enabled=True, rb_enabled=True, tpr_enabled=True, mex_enabled=True,
    ))
    assert main._enabled_strategy_count() == 1


def test_strategy_enabled_respects_individual_strategy_flags():
    settings = Settings(bot_module="ORB", orb_enabled=True, rb_enabled=False, tpr_enabled=True)
    assert settings.strategy_enabled("ORB") is True
    assert settings.strategy_enabled("RB") is False
    assert settings.strategy_enabled("TPR") is True
    assert settings.strategy_enabled("MEX") is False


def test_strategy_enabled_respects_mex_flag():
    settings = Settings(bot_module="MEX", mex_enabled=False)
    assert settings.strategy_enabled("MEX") is False
