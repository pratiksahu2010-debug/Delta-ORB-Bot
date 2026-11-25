import numpy as np
import pandas as pd

import app.advika_choice as advika
from app.config import Settings


def _bars(last_timestamp="2026-11-24T04:55:00Z"):
    stamps = pd.date_range(end=pd.Timestamp(last_timestamp), periods=240, freq="5min")
    close = np.linspace(100.0, 101.0, 240)
    return pd.DataFrame({
        "timestamp": stamps, "open": close - 0.1, "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": np.full(240, 100.0),
    })


def _mock_indicators(monkeypatch, side="LONG"):
    def prepare(frame):
        data = frame.copy()
        for name, value in {
            "ema20": 109.2, "ema50": 105.0, "ema200": 100.0,
            "rsi14": 45.0, "atr14": 1.0, "adx14": 30.0,
            "macd": -0.2, "macd_signal": -0.1, "macd_hist": -0.1,
            "volume_avg20": 100.0,
        }.items():
            data[name] = value
        if side == "LONG":
            data.loc[data.index[-2], "rsi14"] = 45.0
            data.loc[data.index[-1], ["open", "high", "low", "close", "volume"]] = [109, 111, 104, 110, 160]
            data.loc[data.index[-1], ["macd", "macd_signal", "macd_hist", "rsi14"]] = [-0.05, -0.1, 0.05, 47]
        else:
            data.loc[data.index[-1], ["ema20", "ema50", "ema200"]] = [92, 95, 100]
            data.loc[data.index[-2], ["macd", "macd_signal", "macd_hist", "rsi14"]] = [0.2, 0.1, 0.1, 55]
            data.loc[data.index[-1], ["open", "high", "low", "close", "volume"]] = [91, 96, 89, 90, 160]
            data.loc[data.index[-1], ["macd", "macd_signal", "macd_hist", "rsi14"]] = [0.05, 0.1, -0.05, 53]
        return data
    monkeypatch.setattr(advika, "_with_indicators", prepare)


def test_advika_long_uses_closed_candle_and_1_5r_target(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    result = advika.evaluate_advika_choice(_bars(), symbol="BTCUSD")
    assert result["side"] == "LONG"
    assert result["entry_reference"] == 110
    assert result["stop_loss"] == 108
    assert result["take_profit_1"] == 113
    assert result["risk_percent_cap"] == 1.0
    assert result["partial_exit_percent"] == 50
    assert result["quality_score"] >= 75
    assert result["relative_volume"] == 1.6


def test_advika_short_is_the_reverse_setup(monkeypatch):
    _mock_indicators(monkeypatch, "SHORT")
    result = advika.evaluate_advika_choice(_bars(), symbol="BTCUSD")
    assert result["side"] == "SHORT"
    assert result["entry_reference"] == 90
    assert result["stop_loss"] == 92
    assert result["take_profit_1"] == 87


def test_advika_alt_long_fails_closed_without_same_candle_btc_filter(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    diagnostics = {}
    result = advika.evaluate_advika_choice(
        _bars(), symbol="ETHUSD", btc_frame=None, diagnostics=diagnostics,
    )
    assert result is None
    assert diagnostics["reason"] == "btc_ema50_filter_failed_for_alt_long"


def test_advika_funding_blackout_uses_ist_candle_close():
    # 23:55 UTC + 5 minutes is 05:30 IST.
    diagnostics = {}
    result = advika.evaluate_advika_choice(
        _bars("2026-11-23T23:55:00Z"), symbol="BTCUSD", diagnostics=diagnostics,
    )
    assert result is None
    assert diagnostics["reason"] == "funding_time_blackout"


def test_advika_dead_hours_are_configurable_in_ist(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    diagnostics = {}
    result = advika.evaluate_advika_choice(
        _bars(), symbol="BTCUSD", dead_hours_ist="10:00-11:00",
        diagnostics=diagnostics,
    )
    assert result is None
    assert diagnostics["reason"] == "configured_dead_hour"


def test_advika_requires_240_contiguous_closed_candles():
    diagnostics = {}
    result = advika.evaluate_advika_choice(
        _bars().iloc[-120:], symbol="BTCUSD", diagnostics=diagnostics,
    )
    assert result is None
    assert diagnostics["reason"] == "insufficient_history"


def test_advika_module_is_independent_and_does_not_change_all_mode():
    all_settings = Settings(bot_module="ALL")
    assert all_settings.enabled_modules == ["ORB", "MEX", "ADVIKA"]
    assert all_settings.module_enabled("ADVIKA")

    advika_settings = Settings(bot_module="ADVIKA")
    assert advika_settings.enabled_modules == ["ADVIKA"]
    assert advika_settings.strategy_enabled("ADVIKA")
    assert not advika_settings.strategy_enabled("ORB")
    assert not advika_settings.strategy_enabled("MEX")


def test_advika_dead_hour_configuration_is_validated():
    settings = Settings(bot_module="ADVIKA", advika_dead_hours_ist="00:00-05:00,12:00-13:00")
    settings.validate_safety()


def test_advika_can_be_disabled_without_disabling_other_modules():
    settings = Settings(bot_module="ORB", advika_enabled=False)
    assert settings.enabled_modules == ["ORB"]
    assert not settings.strategy_enabled("ADVIKA")
    assert settings.strategy_enabled("ORB")


def test_advika_waits_for_partial_target_before_trailing(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    result = advika.evaluate_advika_exit(
        {"side": "LONG", "stop_loss": 90, "take_profit_1": 120, "trailing_stop": 90},
        _bars(),
    )
    assert result["action"] == "HOLD"
    assert result["reason"] == "waiting_for_1_5r_partial_target"


def test_advika_trailing_stop_is_monotonic_after_partial(monkeypatch):
    def prepare(frame):
        data = frame.copy()
        data["ema20"] = 95.0
        data["macd"] = -0.05
        data["macd_signal"] = -0.1
        data["macd_hist"] = np.linspace(0.1, 0.4, len(data))
        return data
    monkeypatch.setattr(advika, "_with_indicators", prepare)
    result = advika.evaluate_advika_exit(
        {"side": "LONG", "stop_loss": 90, "take_profit_1": 120,
         "trailing_stop": 100, "partial_taken": True},
        _bars(),
    )
    assert result["action"] == "UPDATE_TRAILING_STOP"
    assert result["trailing_stop"] == 100



def test_advika_adx_is_scored_not_a_mandatory_gate(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    original = advika._with_indicators

    def low_adx(frame):
        data = original(frame)
        data["adx14"] = 12.0
        return data

    monkeypatch.setattr(advika, "_with_indicators", low_adx)
    diagnostics = {}
    result = advika.evaluate_advika_choice(_bars(), symbol="BTCUSD", diagnostics=diagnostics)
    assert result is not None
    assert result["quality_score_breakdown"]["adx_strength"] == 0


def test_advika_relative_volume_minimum_is_1_2x(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    original = advika._with_indicators

    def low_volume(frame):
        data = original(frame)
        data.loc[data.index[-1], "volume"] = 119.0
        return data

    monkeypatch.setattr(advika, "_with_indicators", low_volume)
    diagnostics = {}
    result = advika.evaluate_advika_choice(_bars(), symbol="BTCUSD", diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "signal_volume_below_minimum"


def test_advika_minimum_quality_score_is_enforced(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    diagnostics = {}
    result = advika.evaluate_advika_choice(
        _bars(), symbol="BTCUSD", min_score=100, diagnostics=diagnostics,
    )
    assert result is None
    assert diagnostics["reason"] == "advika_score_below_minimum"


def test_advika_settings_default_to_recommended_thresholds():
    settings = Settings(bot_module="ADVIKA")
    assert settings.advika_min_relative_volume == 1.2
    assert settings.advika_min_score == 75


def test_advika_exit_honors_active_trailing_stop_after_partial(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    result = advika.evaluate_advika_exit(
        {"side": "LONG", "stop_loss": 90, "take_profit_1": 120,
         "trailing_stop": 105, "partial_taken": True},
        _bars(),
    )
    assert result["action"] == "EXIT_REMAINDER"
    assert result["reason"] == "trailing_stop_touched"


def test_advika_partial_target_starts_ema20_trailing_stop(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    result = advika.evaluate_advika_exit(
        {"side": "LONG", "stop_loss": 90, "take_profit_1": 110.5,
         "trailing_stop": 90, "partial_taken": False},
        _bars(),
    )
    assert result["action"] == "TAKE_PARTIAL_50"
    assert result["trailing_stop"] == 109.2


def test_advika_reports_noncontiguous_history_separately():
    bars = _bars()
    bars.loc[bars.index[-1], "timestamp"] += pd.Timedelta(minutes=5)
    diagnostics = {}
    assert advika.evaluate_advika_choice(
        bars, symbol="BTCUSD", diagnostics=diagnostics,
    ) is None
    assert diagnostics["reason"] == "noncontiguous_5m_history"


def test_advika_reports_invalid_ohlc_relationship_separately():
    bars = _bars()
    bars.loc[bars.index[-1], "high"] = bars.loc[bars.index[-1], "low"] - 1
    diagnostics = {}
    assert advika.evaluate_advika_choice(
        bars, symbol="BTCUSD", diagnostics=diagnostics,
    ) is None
    assert diagnostics["reason"] == "invalid_ohlc_relationship"


def test_advika_ema_filters_are_scored_not_hard_gates(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    original = advika._with_indicators

    def countertrend(frame):
        data = original(frame)
        data.loc[data.index[-1], ["ema50", "ema200"]] = [111.0, 112.0]
        return data

    monkeypatch.setattr(advika, "_with_indicators", countertrend)
    diagnostics = {}
    result = advika.evaluate_advika_choice(
        _bars(), symbol="BTCUSD", min_score=50, diagnostics=diagnostics,
    )
    assert result is not None
    assert result["side"] == "LONG"
    assert result["quality_score_breakdown"]["ema200_alignment"] == 0


def test_advika_macd_crossover_does_not_require_macd_below_zero(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    original = advika._with_indicators

    def positive_cross(frame):
        data = original(frame)
        data.loc[data.index[-2], ["macd", "macd_signal", "macd_hist"]] = [0.05, 0.10, -0.05]
        data.loc[data.index[-1], ["macd", "macd_signal", "macd_hist"]] = [0.15, 0.10, 0.05]
        return data

    monkeypatch.setattr(advika, "_with_indicators", positive_cross)
    result = advika.evaluate_advika_choice(_bars(), symbol="BTCUSD", min_score=50)
    assert result is not None
    assert result["setup_type"] == "MACD_CROSSOVER"


def test_advika_long_pullback_continuation_setup(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    original = advika._with_indicators

    def continuation(frame):
        data = original(frame)
        data.loc[data.index[-3], ["macd", "macd_signal", "macd_hist"]] = [0.2, 0.1, 0.10]
        data.loc[data.index[-2], ["macd", "macd_signal", "macd_hist"]] = [0.2, 0.1, 0.06]
        data.loc[data.index[-1], ["macd", "macd_signal", "macd_hist"]] = [0.24, 0.1, 0.14]
        data.loc[data.index[-2], ["open", "high", "low", "close"]] = [101.2, 101.5, 100.4, 100.8]
        return data

    monkeypatch.setattr(advika, "_with_indicators", continuation)
    result = advika.evaluate_advika_choice(_bars(), symbol="BTCUSD", min_score=50)
    assert result is not None
    assert result["side"] == "LONG"
    assert result["setup_type"] == "MACD_PULLBACK_CONTINUATION"


def test_advika_short_pullback_continuation_setup(monkeypatch):
    _mock_indicators(monkeypatch, "SHORT")
    original = advika._with_indicators

    def continuation(frame):
        data = original(frame)
        data.loc[data.index[-3], ["macd", "macd_signal", "macd_hist"]] = [-0.2, -0.1, -0.10]
        data.loc[data.index[-2], ["macd", "macd_signal", "macd_hist"]] = [-0.2, -0.1, -0.06]
        data.loc[data.index[-1], ["macd", "macd_signal", "macd_hist"]] = [-0.24, -0.1, -0.14]
        data.loc[data.index[-2], ["open", "high", "low", "close"]] = [100.8, 101.5, 100.4, 101.2]
        return data

    monkeypatch.setattr(advika, "_with_indicators", continuation)
    result = advika.evaluate_advika_choice(_bars(), symbol="BTCUSD", min_score=50)
    assert result is not None
    assert result["side"] == "SHORT"
    assert result["setup_type"] == "MACD_PULLBACK_CONTINUATION"


def test_advika_accepts_exact_rvol_threshold(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    original = advika._with_indicators

    def threshold_volume(frame):
        data = original(frame)
        data.loc[data.index[-1], "volume"] = 120.0
        return data

    monkeypatch.setattr(advika, "_with_indicators", threshold_volume)
    result = advika.evaluate_advika_choice(
        _bars(), symbol="BTCUSD", min_score=50,
    )
    assert result is not None
    assert result["relative_volume"] == 1.2


def test_advika_ema50_is_quality_score_not_entry_gate(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    original = advika._with_indicators

    def distant_ema50(frame):
        data = original(frame)
        data.loc[data.index[-1], "ema50"] = 130.0
        return data

    monkeypatch.setattr(advika, "_with_indicators", distant_ema50)
    result = advika.evaluate_advika_choice(
        _bars(), symbol="BTCUSD", min_score=50,
    )
    assert result is not None
    assert result["quality_score_breakdown"]["ema50_pullback_quality"] == 0


def test_advika_missing_adx_is_zero_score_not_a_rejection(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    original = advika._with_indicators

    def missing_adx(frame):
        data = original(frame)
        data["adx14"] = np.nan
        return data

    monkeypatch.setattr(advika, "_with_indicators", missing_adx)
    result = advika.evaluate_advika_choice(_bars(), symbol="BTCUSD")
    assert result is not None
    assert result["quality_score_breakdown"]["adx_strength"] == 0


def test_advika_missing_ema_filters_only_removes_quality_points(monkeypatch):
    _mock_indicators(monkeypatch, "LONG")
    original = advika._with_indicators

    def missing_emas(frame):
        data = original(frame)
        data["ema50"] = np.nan
        data["ema200"] = np.nan
        return data

    monkeypatch.setattr(advika, "_with_indicators", missing_emas)
    result = advika.evaluate_advika_choice(
        _bars(), symbol="BTCUSD", min_score=50,
    )
    assert result is not None
    assert result["quality_score_breakdown"]["ema200_alignment"] == 0
    assert result["quality_score_breakdown"]["ema50_pullback_quality"] == 0
