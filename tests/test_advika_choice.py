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
            "ema20": 108.0, "ema50": 105.0, "ema200": 100.0,
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
    assert diagnostics["reason"] == "history_or_ohlcv_invalid"


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
