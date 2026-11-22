import pandas as pd
import pytest

from app.replay import replay_dataframe, validate_replay_data


def sample_frame(count=60):
    stamps = pd.date_range("2026-11-10T00:00:00Z", periods=count, freq="5min")
    return pd.DataFrame({
        "symbol": ["BTCUSD"] * count,
        "timestamp": stamps,
        "open": [100.0] * count,
        "high": [100.2] * count,
        "low": [99.8] * count,
        "close": [100.0] * count,
        "volume": [100.0] * count,
    })


def test_replay_rejects_retired_normal_strategy():
    with pytest.raises(ValueError, match="Unsupported replay strategies"):
        replay_dataframe(sample_frame(), strategies=("NORMAL",))


def test_replay_validates_duplicate_timestamps():
    frame = sample_frame()
    frame.loc[59, "timestamp"] = frame.loc[58, "timestamp"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_replay_data(frame)


def test_replay_rejects_non_finite_and_invalid_ohlc():
    frame = sample_frame()
    frame.loc[59, "close"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_replay_data(frame)
    frame = sample_frame()
    frame.loc[59, "high"] = 98.0
    with pytest.raises(ValueError, match="invalid high"):
        validate_replay_data(frame)


def test_replay_reports_raw_detector_setups(monkeypatch):
    import app.replay as replay

    def fake_evaluate(strategy, candles, thresholds):
        return ({
            "side": "LONG", "entry_reference": 100.0, "trigger_price": 100.2,
            "stop_loss": 98.0, "take_profit_1": 103.0, "atr": 2.0,
            "reason": "synthetic test setup",
        }, {})

    monkeypatch.setattr(replay, "_evaluate", fake_evaluate)
    events, summary = replay_dataframe(
        sample_frame(), strategies=("ORB",),
        threshold_overrides={"tpr_adx_min": 15.0},
    )
    assert len(events) == 1
    assert events.iloc[0]["decision"] == "setup"
    assert summary["strategies"]["ORB"]["evaluations"] == 1
    assert summary["strategies"]["ORB"]["setups"] == 1
    assert summary["mode"] == "detector_replay_only"
    assert summary["thresholds"]["tpr_adx_min"] == 15.0
    assert "No live L1 bid/ask quote confirmation is simulated." in summary["limitations"]


def test_replay_resets_warmup_at_candle_gaps(monkeypatch):
    import app.replay as replay

    frame = sample_frame(119)
    frame.loc[59:, "timestamp"] += pd.Timedelta(minutes=5)

    calls = []

    def fake_evaluate(strategy, candles, thresholds):
        calls.append(candles.iloc[-1]["timestamp"])
        return None, {"reason": "synthetic_reject"}

    monkeypatch.setattr(replay, "_evaluate", fake_evaluate)
    _events, summary = replay_dataframe(frame, strategies=("ORB",))
    # Segment one has 59 candles (not enough); segment two has 60 and earns
    # exactly one evaluation after its own 60-candle warm-up.
    assert summary["gap_boundaries"] == 1
    assert summary["strategies"]["ORB"]["evaluations"] == 1
    assert len(calls) == 1



def test_replay_rejects_invalid_threshold_overrides():
    frame = sample_frame()
    with pytest.raises(ValueError, match="ADX minima non-negative"):
        replay_dataframe(frame, strategies=("TPR",), threshold_overrides={"tpr_adx_min": -1})
    with pytest.raises(ValueError, match="ADX minima non-negative"):
        replay_dataframe(frame, strategies=("RB",), threshold_overrides={"rb_min_rvol": 0})



def test_replay_applies_strategy_symbol_universe():
    frame = sample_frame()
    events, summary = replay_dataframe(
        frame, strategies=("ORB",), strategy_symbols={"ORB": ["ETHUSD"]},
    )
    assert events.empty
    assert summary["strategies"]["ORB"]["eligible_symbols"] == 0
    assert summary["strategies"]["ORB"]["evaluations"] == 0




def test_load_strategy_universes_rejects_overlap(tmp_path):
    from app.replay import load_strategy_universes

    momentum = [f"S{i:03d}USD" for i in range(60)]
    tpr_rb = [f"T{i:03d}USD" for i in range(59)] + [momentum[0]]
    (tmp_path / "momentum_symbols.txt").write_text("\n".join(momentum), encoding="utf-8")
    (tmp_path / "tpr_rb_symbols.txt").write_text("\n".join(tpr_rb), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_strategy_universes(tmp_path)
