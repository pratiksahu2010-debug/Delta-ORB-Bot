import asyncio
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.models import Candle, Quote
from app.signal_engine import PendingSetup, SignalEngine


class FakeNotifier:
    def __init__(self):
        self.alerts = []

    async def send_signal(self, alert, feedback_token=""):
        self.alerts.append(alert)
        return True


def make_engine(tmp_path):
    settings = SimpleNamespace(
        delta_base_url="https://api.india.delta.exchange",
        state_db_path=str(tmp_path / "signals.sqlite3"),
        symbol_list=["BTCUSD"],
        tpr_rb_symbol_list=["BTCUSD"],
        momentum_symbol_list=[],
        orb_enabled=False,
        orb_min_rvol=1.10,
        tpr_enabled=True,
        rb_enabled=True,
        mex_enabled=False,
        rb_min_rvol=1.35,
        rb_range_lookback=12,
        mex_min_rvol=1.50,
        mex_min_atr_expansion=1.15,
        mex_live_min_eval_interval_seconds=0.75,
        quote_max_age_seconds=3.0,
        max_entry_extension_atr=0.8,
        max_candle_lag_seconds=600,
        candle_poll_interval_seconds=0.0,
        adx_min=15.0,
        signal_max_age_bars=2,
        trigger_confirmation_seconds=0.5,
        trigger_confirmation_max_gap_seconds=1.5,
    )
    notifier = FakeNotifier()
    return SignalEngine(settings, notifier), notifier


async def wait_for_outbox(engine):
    task = engine.outbox_task
    if task is not None:
        await task


def make_setup(side, now):
    return PendingSetup(
        symbol="BTCUSD", side=side, candle_time=now - timedelta(minutes=5),
        trigger_price=101.0 if side == "LONG" else 99.0,
        reference=100.0, stop_loss=98.0 if side == "LONG" else 102.0,
        take_profit_1=103.0 if side == "LONG" else 97.0,
        atr=2.0, bars_remaining=2, reason="test setup",
        context_15m="BULLISH", expires_at=now + timedelta(minutes=5),
    )





def test_advika_btc_frame_is_reused_until_candle_snapshot_changes(tmp_path):
    engine, _notifier = make_engine(tmp_path)
    now = datetime.now(timezone.utc)
    first = [Candle(now, 100, 102, 99, 101, 10)]
    engine.candles["BTCUSD"] = first

    frame_one = engine._advika_btc_context_frame()
    frame_two = engine._advika_btc_context_frame()
    assert frame_one is frame_two
    assert frame_one.iloc[-1]["close"] == 101

    second = [Candle(now + timedelta(minutes=5), 101, 103, 100, 102, 12)]
    engine.candles["BTCUSD"] = second
    frame_three = engine._advika_btc_context_frame()
    assert frame_three is not frame_one
    assert frame_three.iloc[-1]["close"] == 102
    asyncio.run(engine.rest.close())


def test_engine_caches_symbol_universe_for_hot_paths(tmp_path):
    engine, _notifier = make_engine(tmp_path)
    assert engine.configured_symbols == ["BTCUSD"]
    assert engine.configured_symbol_set == frozenset({"BTCUSD"})
    assert engine.tpr_rb_symbols == frozenset({"BTCUSD"})

    # Runtime membership uses the startup snapshot, not mutable Settings lists.
    engine.settings.symbol_list = ["ETHUSD"]
    engine.settings.tpr_rb_symbol_list = ["ETHUSD"]
    assert "BTCUSD" in engine.configured_symbol_set
    assert "ETHUSD" not in engine.configured_symbol_set
    assert "BTCUSD" in engine.tpr_rb_symbols
    assert "ETHUSD" not in engine.tpr_rb_symbols
    asyncio.run(engine.rest.close())


def test_candle_gap_invalidates_all_pending_strategy_setups(tmp_path):
    engine, _notifier = make_engine(tmp_path)
    now = datetime.now(timezone.utc)
    tpr = make_setup("LONG", now)
    rb = make_setup("SHORT", now)
    rb.strategy = "RB"
    engine.pending["BTCUSD"] = tpr
    engine.momentum_pending[("BTCUSD", "RB")] = rb

    engine._invalidate_pending_setups_for_symbol(
        "BTCUSD", "invalidated_missing_5m_candle_intervals",
    )

    assert "BTCUSD" not in engine.pending
    assert ("BTCUSD", "RB") not in engine.momentum_pending
    assert engine.store.setup_lifecycle(
        "BTCUSD", "TPR", tpr.candle_time.isoformat(),
    )["state"] == "INVALIDATED"
    assert engine.store.setup_lifecycle(
        "BTCUSD", "RB", rb.candle_time.isoformat(),
    )["state"] == "INVALIDATED"
    assert engine.store.load_pending_setups() == []
    import asyncio
    asyncio.run(engine.rest.close())


def test_long_trigger_uses_ask_and_recalculates_target(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.pending["BTCUSD"] = make_setup("LONG", now)
        quote = Quote(price=101.0, bid=100.9, ask=101.1, received_at=now)
        await engine.on_quote("BTCUSD", quote, now)
        assert notifier.alerts == []
        later = now + timedelta(seconds=0.6)
        quote = Quote(price=101.0, bid=100.9, ask=101.1, received_at=later)
        await engine.on_quote("BTCUSD", quote, later)
        await wait_for_outbox(engine)
        assert len(notifier.alerts) == 1
        alert = notifier.alerts[0]
        assert alert["entry"] == 101.1
        assert alert["take_profit_1"] == pytest.approx(107.3)
        assert alert["context_15m"] == "BULLISH"
        await engine.rest.close()
    asyncio.run(run())


def test_short_trigger_uses_bid(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.pending["BTCUSD"] = make_setup("SHORT", now)
        quote = Quote(price=99.0, bid=98.9, ask=99.1, received_at=now)
        await engine.on_quote("BTCUSD", quote, now)
        assert notifier.alerts == []
        later = now + timedelta(seconds=0.6)
        quote = Quote(price=99.0, bid=98.9, ask=99.1, received_at=later)
        await engine.on_quote("BTCUSD", quote, later)
        await wait_for_outbox(engine)
        assert len(notifier.alerts) == 1
        assert notifier.alerts[0]["entry"] == 98.9
        assert notifier.alerts[0]["take_profit_1"] == pytest.approx(92.7)
        await engine.rest.close()
    asyncio.run(run())


def test_expired_setup_cannot_trigger(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        setup.expires_at = now - timedelta(seconds=1)
        engine.pending["BTCUSD"] = setup
        quote = Quote(price=102.0, bid=101.9, ask=102.1, received_at=now)
        await engine.on_quote("BTCUSD", quote, now)
        assert notifier.alerts == []
        assert "BTCUSD" not in engine.pending
        await engine.rest.close()
    asyncio.run(run())


def test_long_trigger_fails_closed_without_ask(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.pending["BTCUSD"] = make_setup("LONG", now)
        quote = Quote(price=102.0, bid=101.9, ask=None, received_at=now)
        await engine.on_quote("BTCUSD", quote, now)
        assert notifier.alerts == []
        assert "BTCUSD" in engine.pending
        await engine.rest.close()
    asyncio.run(run())


def test_short_trigger_fails_closed_without_bid(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.pending["BTCUSD"] = make_setup("SHORT", now)
        quote = Quote(price=98.0, bid=None, ask=98.1, received_at=now)
        await engine.on_quote("BTCUSD", quote, now)
        assert notifier.alerts == []
        assert "BTCUSD" in engine.pending
        await engine.rest.close()
    asyncio.run(run())



def test_one_tick_breakout_that_reverses_is_not_alerted(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.pending["BTCUSD"] = make_setup("LONG", now)
        crossed = Quote(price=101.2, bid=101.1, ask=101.2, received_at=now)
        await engine.on_quote("BTCUSD", crossed, now)
        reversed_at = now + timedelta(seconds=0.2)
        reversed_quote = Quote(price=100.8, bid=100.7, ask=100.8, received_at=reversed_at)
        await engine.on_quote("BTCUSD", reversed_quote, reversed_at)
        later = now + timedelta(seconds=0.8)
        crossed_again = Quote(price=101.2, bid=101.1, ask=101.2, received_at=later)
        await engine.on_quote("BTCUSD", crossed_again, later)
        assert notifier.alerts == []
        await engine.rest.close()
    asyncio.run(run())


def test_breakout_requires_distinct_quotes_and_hold_time(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.pending["BTCUSD"] = make_setup("LONG", now)
        first = Quote(price=101.2, bid=101.1, ask=101.2, received_at=now)
        await engine.on_quote("BTCUSD", first, now)
        too_soon = now + timedelta(seconds=0.2)
        second = Quote(price=101.2, bid=101.1, ask=101.2, received_at=too_soon)
        await engine.on_quote("BTCUSD", second, too_soon)
        assert notifier.alerts == []
        enough_time = now + timedelta(seconds=0.6)
        third = Quote(price=101.2, bid=101.1, ask=101.2, received_at=enough_time)
        await engine.on_quote("BTCUSD", third, enough_time)
        await wait_for_outbox(engine)
        assert len(notifier.alerts) == 1
        await engine.rest.close()
    asyncio.run(run())



def test_missing_directional_quote_resets_confirmation_candidate(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        setup.trigger_candidate_at = now - timedelta(seconds=0.4)
        setup.trigger_candidate_last_at = now - timedelta(seconds=0.2)
        engine.pending["BTCUSD"] = setup
        missing_ask = Quote(price=102.0, bid=101.9, ask=None, received_at=now)
        await engine.on_quote("BTCUSD", missing_ask, now)
        assert setup.trigger_candidate_at is None
        assert setup.trigger_candidate_last_at is None
        assert notifier.alerts == []
        await engine.rest.close()
    asyncio.run(run())


def test_missing_directional_bid_resets_short_confirmation_candidate(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("SHORT", now)
        setup.trigger_candidate_at = now - timedelta(seconds=0.4)
        setup.trigger_candidate_last_at = now - timedelta(seconds=0.2)
        engine.pending["BTCUSD"] = setup
        missing_bid = Quote(price=98.0, bid=None, ask=98.1, received_at=now)
        await engine.on_quote("BTCUSD", missing_bid, now)
        assert setup.trigger_candidate_at is None
        assert setup.trigger_candidate_last_at is None
        assert notifier.alerts == []
        await engine.rest.close()
    asyncio.run(run())


def test_refresh_keeps_older_pending_setup_instead_of_replacing_it(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(
                timestamp=latest_time - timedelta(minutes=5 * (59 - i)),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0,
            )
            for i in range(60)
        ]
        old_setup = make_setup("LONG", now)
        old_setup.candle_time = latest_time - timedelta(minutes=5)
        old_setup.expires_at = latest_time + timedelta(minutes=10)
        engine.pending["BTCUSD"] = old_setup

        async def fake_candles(_symbol, _timeframe, _limit):
            return bars

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        monkeypatch.setattr(
            signal_engine_module, "evaluate_tpr",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not re-arm while pending")),
        )
        await engine.refresh_symbol("BTCUSD")
        assert engine.pending["BTCUSD"] is old_setup
        assert old_setup.bars_remaining == 1
        await engine.rest.close()
    asyncio.run(run())


def test_refresh_expires_pending_setup_after_candle_time_deadline(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(
                timestamp=latest_time - timedelta(minutes=5 * (59 - i)),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0,
            )
            for i in range(60)
        ]
        old_setup = make_setup("LONG", now)
        old_setup.candle_time = latest_time - timedelta(minutes=20)
        old_setup.expires_at = latest_time - timedelta(minutes=1)
        engine.pending["BTCUSD"] = old_setup

        async def fake_candles(_symbol, _timeframe, _limit):
            return bars

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        monkeypatch.setattr(signal_engine_module, "evaluate_tpr", lambda *_args, **_kwargs: None)
        await engine.refresh_symbol("BTCUSD")
        assert "BTCUSD" not in engine.pending
        await engine.rest.close()
    asyncio.run(run())



def test_duplicate_or_out_of_order_exchange_quote_resets_confirmation(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        engine.pending["BTCUSD"] = setup
        first = Quote(price=101.2, bid=101.1, ask=101.2, received_at=now)
        await engine.on_quote("BTCUSD", first, now)
        assert setup.trigger_candidate_at == now
        duplicate = Quote(price=101.3, bid=101.2, ask=101.3, received_at=now)
        await engine.on_quote("BTCUSD", duplicate, now + timedelta(seconds=0.6))
        assert setup.trigger_candidate_at is None
        assert setup.trigger_candidate_last_at is None
        assert notifier.alerts == []
        await engine.rest.close()
    asyncio.run(run())


def test_websocket_disconnect_resets_all_pending_confirmations(tmp_path):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        setup.trigger_candidate_at = now
        setup.trigger_candidate_last_at = now
        engine.pending["BTCUSD"] = setup
        engine.last_quote_exchange_at["BTCUSD"] = now
        engine.on_stream_disconnect()
        assert setup.trigger_candidate_at is None
        assert setup.trigger_candidate_last_at is None
        assert engine.last_quote_exchange_at == {}
        await engine.rest.close()
    asyncio.run(run())


def test_duplicate_exchange_quote_does_not_refresh_cached_quote_health(tmp_path):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        exchange_time = datetime.now(timezone.utc)
        first_received = exchange_time
        first = Quote(price=100.5, bid=100.4, ask=100.6, received_at=exchange_time)
        await engine.on_quote("BTCUSD", first, first_received)

        duplicate_received = exchange_time + timedelta(seconds=0.6)
        duplicate = Quote(price=101.5, bid=101.4, ask=101.6, received_at=exchange_time)
        await engine.on_quote("BTCUSD", duplicate, duplicate_received)

        assert engine.quotes["BTCUSD"] is first
        assert engine.received_at["BTCUSD"] == first_received
        assert engine.last_quote_exchange_at["BTCUSD"] == exchange_time
        await engine.rest.close()

    asyncio.run(run())


def test_quote_health_uses_delta_ob_l1_idle_cadence_without_relaxing_entry_gate(tmp_path):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.quotes["BTCUSD"] = Quote(
            price=100.5, bid=100.4, ask=100.6, received_at=now - timedelta(seconds=4)
        )
        engine.received_at["BTCUSD"] = now - timedelta(seconds=4)

        health = engine.health()
        assert health["quote_health_max_age_seconds"] == 7.0
        assert health["quote_stale_symbols"] == []
        assert health["stale_quotes"] == 0
        # The actual entry/trigger freshness remains at the configured 3 seconds.
        assert health["entry_quote_stale_symbols"] == ["BTCUSD"]
        assert health["entry_stale_quotes"] == 1
        await engine.rest.close()

    asyncio.run(run())


def test_strategy_exception_does_not_mark_completed_candle_as_evaluated(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(
                timestamp=latest_time - timedelta(minutes=5 * (59 - i)),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0,
            )
            for i in range(60)
        ]

        async def fake_candles(_symbol, _timeframe, _limit, *, since=None):
            return bars

        async def no_context(_symbol):
            return None

        calls = {"count": 0}

        def flaky_evaluation(_frame, *, adx_min, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("temporary indicator calculation failure")
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        monkeypatch.setattr(signal_engine_module, "evaluate_tpr", flaky_evaluation)

        with pytest.raises(RuntimeError, match="temporary indicator"):
            await engine.refresh_symbol("BTCUSD")
        assert "BTCUSD" not in engine.last_evaluated

        # The same completed candle is fetched again and evaluated on next poll.
        await engine.refresh_symbol("BTCUSD")
        assert calls["count"] == 2
        assert engine.last_evaluated["BTCUSD"] == latest_time
        await engine.rest.close()

    asyncio.run(run())

def test_refresh_ignores_regressed_rest_history_snapshot(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        newest = now - timedelta(minutes=5)
        cached = [
            Candle(timestamp=newest - timedelta(minutes=5 * (59 - i)),
                   open=100, high=101, low=99, close=100.5, volume=10)
            for i in range(60)
        ]
        regressed_latest = newest - timedelta(minutes=5)
        regressed = [
            Candle(timestamp=regressed_latest - timedelta(minutes=5 * (59 - i)),
                   open=100, high=101, low=99, close=100.5, volume=10)
            for i in range(60)
        ]
        engine.candles["BTCUSD"] = cached
        engine.last_evaluated["BTCUSD"] = newest

        async def fake_candles(_symbol, _timeframe, _limit, *, since=None):
            return regressed

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        monkeypatch.setattr(
            signal_engine_module, "evaluate_tpr",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("regressed history must not be evaluated")),
        )
        await engine.refresh_symbol("BTCUSD")
        assert engine.candles["BTCUSD"] is cached
        assert engine.candles["BTCUSD"][-1].timestamp == newest
        assert engine.last_evaluated["BTCUSD"] == newest
        await engine.rest.close()

    asyncio.run(run())



def test_incremental_candle_refresh_merges_into_full_history(tmp_path, monkeypatch):
    from app.models import Candle

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        cached = [
            Candle(timestamp=latest_time - timedelta(minutes=5 * (119 - i)),
                   open=100, high=101, low=99, close=100.5, volume=10)
            for i in range(120)
        ]
        engine.candles["BTCUSD"] = cached
        engine.last_evaluated["BTCUSD"] = latest_time
        calls = []
        corrected = Candle(
            timestamp=latest_time, open=100, high=101.5, low=99,
            close=100.8, volume=12,
        )

        async def incremental(_symbol, _timeframe, _limit, *, since=None):
            calls.append(since)
            return [corrected]

        monkeypatch.setattr(engine.rest, "candles", incremental)
        await engine.refresh_symbol("BTCUSD")

        assert calls == [int(latest_time.timestamp()) - 600]
        assert len(engine.candles["BTCUSD"]) == 120
        assert engine.candles["BTCUSD"][-1] == corrected
        assert engine.last_evaluated["BTCUSD"] == latest_time
        await engine.rest.close()

    asyncio.run(run())


def test_refresh_does_not_re_evaluate_older_candle_when_cache_is_empty(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(timestamp=latest_time - timedelta(minutes=5 * (59 - i)),
                   open=100, high=101, low=99, close=100.5, volume=10)
            for i in range(60)
        ]
        engine.last_evaluated["BTCUSD"] = latest_time + timedelta(minutes=5)

        async def fake_candles(_symbol, _timeframe, _limit):
            return bars

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        monkeypatch.setattr(
            signal_engine_module, "evaluate_tpr",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("older candle must not be evaluated")),
        )
        await engine.refresh_symbol("BTCUSD")
        assert engine.last_evaluated["BTCUSD"] == latest_time + timedelta(minutes=5)
        assert engine.pending == {}
        await engine.rest.close()

    asyncio.run(run())



def test_old_exchange_timestamp_does_not_reject_locally_fresh_quote(tmp_path):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        received = datetime.now(timezone.utc)
        setup = make_setup("LONG", received)
        engine.pending["BTCUSD"] = setup
        quote = Quote(
            price=101.2, bid=101.1, ask=101.2,
            received_at=received - timedelta(minutes=6),
        )

        await engine.on_quote("BTCUSD", quote, received)

        assert engine.quotes["BTCUSD"] is quote
        assert engine.received_at["BTCUSD"] == received
        assert setup.trigger_candidate_at == received
        assert engine.quote_rejection_diagnostics["counts"] == {}
        await engine.rest.close()

    asyncio.run(run())


def test_old_exchange_timestamp_does_not_reset_active_confirmation(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        received = datetime.now(timezone.utc)
        setup = make_setup("LONG", received)
        setup.trigger_candidate_at = received - timedelta(seconds=0.4)
        setup.trigger_candidate_last_at = received - timedelta(seconds=0.2)
        engine.pending["BTCUSD"] = setup
        quote = Quote(
            price=101.2, bid=101.1, ask=101.2,
            received_at=received - timedelta(minutes=6),
        )

        await engine.on_quote("BTCUSD", quote, received)

        assert setup.trigger_candidate_at == received - timedelta(seconds=0.4)
        assert setup.trigger_candidate_last_at == received
        assert engine.quotes["BTCUSD"] is quote
        assert notifier.alerts == []
        await engine.rest.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("side", "invalid_stop"),
    [("LONG", 102.0), ("SHORT", 98.0)],
)
def test_trigger_rejects_stop_loss_on_wrong_side_of_entry(tmp_path, side, invalid_stop):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup(side, now)
        setup.stop_loss = invalid_stop
        engine.pending["BTCUSD"] = setup

        if side == "LONG":
            quote_args = {"price": 101.2, "bid": 101.1, "ask": 101.2}
        else:
            quote_args = {"price": 98.9, "bid": 98.9, "ask": 99.0}

        first = Quote(**quote_args, received_at=now)
        await engine.on_quote("BTCUSD", first, now)
        later = now + timedelta(seconds=0.6)
        second = Quote(**quote_args, received_at=later)
        await engine.on_quote("BTCUSD", second, later)

        assert notifier.alerts == []
        assert setup.trigger_candidate_at is None
        assert setup.trigger_candidate_last_at is None
        assert engine.pending["BTCUSD"] is setup
        await engine.rest.close()

    asyncio.run(run())


def test_trigger_rejects_nonfinite_stop_loss(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        setup.stop_loss = float("nan")
        engine.pending["BTCUSD"] = setup
        quote = Quote(price=101.2, bid=101.1, ask=101.2, received_at=now)
        await engine.on_quote("BTCUSD", quote, now)
        later = now + timedelta(seconds=0.6)
        await engine.on_quote(
            "BTCUSD",
            Quote(price=101.2, bid=101.1, ask=101.2, received_at=later),
            later,
        )
        assert notifier.alerts == []
        assert setup.trigger_candidate_at is None
        await engine.rest.close()

    asyncio.run(run())


def test_outbox_delivers_persisted_alert_after_engine_restart(tmp_path):
    async def run():
        from app.state_store import SignalStore

        key = "BTCUSD:LONG:2026-11-15T10:00:00+00:00"
        alert = {
            "symbol": "BTCUSD", "side": "LONG",
            "entry": 100.0, "stop_loss": 98.0, "take_profit_1": 103.0,
            "setup_time": "2026-11-15T10:00:00+00:00",
            "reason": "persisted trigger", "context_15m": "BULLISH",
        }
        first_store = SignalStore(str(tmp_path / "signals.sqlite3"))
        first_store.enqueue_alert(key, alert)

        # A fresh engine instance represents a process restart.
        engine, notifier = make_engine(tmp_path)
        await engine.deliver_outbox()

        assert notifier.alerts == [alert]
        assert engine.store.is_sent(key)
        assert engine.store.due_alerts() == []
        await engine.rest.close()

    asyncio.run(run())






def test_momentum_only_symbol_does_not_crash_when_no_pullback_is_pending(tmp_path, monkeypatch):
    from app.models import Candle

    engine, _notifier = make_engine(tmp_path)
    engine.settings.momentum_symbol_list = ["ALTUSD"]
    engine.settings.orb_enabled = False
    engine.settings.mex_enabled = False
    engine.settings.orb_sessions = "asia,london,new_york"

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    latest_open = now - timedelta(minutes=5)
    bars = [
        Candle(
            timestamp=latest_open - timedelta(minutes=5 * (119 - i)),
            open=100.0, high=100.2, low=99.8, close=100.0, volume=10.0,
        )
        for i in range(120)
    ]

    async def fake_refresh_context(_symbol):
        raise AssertionError("informational 15m context must not block the 5m scanner")

    async def fake_candles(_symbol, _timeframe, _limit):
        return bars

    async def fail_pullback_evaluation(*_args, **_kwargs):
        raise AssertionError("pullback evaluator must not run for momentum-only symbols")

    engine.refresh_context = fake_refresh_context
    engine.rest.candles = fake_candles
    monkeypatch.setattr("app.signal_engine.evaluate_tpr", fail_pullback_evaluation)

    async def run():
        await engine.refresh_symbol("ALTUSD")
        assert engine.last_evaluated["ALTUSD"] == bars[-1].timestamp
        await engine.rest.close()

    asyncio.run(run())


def test_refresh_symbol_throttles_repeated_5m_rest_polls(tmp_path, monkeypatch):
    from app.models import Candle

    async def run():
        engine, _notifier = make_engine(tmp_path)
        engine.settings.candle_poll_interval_seconds = 30.0
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(timestamp=latest_time - timedelta(minutes=5 * (59 - i)),
                   open=100, high=101, low=99, close=100.5, volume=10)
            for i in range(60)
        ]
        calls = {"count": 0}

        async def fake_candles(_symbol, _timeframe, _limit):
            calls["count"] += 1
            return bars

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        await engine.refresh_symbol("BTCUSD")
        await engine.refresh_symbol("BTCUSD")
        assert calls["count"] == 1
        await engine.rest.close()

    asyncio.run(run())


def test_15m_context_fetches_once_per_bucket_and_retries_unavailable_slowly(tmp_path, monkeypatch):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        calls = {"count": 0}

        async def no_bars(_symbol, _timeframe, _limit):
            calls["count"] += 1
            return []

        monkeypatch.setattr(engine.rest, "candles", no_bars)
        await engine.refresh_context("BTCUSD")
        await engine.refresh_context("BTCUSD")
        assert calls["count"] == 1
        assert engine.context_15m["BTCUSD"] == "UNAVAILABLE"
        await engine.rest.close()

    asyncio.run(run())



def test_short_history_is_reported_as_warming_and_never_evaluated(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(timestamp=latest_time - timedelta(minutes=5 * (29 - i)),
                   open=100, high=101, low=99, close=100.5, volume=10)
            for i in range(30)
        ]

        async def fake_candles(_symbol, _timeframe, _limit):
            return bars

        async def no_context(_symbol):
            return None

        def fail_evaluation(*_args, **_kwargs):
            raise AssertionError("indicator engine must not run before 60 candles")

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        monkeypatch.setattr(signal_engine_module, "evaluate_tpr", fail_evaluation)

        await engine.refresh_symbol("BTCUSD")
        health = engine.health()
        assert "BTCUSD" not in engine.candles
        assert health["candle_warming_count"] == 1
        assert health["candle_warming_symbols"] == ["BTCUSD"]
        assert health["candle_ready_count"] == 0
        assert health["candle_diagnostics"]["BTCUSD"]["bars"] == 30
        await engine.rest.close()

    asyncio.run(run())


def test_empty_candle_response_is_reported_separately_from_warmup(tmp_path, monkeypatch):
    async def run():
        engine, _notifier = make_engine(tmp_path)

        async def fake_candles(_symbol, _timeframe, _limit):
            return []

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        await engine.refresh_symbol("BTCUSD")
        health = engine.health()
        assert health["candle_no_data_count"] == 1
        assert health["candle_no_data_symbols"] == ["BTCUSD"]
        assert health["candle_warming_count"] == 0
        await engine.rest.close()

    asyncio.run(run())


def test_transient_candle_api_failure_is_reported_and_throttled(tmp_path, monkeypatch):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        calls = {"count": 0}

        async def flaky_candles(_symbol, _timeframe, _limit):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("temporary upstream failure")
            return []

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", flaky_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)

        with pytest.raises(RuntimeError, match="temporary upstream failure"):
            await engine.refresh_symbol("BTCUSD")
        health = engine.health()
        assert health["candle_error_count"] == 1
        assert health["candle_error_symbols"] == ["BTCUSD"]
        assert health["candle_diagnostics"]["BTCUSD"]["last_attempt_status"] == "error"

        engine.settings.candle_poll_interval_seconds = 30.0
        await engine.refresh_symbol("BTCUSD")
        assert calls["count"] == 1
        await engine.rest.close()

    asyncio.run(run())


def test_fresh_contiguous_60_candles_are_marked_ready(tmp_path, monkeypatch):
    from app.models import Candle

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(timestamp=latest_time - timedelta(minutes=5 * (59 - i)),
                   open=100, high=101, low=99, close=100.5, volume=10)
            for i in range(60)
        ]

        async def fake_candles(_symbol, _timeframe, _limit):
            return bars

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        await engine.refresh_symbol("BTCUSD")
        health = engine.health()
        assert health["candle_ready_count"] == 1
        assert health["candle_warming_count"] == 0
        assert health["candle_diagnostics"]["BTCUSD"]["state"] == "ready"
        await engine.rest.close()

    asyncio.run(run())


def test_candle_gap_is_reported_and_symbol_is_not_ready(tmp_path, monkeypatch):
    from app.models import Candle

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(timestamp=latest_time - timedelta(minutes=5 * (59 - i) + (5 if i >= 30 else 0)),
                   open=100, high=101, low=99, close=100.5, volume=10)
            for i in range(60)
        ]

        async def fake_candles(_symbol, _timeframe, _limit):
            return bars

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        await engine.refresh_symbol("BTCUSD")
        health = engine.health()
        assert health["candle_gap_symbols"] == 1
        assert health["candle_gap_list"] == ["BTCUSD"]
        assert health["candle_ready_count"] == 0
        await engine.rest.close()

    asyncio.run(run())





def test_fenced_trigger_confirmation_never_enqueues_alert(tmp_path, monkeypatch):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        engine.pending["BTCUSD"] = setup
        original_transition = engine._transition_setup

        def reject_confirmed_transition(candidate, to_state, reason):
            if to_state == "TRIGGER_CONFIRMED":
                return False
            return original_transition(candidate, to_state, reason)

        monkeypatch.setattr(engine, "_transition_setup", reject_confirmed_transition)
        first = Quote(price=101.2, bid=101.1, ask=101.2, received_at=now)
        await engine.on_quote("BTCUSD", first, now)
        later = now + timedelta(seconds=0.6)
        second = Quote(price=101.2, bid=101.1, ask=101.2, received_at=later)
        await engine.on_quote("BTCUSD", second, later)

        assert engine.store.due_alerts() == []
        assert notifier.alerts == []
        assert setup.lifecycle_state == "CONFIRMING"
        await engine.rest.close()

    asyncio.run(run())


def test_trigger_lifecycle_health_records_confirmation_and_delivery_path(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.pending["BTCUSD"] = make_setup("LONG", now)

        first = Quote(price=101.2, bid=101.1, ask=101.2, received_at=now)
        await engine.on_quote("BTCUSD", first, now)
        later = now + timedelta(seconds=0.6)
        second = Quote(price=101.2, bid=101.1, ask=101.2, received_at=later)
        await engine.on_quote("BTCUSD", second, later)

        lifecycle = engine.health()["trigger_lifecycle"]["ORB"]["setup_types"]["TPR"]["events"]
        assert lifecycle["confirmation_started"] == 1
        assert lifecycle["trigger_confirmed"] == 1
        await wait_for_outbox(engine)
        assert len(notifier.alerts) == 1
        await engine.rest.close()

    asyncio.run(run())


def test_trigger_lifecycle_records_non_crossed_and_missing_side_quote(tmp_path):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.pending["BTCUSD"] = make_setup("LONG", now)

        missing_ask = Quote(price=100.5, bid=100.4, ask=None, received_at=now)
        await engine.on_quote("BTCUSD", missing_ask, now)
        later = now + timedelta(seconds=0.6)
        not_crossed = Quote(price=100.5, bid=100.4, ask=100.5, received_at=later)
        await engine.on_quote("BTCUSD", not_crossed, later)

        events = engine.health()["trigger_lifecycle"]["ORB"]["setup_types"]["TPR"]["events"]
        assert events["missing_ask_for_long"] == 1
        assert events["trigger_not_crossed"] == 1
        await engine.rest.close()

    asyncio.run(run())


def test_trigger_lifecycle_records_websocket_confirmation_reset(tmp_path):
    engine, _notifier = make_engine(tmp_path)
    now = datetime.now(timezone.utc)
    setup = make_setup("LONG", now)
    setup.trigger_candidate_at = now
    setup.trigger_candidate_last_at = now
    engine.pending["BTCUSD"] = setup

    engine.on_stream_disconnect()
    events = engine.health()["trigger_lifecycle"]["ORB"]["setup_types"]["TPR"]["events"]
    assert events["websocket_disconnect"] == 1
    assert events["confirmation_reset_websocket_disconnect"] == 1
    assert setup.trigger_candidate_at is None

    asyncio.run(engine.rest.close())


def test_trigger_lifecycle_distinguishes_quote_extension_rejection(tmp_path):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        engine.pending["BTCUSD"] = make_setup("LONG", now)
        # Ask is 1 ATR above the reference, beyond the configured 0.8 ATR cap.
        quote = Quote(price=102.0, bid=101.9, ask=102.0, received_at=now)
        await engine.on_quote("BTCUSD", quote, now)
        events = engine.health()["trigger_lifecycle"]["ORB"]["setup_types"]["TPR"]["events"]
        assert events["quote_validation_rejected_entry_too_extended"] == 1
        await engine.rest.close()

    asyncio.run(run())


def test_trigger_lifecycle_records_setup_expired_on_completed_candle(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(timestamp=latest_time - timedelta(minutes=5 * (59 - i)),
                   open=100, high=101, low=99, close=100.5, volume=10)
            for i in range(60)
        ]
        setup = make_setup("LONG", now)
        setup.expires_at = latest_time
        engine.pending["BTCUSD"] = setup

        async def fake_candles(_symbol, _timeframe, _limit):
            return bars

        async def no_context(_symbol):
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        monkeypatch.setattr(signal_engine_module, "evaluate_tpr", lambda *_args, **_kwargs: None)
        await engine.refresh_symbol("BTCUSD")

        events = engine.health()["trigger_lifecycle"]["ORB"]["setup_types"]["TPR"]["events"]
        assert events["setup_expired_on_candle"] == 1
        assert "BTCUSD" not in engine.pending
        await engine.rest.close()

    asyncio.run(run())




def test_pending_setup_arm_persists_before_memory_activation(tmp_path):
    engine, _ = make_engine(tmp_path)
    now = datetime.now(timezone.utc)
    latest = __import__("app.models", fromlist=["Candle"]).Candle(
        timestamp=now - timedelta(minutes=5), open=100, high=102, low=99, close=101, volume=10
    )
    engine._arm_setup("BTCUSD", "RB", {
        "side": "LONG", "trigger_price": 102, "entry_reference": 101,
        "stop_loss": 99, "take_profit_1": 104, "atr": 2,
        "reason": "test RB setup",
    }, latest)
    assert ("BTCUSD", "RB") in engine.momentum_pending
    assert engine.store.pending_setup_count() == 1
    asyncio.run(engine.rest.close())


@pytest.mark.parametrize("strategy", ["TPR", "RB", "MEX"])
def test_complete_setup_to_durable_telegram_alert_path_for_each_strategy(tmp_path, strategy):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        setup.strategy = strategy
        engine._persist_pending_setup(setup)
        if strategy == "TPR":
            engine.pending["BTCUSD"] = setup
        else:
            engine.momentum_pending[("BTCUSD", strategy)] = setup

        first = Quote(price=101.1, bid=101.0, ask=101.1, received_at=now)
        await engine.on_quote("BTCUSD", first, now)
        assert notifier.alerts == []

        later = now + timedelta(seconds=0.6)
        second = Quote(price=101.1, bid=101.0, ask=101.1, received_at=later)
        await engine.on_quote("BTCUSD", second, later)

        await wait_for_outbox(engine)
        assert len(notifier.alerts) == 1
        assert notifier.alerts[0]["strategy"] == strategy
        key = f"BTCUSD:{strategy}:LONG:{setup.candle_time.isoformat()}"
        assert engine.store.is_sent(key)
        assert engine.store.outbox_health()["sent"] == 1
        assert engine.store.pending_setup_count() == 0
        await engine.rest.close()

    asyncio.run(run())


def test_health_exposes_stale_quote_and_websocket_heartbeat(tmp_path):
    engine, _ = make_engine(tmp_path)
    now = datetime.now(timezone.utc)
    engine.stream.connected = True
    engine.stream.last_heartbeat_at = now - timedelta(seconds=40)
    engine.received_at["BTCUSD"] = now - timedelta(seconds=5)
    engine.quotes["BTCUSD"] = Quote(price=100, bid=99.9, ask=100.1, received_at=now - timedelta(seconds=5))

    health = engine.health()
    assert health["websocket_heartbeat_stale"] is True
    # Five seconds is within Delta ob_l1's documented no-change publication interval.
    assert health["quote_stale_symbols"] == []
    assert health["entry_quote_stale_symbols"] == ["BTCUSD"]
    assert health["quote_age_seconds"]["BTCUSD"] >= 5

    asyncio.run(engine.rest.close())










def test_symbol_validation_subscribes_only_live_configured_perpetuals(tmp_path, monkeypatch):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        engine.configured_symbols = ["BTCUSD", "ETHUSD", "SOLUSD"]
        async def products():
            return {"BTCUSD", "SOLUSD", "XRPUSD"}
        async def tickers():
            return []
        monkeypatch.setattr(engine.rest, "live_perpetual_symbols", products)
        monkeypatch.setattr(engine.rest, "tickers", tickers)
        assert await engine.validate_symbol_universe() is True
        assert engine.active_symbols == ["BTCUSD", "SOLUSD"]
        assert engine.unavailable_symbols == ["ETHUSD"]
        assert engine.stream.symbols == ["BTCUSD", "SOLUSD"]
        assert engine.symbol_validation["status"] == "partial"
        await engine.rest.close()

    asyncio.run(run())


def test_symbol_validation_fails_closed_on_products_api_error(tmp_path, monkeypatch):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        async def products():
            raise RuntimeError("temporary products API failure")
        monkeypatch.setattr(engine.rest, "live_perpetual_symbols", products)
        assert await engine.validate_symbol_universe() is False
        assert engine.active_symbols == []
        assert engine.stream.symbols == []
        assert engine.symbol_validation["status"] == "error"
        await engine.rest.close()

    asyncio.run(run())



def test_orb_detector_remains_wired_after_normal_removal(tmp_path, monkeypatch):
    import app.signal_engine as signal_engine_module
    from app.models import Candle

    async def run():
        engine, _notifier = make_engine(tmp_path)
        engine.settings.orb_enabled = True
        engine.settings.tpr_enabled = False
        engine.settings.rb_enabled = False
        engine.settings.mex_enabled = False
        now = datetime.now(timezone.utc)
        latest_time = datetime.fromtimestamp((int(now.timestamp()) // 300) * 300 - 300, timezone.utc)
        bars = [Candle(timestamp=latest_time - timedelta(minutes=5 * (59 - i)),
                       open=100.0, high=101.0, low=99.0, close=100.5, volume=100.0)
                for i in range(60)]
        async def fake_candles(_symbol, _timeframe, _limit):
            return bars
        async def no_context(_symbol):
            return None
        def orb_detector(_frame, **_kwargs):
            return {"side": "LONG", "entry_reference": 100.5, "trigger_price": 101.1,
                    "stop_loss": 98.5, "take_profit_1": 103.5, "atr": 2.0, "reason": "orb setup"}
        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        monkeypatch.setattr(signal_engine_module, "evaluate_orb", orb_detector)
        await engine.refresh_symbol("BTCUSD")
        assert ("BTCUSD", "ORB") in engine.momentum_pending
        assert "NORMAL" not in engine.strategy_evaluations
        assert "NORMAL" not in engine.strategy_diagnostics
        assert "NORMAL" not in engine.trigger_lifecycle
        await engine.rest.close()
    asyncio.run(run())


def test_breakout_confirmation_restarts_after_long_gap_between_quotes(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        engine.pending["BTCUSD"] = setup

        first = Quote(price=101.2, bid=101.1, ask=101.2, received_at=now)
        await engine.on_quote("BTCUSD", first, now)
        assert setup.trigger_candidate_at == now

        # A second quote 1.6s later is too far away to prove a continuous hold.
        gap_time = now + timedelta(seconds=1.6)
        after_gap = Quote(
            price=101.2, bid=101.1, ask=101.2, received_at=gap_time
        )
        await engine.on_quote("BTCUSD", after_gap, gap_time)
        assert notifier.alerts == []
        assert setup.trigger_candidate_at == gap_time
        assert setup.trigger_candidate_last_at == gap_time
        assert engine.trigger_lifecycle["TPR"]["events"]["confirmation_gap_reset"] == 1

        # Two timely, distinct quotes after the reset complete confirmation.
        confirmed_at = gap_time + timedelta(seconds=0.6)
        confirmed = Quote(
            price=101.2, bid=101.1, ask=101.2, received_at=confirmed_at
        )
        await engine.on_quote("BTCUSD", confirmed, confirmed_at)
        await wait_for_outbox(engine)
        assert len(notifier.alerts) == 1
        await engine.rest.close()

    asyncio.run(run())



def test_engine_close_releases_shared_rest_client(tmp_path, monkeypatch):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        closed = []

        async def fake_close():
            closed.append(True)

        monkeypatch.setattr(engine.rest, "close", fake_close)
        await engine.close()
        assert closed == [True]

    asyncio.run(run())



def test_rest_quote_snapshot_is_fresh_without_overwriting_exchange_sequence(tmp_path):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        previous_ws_time = now - timedelta(seconds=10)
        engine.last_quote_exchange_at["BTCUSD"] = previous_ws_time
        quote = Quote(
            price=100.5, bid=100.0, ask=101.0,
            received_at=now, source="rest",
        )
        await engine.on_quote("BTCUSD", quote, now)
        assert engine.quotes["BTCUSD"] is quote
        assert engine.received_at["BTCUSD"] == now
        assert engine.last_quote_exchange_at["BTCUSD"] == previous_ws_time
        await engine.rest.close()
    asyncio.run(run())


def test_terminal_setup_lifecycle_is_safe_dedupe_not_persistence_failure():
    class TerminalStore:
        def __init__(self):
            self.transitions = []

        def transition_setup(self, *args, **kwargs):
            self.transitions.append((args, kwargs))
            return False

        def setup_lifecycle(self, *args):
            return {"state": "INVALIDATED"}

    class JournalMustNotBeUpdated:
        def update_lifecycle(self, *args, **kwargs):
            raise AssertionError("terminal dedupe must not write a fake journal lifecycle")

    engine = SignalEngine.__new__(SignalEngine)
    engine.settings = SimpleNamespace(signal_max_age_bars=4)
    engine.context_15m = {}
    engine.store = TerminalStore()
    engine.journal = JournalMustNotBeUpdated()
    engine.pending = {}
    engine.momentum_pending = {}
    engine.trigger_lifecycle = {
        "ORB": {"events": {}, "last_event": None},
    }

    candle = SimpleNamespace(
        timestamp=datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc),
        high=101.0, low=99.0, close=100.0,
    )
    result = {
        "side": "LONG", "trigger_price": 101.0, "entry_reference": 100.0,
        "stop_loss": 98.0, "take_profit_1": 104.0, "atr": 1.0,
        "reason": "test_setup",
    }

    engine._arm_setup("BTCUSD", "ORB", result, candle)

    assert len(engine.store.transitions) == 1
    assert engine.pending == {}
    assert engine.momentum_pending == {}
    assert engine.trigger_lifecycle["ORB"]["events"] == {
        "setup_lifecycle_already_terminal": 1
    }


def test_structural_risk_cap_log_exposes_entry_stop_and_risk_atr(tmp_path, caplog):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        setup.trigger_price = 100.0
        setup.stop_loss = 98.0
        setup.atr = 0.5
        engine.pending["BTCUSD"] = setup
        first = Quote(price=100.1, bid=100.0, ask=100.1, received_at=now)
        await engine.on_quote("BTCUSD", first, now)
        later = now + timedelta(seconds=0.6)
        second = Quote(price=100.1, bid=100.0, ask=100.1, received_at=later)
        with caplog.at_level("INFO", logger="app.signal_engine"):
            await engine.on_quote("BTCUSD", second, later)
        assert notifier.alerts == []
        assert "risk_atr=4.2000" in caplog.text
        assert "entry=100.1" in caplog.text
        assert "stop_loss=98" in caplog.text
        assert "max_risk_atr=3.0" in caplog.text
        await engine.rest.close()
    asyncio.run(run())



def test_rest_snapshots_never_start_or_advance_breakout_confirmation(tmp_path):
    async def run():
        engine, notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc)
        setup = make_setup("LONG", now)
        engine.pending["BTCUSD"] = setup

        # Repeated REST snapshots may refresh the cached quote, but each poll
        # receives a new local timestamp and must not count as a new market event.
        for seconds in (0.0, 0.6, 1.2):
            received = now + timedelta(seconds=seconds)
            quote = Quote(
                price=101.2, bid=101.1, ask=101.2,
                received_at=received, source="rest",
            )
            await engine.on_quote("BTCUSD", quote, received)

        assert notifier.alerts == []
        assert setup.trigger_candidate_at is None
        assert setup.trigger_candidate_last_at is None

        # A genuine exchange-published update can start the confirmation window.
        await engine.on_quote(
            "BTCUSD",
            Quote(price=101.2, bid=101.1, ask=101.2, received_at=now),
            now,
        )
        assert setup.trigger_candidate_at == now

        # A REST poll after the hold interval still cannot complete it.
        rest_time = now + timedelta(seconds=0.6)
        await engine.on_quote(
            "BTCUSD",
            Quote(
                price=101.2, bid=101.1, ask=101.2,
                received_at=rest_time, source="rest",
            ),
            rest_time,
        )
        assert notifier.alerts == []
        assert setup.trigger_candidate_last_at == now

        # A second distinct WebSocket update completes confirmation.
        confirmed_at = now + timedelta(seconds=0.6)
        await engine.on_quote(
            "BTCUSD",
            Quote(price=101.2, bid=101.1, ask=101.2, received_at=confirmed_at),
            confirmed_at,
        )
        await wait_for_outbox(engine)
        assert len(notifier.alerts) == 1
        await engine.rest.close()

    asyncio.run(run())


def test_rest_quote_fallback_skips_poll_when_all_websocket_quotes_are_fresh():
    from app.signal_engine import _rest_quote_fallback_needed

    now = datetime.now(timezone.utc)
    received = {"BTCUSD": now - timedelta(seconds=0.2), "ETHUSD": now - timedelta(seconds=1.0)}
    assert not _rest_quote_fallback_needed(
        ["BTCUSD", "ETHUSD"], received, now, stale_after_seconds=1.5,
    )


def test_rest_quote_fallback_polls_when_any_symbol_quote_is_stale_or_missing():
    from app.signal_engine import _rest_quote_fallback_needed

    now = datetime.now(timezone.utc)
    received = {"BTCUSD": now - timedelta(seconds=0.2)}
    assert _rest_quote_fallback_needed(
        ["BTCUSD", "ETHUSD"], received, now, stale_after_seconds=1.5,
    )
    received["ETHUSD"] = now - timedelta(seconds=2.0)
    assert _rest_quote_fallback_needed(
        ["BTCUSD", "ETHUSD"], received, now, stale_after_seconds=1.5,
    )

    
def test_rest_quote_fallback_targets_only_stale_symbols():
    from app.signal_engine import _stale_quote_symbols

    now = datetime.now(timezone.utc)
    received = {
        "BTCUSD": now - timedelta(seconds=0.2),
        "ETHUSD": now - timedelta(seconds=2.0),
    }
    assert _stale_quote_symbols(
        ["BTCUSD", "ETHUSD", "SOLUSD"], received, now, stale_after_seconds=1.5,
    ) == {"ETHUSD", "SOLUSD"}


def test_advika_uses_240_bars_without_changing_legacy_120_bar_inputs(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        engine.settings.advika_enabled = True
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(
                timestamp=latest_time - timedelta(minutes=5 * (239 - i)),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0,
            )
            for i in range(240)
        ]
        received = {"legacy": [], "advika": []}

        async def fake_candles(_symbol, _timeframe, _limit, *, since=None):
            assert _limit == 240
            return bars

        async def no_context(_symbol):
            return None

        def legacy_evaluator(frame, **_kwargs):
            received["legacy"].append(len(frame))
            return None

        def advika_evaluator(frame, **_kwargs):
            received["advika"].append(len(frame))
            return None

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(engine, "refresh_context", no_context)
        monkeypatch.setattr(signal_engine_module, "evaluate_tpr", legacy_evaluator)
        monkeypatch.setattr(signal_engine_module, "evaluate_rb", legacy_evaluator)
        monkeypatch.setattr(signal_engine_module, "evaluate_advika_choice", advika_evaluator)
        await engine.refresh_symbol("BTCUSD")
        assert received["legacy"] == [120, 120]
        assert received["advika"] == [240]
        await engine.rest.close()

    asyncio.run(run())


def test_large_symbol_universe_keeps_top_100_by_delta_turnover(tmp_path, monkeypatch):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        engine.configured_symbols = [f"COIN{i:03d}USD" for i in range(120)]
        engine.configured_symbol_set = frozenset(engine.configured_symbols)
        async def products():
            return set(engine.configured_symbols)
        async def tickers():
            return [
                {"symbol": symbol, "turnover_usd": str(index + 1)}
                for index, symbol in enumerate(engine.configured_symbols)
            ]
        monkeypatch.setattr(engine.rest, "live_perpetual_symbols", products)
        monkeypatch.setattr(engine.rest, "tickers", tickers)
        assert await engine.validate_symbol_universe() is True
        assert len(engine.active_symbols) == 100
        assert engine.active_symbols == engine.configured_symbols[20:]
        assert engine.symbol_validation["liquidity_filtered_count"] == 20
        assert engine.symbol_validation["liquidity_filtered_symbols"] == engine.configured_symbols[:20]
        await engine.rest.close()
    asyncio.run(run())


def test_large_symbol_universe_fails_closed_if_ticker_snapshot_incomplete(tmp_path, monkeypatch):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        engine.configured_symbols = [f"COIN{i:03d}USD" for i in range(120)]
        async def products():
            return set(engine.configured_symbols)
        async def tickers():
            return [{"symbol": "COIN000USD", "turnover_usd": 1000}]
        monkeypatch.setattr(engine.rest, "live_perpetual_symbols", products)
        monkeypatch.setattr(engine.rest, "tickers", tickers)
        assert await engine.validate_symbol_universe() is False
        assert engine.active_symbols == []
        assert engine.stream.symbols == []
        assert engine.symbol_validation["status"] == "error"
        await engine.rest.close()
    asyncio.run(run())


def test_engine_subscribes_to_delta_completed_candle_channel(tmp_path):
    engine, _notifier = make_engine(tmp_path)
    assert engine.stream.on_candle == engine.on_candle
    assert "candlestick_5m" in engine.stream.required_channels
    asyncio.run(engine.rest.close())


def test_live_candles_only_merge_after_their_five_minute_bucket_closes(tmp_path):
    engine, _notifier = make_engine(tmp_path)
    now = datetime.now(timezone.utc)
    closed_at = now - timedelta(minutes=10)
    forming_at = now - timedelta(minutes=2)
    closed = Candle(closed_at, 100, 102, 99, 101, 15)
    forming = Candle(forming_at, 101, 103, 100, 102, 8)

    async def run():
        await engine.on_candle("BTCUSD", closed, now)
        await engine.on_candle("BTCUSD", forming, now)
        merged = engine._merge_closed_live_candles("BTCUSD", [], 240)
        assert [bar.timestamp for bar in merged] == [closed_at]
        assert merged[0].volume == 15
        await engine.rest.close()

    asyncio.run(run())


def test_live_candle_cache_is_bounded_and_unknown_symbols_are_ignored(tmp_path):
    engine, _notifier = make_engine(tmp_path)
    now = datetime.now(timezone.utc)

    async def run():
        await engine.on_candle("UNKNOWNUSD", Candle(now, 100, 101, 99, 100, 1), now)
        assert "UNKNOWNUSD" not in engine.live_candles
        for index in range(ADVIKA_HISTORY_BARS + 61):
            stamp = now - timedelta(minutes=5 * (ADVIKA_HISTORY_BARS + 61 - index))
            await engine.on_candle("BTCUSD", Candle(stamp, 100, 101, 99, 100, 1), now)
        assert len(engine.live_candles["BTCUSD"]) == ADVIKA_HISTORY_BARS + 60
        await engine.rest.close()

    from app.advika_choice import HISTORY_BARS as ADVIKA_HISTORY_BARS
    asyncio.run(run())


def test_empty_incremental_candle_response_keeps_fresh_cache_without_full_refetch(tmp_path, monkeypatch):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest = now.replace(minute=(now.minute // 5) * 5) - timedelta(minutes=5)
        bars = [
            Candle(
                timestamp=latest - timedelta(minutes=5 * (119 - index)),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0,
            )
            for index in range(120)
        ]
        engine.candles["BTCUSD"] = bars
        engine.last_evaluated["BTCUSD"] = latest
        calls = []

        async def empty_incremental(symbol, timeframe, limit, *, since=None):
            calls.append({"symbol": symbol, "timeframe": timeframe, "limit": limit, "since": since})
            assert since is not None
            return []

        monkeypatch.setattr(engine.rest, "candles", empty_incremental)
        await engine.refresh_symbol("BTCUSD")

        assert len(calls) == 1
        assert engine.candles["BTCUSD"] == bars
        assert "BTCUSD" not in engine.force_full_candle_refresh
        assert engine.candle_diagnostics["BTCUSD"]["state"] == "ready"
        await engine.rest.close()

    asyncio.run(run())


def test_advika_position_manager_queues_partial_and_final_exit_once(tmp_path, monkeypatch):
    import app.signal_engine as signal_engine_module

    engine, _notifier = make_engine(tmp_path)
    setup_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    position = {
        "signal_key": "BTCUSD:ADVIKA:LONG:test",
        "symbol": "BTCUSD", "side": "LONG", "entry_price": 100.0,
        "stop_loss": 98.0, "take_profit_1": 103.0, "atr": 2.0,
        "trailing_stop": 98.0, "last_notified_trailing_stop": 98.0,
        "partial_taken": False, "setup_time": setup_time.isoformat(),
        "status": "open", "last_processed_candle": setup_time.isoformat(),
    }

    class PositionStore:
        def __init__(self):
            self.positions = [dict(position)]
            self.alerts = {}

        def list_open_advika_positions(self, symbol):
            return [dict(p) for p in self.positions if p["symbol"] == symbol and p["status"] in {"open", "partial"}]

        def enqueue_alert(self, key, payload):
            if key in self.alerts:
                return False
            self.alerts[key] = payload
            return True

        def update_advika_position(self, key, updated, status, candle_time):
            current = next(p for p in self.positions if p["signal_key"] == key)
            current.update(updated)
            current["status"] = status
            current["last_processed_candle"] = candle_time
            return True

    store = PositionStore()
    engine.store = store
    delivered = []

    async def drain_outbox():
        delivered.append(True)

    engine.deliver_outbox = drain_outbox
    calls = {"count": 0}

    def fake_exit(_position, _frame):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "action": "TAKE_PARTIAL_50", "reason": "target_1_5r_reached",
                "trailing_stop": 101.0,
            }
        if calls["count"] == 2:
            return {"action": "UPDATE_TRAILING_STOP", "trailing_stop": 102.0}
        return {"action": "EXIT_REMAINDER", "reason": "trailing_stop_touched"}

    monkeypatch.setattr(signal_engine_module, "evaluate_advika_exit", fake_exit)
    frame = pd.DataFrame([{"close": 104.0}])
    first_candle = setup_time + timedelta(minutes=5)
    second_candle = setup_time + timedelta(minutes=10)
    third_candle = setup_time + timedelta(minutes=15)

    async def run():
        await engine._manage_advika_positions(
            "BTCUSD", Candle(first_candle, 103, 105, 102, 104, 20), frame,
        )
        assert store.positions[0]["status"] == "partial"
        assert store.positions[0]["partial_taken"] is True
        assert store.positions[0]["trailing_stop"] == 101.0
        await engine._manage_advika_positions(
            "BTCUSD", Candle(second_candle, 104, 105, 103, 104, 18), frame,
        )
        assert store.positions[0]["trailing_stop"] == 102.0
        await engine._manage_advika_positions(
            "BTCUSD", Candle(third_candle, 104, 105, 100, 102, 18), frame,
        )
        assert store.positions[0]["status"] == "closed"
        assert len(store.alerts) == 3
        assert {p["action"] for p in store.alerts.values()} == {
            "TAKE_PARTIAL_50", "UPDATE_TRAILING_STOP", "EXIT_REMAINDER",
        }
        await wait_for_outbox(engine)
        # One background drain batches all three queued management alerts.
        assert len(delivered) == 1
        # Reprocessing the same closed candle cannot emit another management alert.
        await engine._manage_advika_positions(
            "BTCUSD", Candle(third_candle, 104, 105, 100, 102, 18), frame,
        )
        assert len(store.alerts) == 3

    asyncio.run(run())
    asyncio.run(engine.rest.close())


def test_advika_position_manager_does_not_advance_when_exit_evaluation_is_unavailable(
    tmp_path, monkeypatch,
):
    import app.signal_engine as signal_engine_module

    engine, _notifier = make_engine(tmp_path)
    setup_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    position = {
        "signal_key": "BTCUSD:ADVIKA:LONG:warmup",
        "symbol": "BTCUSD", "side": "LONG", "entry_price": 100.0,
        "stop_loss": 98.0, "take_profit_1": 103.0, "atr": 2.0,
        "trailing_stop": 98.0, "partial_taken": False,
        "setup_time": setup_time.isoformat(), "status": "open",
        "last_processed_candle": setup_time.isoformat(),
    }

    class PositionStore:
        def __init__(self):
            self.position = dict(position)
            self.updates = 0

        def list_open_advika_positions(self, symbol):
            return [dict(self.position)] if symbol == "BTCUSD" else []

        def update_advika_position(self, key, updated, status, candle_time):
            self.updates += 1
            self.position.update(updated)
            self.position["status"] = status
            self.position["last_processed_candle"] = candle_time

    store = PositionStore()
    engine.store = store
    monkeypatch.setattr(signal_engine_module, "evaluate_advika_exit", lambda *_args: None)
    candle_time = setup_time + timedelta(minutes=5)

    async def run():
        await engine._manage_advika_positions(
            "BTCUSD", Candle(candle_time, 100, 101, 99, 100, 10),
            pd.DataFrame([{"close": 100.0}]),
        )
        assert store.updates == 0
        assert store.position["last_processed_candle"] == setup_time.isoformat()
        assert store.position["status"] == "open"
        await engine.rest.close()

    asyncio.run(run())


def test_outbox_delivery_is_scheduled_without_blocking_caller(tmp_path):
    async def run():
        engine, _notifier = make_engine(tmp_path)
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_delivery():
            started.set()
            await release.wait()

        engine.deliver_outbox = blocked_delivery
        engine.schedule_outbox_delivery()
        await started.wait()
        assert engine.outbox_task is not None
        assert not engine.outbox_task.done()
        first_task = engine.outbox_task
        engine.schedule_outbox_delivery()
        assert engine.outbox_task is first_task
        release.set()
        await engine.outbox_task
        await engine.rest.close()

    asyncio.run(run())



def test_advika_readiness_distinguishes_legacy_warmup_from_ema200_warmup(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        engine.settings.advika_enabled = True
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(
                timestamp=latest_time - timedelta(minutes=5 * (59 - index)),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0,
            )
            for index in range(60)
        ]

        async def fake_candles(_symbol, _timeframe, _limit, *, since=None):
            return bars

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(signal_engine_module, "evaluate_advika_choice", lambda *_args, **_kwargs: None)
        await engine.refresh_symbol("BTCUSD")
        health = engine.health()

        assert health["candle_readiness"]["legacy"]["ready_count"] == 1
        assert health["candle_readiness"]["advika"]["enabled"] is True
        assert health["candle_readiness"]["advika"]["required_bars"] == 240
        assert health["candle_readiness"]["advika"]["ready_count"] == 0
        assert health["candle_readiness"]["advika"]["warming_count"] == 1
        assert health["candle_readiness"]["advika"]["not_ready_reasons"] == {
            "BTCUSD": "insufficient_240_bar_history"
        }
        assert health["candle_diagnostics"]["BTCUSD"]["legacy_state"] == "ready"
        assert health["candle_diagnostics"]["BTCUSD"]["advika_state"] == "warming"
        await engine.rest.close()

    asyncio.run(run())


def test_advika_readiness_becomes_ready_only_after_240_contiguous_bars(tmp_path, monkeypatch):
    from app.models import Candle
    import app.signal_engine as signal_engine_module

    async def run():
        engine, _notifier = make_engine(tmp_path)
        engine.settings.advika_enabled = True
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_time = now - timedelta(minutes=5)
        bars = [
            Candle(
                timestamp=latest_time - timedelta(minutes=5 * (239 - index)),
                open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0,
            )
            for index in range(240)
        ]

        async def fake_candles(_symbol, _timeframe, _limit, *, since=None):
            return bars

        monkeypatch.setattr(engine.rest, "candles", fake_candles)
        monkeypatch.setattr(signal_engine_module, "evaluate_advika_choice", lambda *_args, **_kwargs: None)
        await engine.refresh_symbol("BTCUSD")
        health = engine.health()

        assert health["candle_readiness"]["legacy"]["ready_count"] == 1
        assert health["candle_readiness"]["advika"]["ready_count"] == 1
        assert health["candle_readiness"]["advika"]["warming_count"] == 0
        assert health["candle_readiness"]["advika"]["not_ready_reasons"] == {}
        assert health["candle_diagnostics"]["BTCUSD"]["advika_state"] == "ready"
        await engine.rest.close()

    asyncio.run(run())
