import asyncio
from types import SimpleNamespace

import httpx

import app.main as main


def _healthy_engine():
    snapshot = {
        "websocket_connected": True,
        "websocket_reconnects": 1,
        "websocket_heartbeat_age_seconds": 2.0,
        "websocket_heartbeat_stale": False,
        "candle_ready_count": 1,
        "candle_warming_count": 0,
        "candle_no_data_count": 0,
        "candle_error_count": 0,
        "candle_diagnostics": {"BTCUSD": {"state": "ready"}},
        "quote_missing_symbols": [],
        "quote_stale_symbols": [],
        "candle_gap_list": [],
        "stale_candle_list": [],
        "stale_quotes": 0,
        "stale_candle_symbols": 0,
        "candle_gap_symbols": 0,
        "symbols_with_quotes": 1,
        "symbols_with_candles": 1,
        "active_symbol_count": 1,
        "state_store_ok": True,
        "database_backend": "postgresql",
        "telegram_outbox": {"pending": 0, "sending": 0, "sent": 3, "retrying": 0},
        "strategy_evaluations": {"ORB": {"evaluated": 8}},
        "trigger_lifecycle": {"ORB": {"events": {"setup_armed": 2}}},
        "strategy_diagnostics": {"ORB": {"rejections": 1}},
        "telegram_delivery": {"sent": 3, "failed": 0},
        "pending_setups": 2,
        "pending_setup_recovery": {"restored": 2},
        "last_ws_message_at": "2026-11-21T00:00:00+00:00",
        "last_live_candlestick_at": "2026-11-21T00:00:00+00:00",
    }
    return SimpleNamespace(
        active_symbols=["BTCUSD"],
        health=lambda: dict(snapshot),
    )


def _configure(monkeypatch, engine):
    monkeypatch.setattr(main, "engine", engine)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            symbol_list=["BTCUSD"],
            telegram_bot_token="test-token",
            telegram_chat_id="123456",
            signal_only=True,
            orders_enabled=False,
        ),
    )


def test_ready_and_status_report_healthy_runtime_without_secrets(monkeypatch):
    async def run():
        _configure(monkeypatch, _healthy_engine())
        async def worker():
            await asyncio.Event().wait()
        task = asyncio.create_task(worker())
        monkeypatch.setattr(main, "engine_task", task)
        try:
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                ready = await client.get("/ready")
                status = await client.get("/status")
            assert ready.status_code == 200
            assert ready.json()["ready"] is True
            assert status.status_code == 200
            assert status.json()["status"] == "ready"
            assert status.json()["persistence"]["database_backend"] == "postgresql"
            assert status.json()["telegram_outbox"]["sent"] == 3
            assert "test-token" not in status.text
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run())


def test_ready_stays_200_when_market_data_is_stale_and_status_reports_degraded(monkeypatch):
    async def run():
        engine = _healthy_engine()
        original = engine.health

        def stale_health():
            data = original()
            data["quote_stale_symbols"] = ["BTCUSD"]
            return data

        engine.health = stale_health
        _configure(monkeypatch, engine)
        async def worker():
            await asyncio.Event().wait()
        task = asyncio.create_task(worker())
        monkeypatch.setattr(main, "engine_task", task)
        try:
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/ready")
                status = await client.get("/status")
            assert response.status_code == 200
            assert response.json()["ready"] is True
            assert status.status_code == 200
            assert status.json()["status"] == "degraded"
            assert "active_quotes_fresh" in status.json()["readiness"]["reasons"]
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run())


def test_ready_returns_503_before_engine_starts(monkeypatch):
    async def run():
        monkeypatch.setattr(main, "engine", None)
        monkeypatch.setattr(main, "engine_task", None)
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json()["reasons"] == ["engine_not_started"]

    asyncio.run(run())


def test_ready_does_not_depend_on_market_data_health_snapshot(monkeypatch):
    async def run():
        engine = SimpleNamespace(
            active_symbols=["BTCUSD"],
            health=lambda: (_ for _ in ()).throw(RuntimeError("private connection detail")),
        )
        _configure(monkeypatch, engine)
        async def worker():
            await asyncio.Event().wait()
        task = asyncio.create_task(worker())
        monkeypatch.setattr(main, "engine_task", task)
        try:
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/ready")
            assert response.status_code == 200
            assert response.json()["ready"] is True
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run())



def test_status_preserves_nested_mex_gate_diagnostics(monkeypatch):
    async def run():
        engine = _healthy_engine()
        sample = {
            "symbol": "BTCUSD",
            "candle_at": "2026-11-21T10:00:00+00:00",
            "stage": "participation_and_expansion",
            "reason": "volume_or_atr_expansion_below_minimum",
            "values": {
                "rvol": 0.8,
                "min_rvol": 1.2,
                "atr_expansion": 0.9,
                "min_atr_expansion": 1.1,
                "failed_filters": ["rvol", "atr_expansion"],
            },
        }
        snapshot = engine.health()
        snapshot["strategy_diagnostics"] = {
            "ORB": {
                "rejections": 4,
                "setup_types": {
                    "ORB": {"detail_counts": {}, "recent_rejections": []},
                    "RB": {"detail_counts": {}, "recent_rejections": []},
                    "TPR": {"detail_counts": {}, "recent_rejections": []},
                },
            },
            "MEX": {
                "rejections": 25,
                "setup_types": {
                    "MEX": {
                        "rejections": 25,
                        "rejections_by_reason": {
                            "volume_or_atr_expansion_below_minimum": 25
                        },
                        "detail_counts": {"rvol": 18, "atr_expansion": 12},
                        "recent_rejections": [sample],
                        "last_rejection": sample,
                    }
                },
            },
        }
        engine.health = lambda: snapshot
        _configure(monkeypatch, engine)

        async def worker():
            await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        monkeypatch.setattr(main, "engine_task", task)
        try:
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/status")
            assert response.status_code == 200
            diagnostics = response.json()["strategy_diagnostics"]
            mex = diagnostics["MEX"]["setup_types"]["MEX"]
            assert mex["detail_counts"] == {"rvol": 18, "atr_expansion": 12}
            assert mex["recent_rejections"] == [sample]
            assert mex["last_rejection"]["values"]["failed_filters"] == [
                "rvol", "atr_expansion"
            ]
            assert set(diagnostics) == {"ORB", "MEX"}
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run())
