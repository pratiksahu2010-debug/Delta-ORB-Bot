import asyncio

import httpx

from app.notifier import TelegramNotifier


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}" if payload is not None else b""

    def json(self):
        return self._payload


class FakeAsyncClient:
    response = FakeResponse(200, {"ok": True})
    last_request = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aclose(self):
        return None

    async def post(self, *args, **kwargs):
        FakeAsyncClient.last_request = {"args": args, "kwargs": kwargs}
        return self.response


def test_telegram_notifier_accepts_only_successful_telegram_api_response(monkeypatch):
    async def run():
        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
        notifier = TelegramNotifier("test-token", "test-chat")
        signal = {
            "symbol": "TEST", "strategy": "PULLBACK", "side": "LONG", "entry": 100,
            "stop_loss": 98, "take_profit_1": 103,
            "setup_time": "2026-11-17T00:00:00+00:00", "reason": "test",
        }
        FakeAsyncClient.response = FakeResponse(200, {"ok": True})
        assert await notifier.send_signal(signal) is True
        assert notifier.last_delivery["status"] == "sent"
        payload = FakeAsyncClient.last_request["kwargs"]["json"]
        assert payload["parse_mode"] == "HTML"
        assert "reply_markup" not in payload
        assert "🟢 <b>TEST | NORMAL</b>" in payload["text"]
        assert "<b>Side: LONG</b>" in payload["text"]
        assert "<b>Entry: 100</b>" in payload["text"]
        assert "<b>Stop Loss: 98</b>" in payload["text"]
        assert "<b>Target (2R): 103</b>" in payload["text"]
        assert '<a href="https://www.tradingview.com/chart/?symbol=DELTAIN%3ATEST.P&amp;interval=5">📊 OPEN TRADINGVIEW CHART (5m)</a>' in payload["text"]
        assert "<b>Signal only — no order placed.</b>" in payload["text"]

        FakeAsyncClient.response = FakeResponse(200, {"ok": False, "description": "bot was blocked"})
        assert await notifier.send_signal(signal) is False
        assert notifier.last_delivery["status"] == "failed"
        assert "bot was blocked" in notifier.last_delivery["error"]

        FakeAsyncClient.response = FakeResponse(401, {})
        assert await notifier.send_signal(signal) is False
        assert notifier.last_delivery["http_status"] == 401

    asyncio.run(run())


def test_strategy_labels_and_direction_dots():
    long_normal = TelegramNotifier._format_signal({"symbol": "BTCUSD", "strategy": "PULLBACK", "side": "LONG"})
    short_mex = TelegramNotifier._format_signal({"symbol": "ETHUSD", "strategy": "MEX", "side": "SHORT"})
    long_orb = TelegramNotifier._format_signal({"symbol": "SOLUSD", "strategy": "ORB", "side": "LONG"})
    assert "🟢 <b>BTCUSD | NORMAL</b>" in long_normal
    assert "🔴 <b>ETHUSD | MEX</b>" in short_mex
    assert "🟢 <b>SOLUSD | ORB</b>" in long_orb
    assert "DELTAIN%3ABTCUSD.P" in long_normal
    assert "DELTAIN%3AETHUSD.P" in short_mex
    assert "DELTAIN%3ASOLUSD.P" in long_orb


def test_tradingview_chart_url_normalizes_delta_perpetual_symbols():
    assert TelegramNotifier.tradingview_chart_url("BTCUSD") == (
        "https://www.tradingview.com/chart/?symbol=DELTAIN%3ABTCUSD.P&interval=5"
    )
    assert TelegramNotifier.tradingview_chart_url("DELTAIN:ETHUSD.P") == (
        "https://www.tradingview.com/chart/?symbol=DELTAIN%3AETHUSD.P&interval=5"
    )


def test_signal_html_escapes_dynamic_content():
    message = TelegramNotifier._format_signal({
        "symbol": "BTC<USD", "strategy": "MEX", "side": "LONG",
        "reason": "<script>alert(1)</script>",
    })
    assert "BTC&lt;USD" in message
    assert "&lt;script&gt;" in message
    assert "<script>" not in message


def test_telegram_notifier_fails_without_configuration():
    async def run():
        notifier = TelegramNotifier("", "")
        assert await notifier.send_signal({"symbol": "TEST"}) is False
        assert notifier.last_delivery["status"] == "not_configured"

    asyncio.run(run())



def test_signal_includes_inline_review_buttons_when_token_is_supplied(monkeypatch):
    async def run():
        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
        notifier = TelegramNotifier("test-token", "test-chat")
        FakeAsyncClient.response = FakeResponse(200, {"ok": True})
        signal = {"symbol": "BTCUSD", "strategy": "MEX", "side": "LONG"}
        assert await notifier.send_signal(signal, feedback_token="abc123") is True
        keyboard = FakeAsyncClient.last_request["kwargs"]["json"]["reply_markup"]["inline_keyboard"]
        labels = [[button["text"] for button in row] for row in keyboard]
        assert labels == [["👍 Good", "👎 Bad", "⏰ Late"], ["🚫 Missed", "🔎 Needs Review"]]
        assert all(button["callback_data"].endswith("|abc123") for row in keyboard for button in row)

    asyncio.run(run())



def test_delete_webhook_calls_telegram_delete_method(monkeypatch):
    async def run():
        monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
        notifier = TelegramNotifier("test-token", "test-chat")
        FakeAsyncClient.response = FakeResponse(200, {"ok": True})
        assert await notifier.delete_webhook() is True
        assert FakeAsyncClient.last_request["args"][0].endswith("/deleteWebhook")
        assert FakeAsyncClient.last_request["kwargs"]["json"] == {}

    asyncio.run(run())
