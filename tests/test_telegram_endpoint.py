import asyncio
from types import SimpleNamespace

import httpx

import app.main as main


class FakeNotifierForEndpoint:
    def __init__(self):
        self.calls = 0
        self.last_delivery = {"status": "sent", "http_status": 200}

    async def send_signal(self, signal):
        self.calls += 1
        assert signal["symbol"] == "TELEGRAM-TEST"
        assert signal["strategy"] == "SYSTEM_TEST"
        return True


def test_telegram_test_endpoint_requires_configured_secret(monkeypatch):
    async def run():
        monkeypatch.setattr(main.settings, "telegram_test_key", "")
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/admin/telegram-test")
        assert response.status_code == 503

    asyncio.run(run())


def test_telegram_test_endpoint_authenticates_and_sends_one_test(monkeypatch):
    async def run():
        key = "a-long-test-key-used-only-in-this-test"
        notifier = FakeNotifierForEndpoint()
        monkeypatch.setattr(main.settings, "telegram_test_key", key)
        monkeypatch.setattr(main, "engine", SimpleNamespace(notifier=notifier))
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            denied = await client.post("/admin/telegram-test", headers={"X-Telegram-Test-Key": "wrong"})
            accepted = await client.post("/admin/telegram-test", headers={"X-Telegram-Test-Key": key})
        assert denied.status_code == 401
        assert accepted.status_code == 200
        assert accepted.json()["status"] == "sent"
        assert notifier.calls == 1

    asyncio.run(run())
