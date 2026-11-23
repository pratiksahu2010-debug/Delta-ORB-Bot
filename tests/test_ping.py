from app.main import ping


def test_ping_is_bodyless_200_keepalive_only():
    import asyncio

    response = asyncio.run(ping())
    assert response.status_code == 200
    assert response.body == b""


def test_lifespan_starts_and_closes_engine(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import app.main as main

    class FakeNotifier:
        configured = False
        token = "test-token"

        def __init__(self, token, chat_id):
            self.token = token
            self.deleted_webhooks = 0

        async def delete_webhook(self):
            self.deleted_webhooks += 1
            return True

        async def register_webhook(self, webhook_url, secret):
            raise AssertionError("Telegram webhook registration must stay disabled")

    class FakeEngine:
        def __init__(self, settings, notifier):
            self.notifier = notifier
            self.store = SimpleNamespace(backend="test")
            self.pending_restore = {"restored": 0}
            self.closed = False

        def restore_pending_setups(self):
            pass

        async def run(self):
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True

    fake_settings = SimpleNamespace(
        validate_safety=lambda: None,
        telegram_bot_token="test-token",
        telegram_chat_id="",
        telegram_webhook_secret="configured-but-ignored",
        symbol_list=["BTCUSD"],
        signal_only=True,
        orders_enabled=False,
    )
    monkeypatch.setattr(main, "settings", fake_settings)
    monkeypatch.setattr(main, "TelegramNotifier", FakeNotifier)
    monkeypatch.setattr(main, "SignalEngine", FakeEngine)

    async def exercise_lifespan():
        async with main.app.router.lifespan_context(main.app):
            assert main.engine is not None
            assert main.engine_task is not None
            assert main.engine.notifier.deleted_webhooks == 1
            await asyncio.sleep(0)
        assert main.engine_task is None
        assert main.engine.closed is True
        main.engine = None

    asyncio.run(exercise_lifespan())
