from types import SimpleNamespace
import asyncio

from app.market_data import DeltaRestClient


def test_rest_ticker_snapshot_fetches_live_perpetual_tickers():
    async def run():
        client = DeltaRestClient("https://api.india.delta.exchange")
        calls = []

        async def fake_get(path, params):
            calls.append((path, dict(params)))
            return SimpleNamespace(json=lambda: {
                "success": True,
                "result": [
                    {"symbol": "BTCUSD", "quotes": {"best_bid": "100", "best_ask": "101"}}
                ],
            })

        client._get_with_retry = fake_get
        rows = await client.tickers()
        assert rows[0]["symbol"] == "BTCUSD"
        assert calls == [("/v2/tickers", {"contract_types": "perpetual_futures"})]
        await client.close()

    asyncio.run(run())
