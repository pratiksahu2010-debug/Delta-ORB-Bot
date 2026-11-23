from datetime import datetime, timezone
from app.market_data import parse_rest_candles, _utc_from_epoch


def test_rest_parser_excludes_forming_candle_and_sorts():
    now = datetime.fromtimestamp(1800000300, timezone.utc)
    rows = [
        {"time": 1800000000, "open": 10, "high": 12, "low": 9, "close": 11, "volume": 5},
        {"time": 1800000300, "open": 11, "high": 13, "low": 10, "close": 12, "volume": 4},
    ]
    result = parse_rest_candles({"success": True, "result": rows}, "5m", now)
    assert len(result) == 1
    assert result[0].timestamp == datetime.fromtimestamp(1800000000, timezone.utc)


def test_microsecond_epoch_normalizes_to_utc():
    assert _utc_from_epoch(1775814834503627).year == 2026


def test_invalid_resolution_is_not_silently_accepted():
    try:
        parse_rest_candles([], "2m", datetime.now(timezone.utc))
    except KeyError:
        pass
    else:
        raise AssertionError("unsupported resolution must fail")



def test_stream_symbol_index_tracks_subscription_updates():
    import asyncio
    from app.market_data import DeltaPublicStream

    received = []

    async def on_quote(symbol, quote, local_received):
        received.append(symbol)

    stream = DeltaPublicStream(["BTCUSD"], on_quote)
    stream.set_symbols(["ETHUSD"])
    asyncio.run(stream._handle_message({
        "d": [
            {"sy": "BTCUSD", "bp": "100", "ap": "101", "ts": 1775037882748105},
            {"sy": "ETHUSD", "bp": "200", "ap": "201", "ts": 1775037882748105},
        ]
    }))
    assert stream.symbol_set == frozenset({"ETHUSD"})
    assert received == ["ETHUSD"]


def test_ob_l1_message_preserves_bid_ask_and_publish_time():
    import asyncio
    from app.market_data import DeltaPublicStream

    received = []

    async def on_quote(symbol, quote, local_received):
        received.append((symbol, quote, local_received))

    stream = DeltaPublicStream(["BTCUSD"], on_quote)
    asyncio.run(stream._handle_message({
        "type": "ob_l1", "sy": "BTCUSD", "bp": "100.0", "ap": "100.5",
        "lts": 1775037882675402, "ts": 1775037882748105,
    }))
    assert len(received) == 1
    symbol, quote, local_received = received[0]
    assert symbol == "BTCUSD"
    assert quote.price == 100.25
    assert quote.bid == 100.0 and quote.ask == 100.5
    assert quote.received_at.timestamp() == 1775037882.748105
    assert local_received.tzinfo == timezone.utc


def test_ob_l1_rejects_crossed_or_nonpositive_book():
    import asyncio
    from app.market_data import DeltaPublicStream

    received = []

    async def on_quote(symbol, quote, local_received):
        received.append(quote)

    stream = DeltaPublicStream(["BTCUSD"], on_quote)
    asyncio.run(stream._handle_message({
        "type": "ob_l1", "sy": "BTCUSD", "bp": "101", "ap": "100",
        "ts": 1775037882748105,
    }))
    assert received == []


def test_rest_parser_skips_malformed_and_nonfinite_rows():
    now = datetime.fromtimestamp(1800000600, timezone.utc)
    rows = [
        {"time": 1800000000, "open": 10, "high": 12, "low": 9, "close": 11, "volume": 5},
        {"time": 1800000300, "open": "bad", "high": 12, "low": 9, "close": 11, "volume": 5},
        {"time": 1800000600, "open": 11, "high": float("nan"), "low": 10, "close": 12, "volume": 5},
    ]
    result = parse_rest_candles({"success": True, "result": rows}, "5m", now)
    assert len(result) == 1
    assert result[0].open == 10



def test_stream_disconnect_callback_runs_when_connection_ends():
    import asyncio
    import json
    from unittest.mock import patch
    from app.market_data import DeltaPublicStream

    calls = []
    async def on_quote(*_args):
        pass
    def on_disconnect():
        calls.append("disconnected")

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
        async def send(self, payload):
            self.sent.append(payload)
        async def recv(self):
            if not hasattr(self, "ack_sent"):
                self.ack_sent = True
                return json.dumps({"type": "subscriptions", "channels": [
                    {"name": "ob_l1", "symbols": ["BTCUSD"]}
                ]})
            raise StopAsyncIteration

    class FakeConnection:
        def __init__(self):
            self.ws = FakeWebSocket()
        async def __aenter__(self):
            return self.ws
        async def __aexit__(self, *_args):
            return False

    async def run():
        stream = DeltaPublicStream(["BTCUSD"], on_quote, on_disconnect)
        connection = FakeConnection()
        connect_options = {}
        def fake_connect(*args, **kwargs):
            connect_options.update(kwargs)
            return connection
        with patch("app.market_data.websockets.connect", side_effect=fake_connect), patch(
            "app.market_data.asyncio.sleep", side_effect=asyncio.CancelledError
        ):
            try:
                await stream.run_forever()
            except asyncio.CancelledError:
                pass
        assert calls == ["disconnected"]
        assert connect_options["open_timeout"] == 45
        assert connect_options["ping_interval"] is None
        assert connect_options["ping_timeout"] is None
        assert any('"type": "enable_heartbeat"' in payload for payload in connection.ws.sent)

    asyncio.run(run())


def test_malformed_quote_item_does_not_drop_other_items_in_batch():
    import asyncio
    from app.market_data import DeltaPublicStream

    received = []

    async def on_quote(symbol, quote, local_received):
        received.append((symbol, quote))

    stream = DeltaPublicStream(["BTCUSD", "ETHUSD"], on_quote)
    asyncio.run(stream._handle_message({
        "d": [
            {"sy": "BTCUSD", "bp": "not-a-number", "ap": "101", "ts": 1775037882748105},
            {"sy": "ETHUSD", "bp": "200", "ap": "201", "ts": 1775037882748105},
        ]
    }))
    assert len(received) == 1
    assert received[0][0] == "ETHUSD"
    assert received[0][1].bid == 200.0
    assert received[0][1].ask == 201.0


def test_rest_candle_request_retries_transient_http_errors(monkeypatch):
    import asyncio
    import httpx
    from app.market_data import DeltaRestClient

    async def run():
        client = DeltaRestClient("https://api.india.delta.exchange")
        request = httpx.Request("GET", "https://api.india.delta.exchange/test")
        responses = [
            httpx.Response(503, request=request),
            httpx.Response(200, json={"success": True, "result": []}, request=request),
        ]
        calls = []

        async def fake_get(_path, params=None):
            calls.append(params)
            return responses.pop(0)

        async def no_wait(_delay):
            return None

        monkeypatch.setattr(client.client, "get", fake_get)
        monkeypatch.setattr("app.market_data.asyncio.sleep", no_wait)
        monkeypatch.setattr("app.market_data.parse_rest_candles", lambda *_args: [])
        assert await client.candles("BTCUSD", "5m", 60) == []
        assert len(calls) == 2
        await client.close()

    asyncio.run(run())



def test_rest_candle_request_supports_incremental_start(monkeypatch):
    import asyncio
    import httpx
    from app.market_data import DeltaRestClient

    async def run():
        client = DeltaRestClient("https://api.india.delta.exchange")
        request = httpx.Request("GET", "https://api.india.delta.exchange/test")
        calls = []

        async def fake_get(_path, params=None):
            calls.append(params)
            return httpx.Response(200, json={"success": True, "result": []}, request=request)

        monkeypatch.setattr(client.client, "get", fake_get)
        monkeypatch.setattr("app.market_data.parse_rest_candles", lambda *_args: [])
        start = int(__import__("time").time()) - 1000
        assert await client.candles("BTCUSD", "5m", 120, since=start) == []
        assert calls[0]["start"] == start
        assert calls[0]["symbol"] == "BTCUSD"
        assert calls[0]["resolution"] == "5m"
        await client.close()

    asyncio.run(run())


def test_rest_candle_request_does_not_retry_permanent_client_errors(monkeypatch):
    import asyncio
    import httpx
    import pytest
    from app.market_data import DeltaRestClient

    async def run():
        client = DeltaRestClient("https://api.india.delta.exchange")
        calls = []

        async def forbidden(_path, params=None):
            calls.append(params)
            return httpx.Response(401, request=httpx.Request("GET", "https://api.india.delta.exchange/test"))

        async def unexpected_wait(_delay):
            raise AssertionError("permanent 4xx must not be retried")

        monkeypatch.setattr(client.client, "get", forbidden)
        monkeypatch.setattr("app.market_data.asyncio.sleep", unexpected_wait)
        with pytest.raises(httpx.HTTPStatusError):
            await client.candles("BTCUSD", "5m", 60)
        assert len(calls) == 1
        await client.close()

    asyncio.run(run())


def test_rest_candle_request_retries_transport_errors(monkeypatch):
    import asyncio
    import httpx
    from app.market_data import DeltaRestClient

    async def run():
        client = DeltaRestClient("https://api.india.delta.exchange")
        calls = []

        async def flaky(_path, params=None):
            calls.append(params)
            if len(calls) == 1:
                raise httpx.ConnectError("temporary network failure")
            return httpx.Response(200, json={"success": True, "result": []},
                                  request=httpx.Request("GET", "https://api.india.delta.exchange/test"))

        async def no_wait(_delay):
            return None

        monkeypatch.setattr(client.client, "get", flaky)
        monkeypatch.setattr("app.market_data.asyncio.sleep", no_wait)
        monkeypatch.setattr("app.market_data.parse_rest_candles", lambda *_args: [])
        assert await client.candles("BTCUSD", "5m", 60) == []
        assert len(calls) == 2
        await client.close()

    asyncio.run(run())




def test_inactive_socket_timeout_disconnects_and_reconnects_stream(monkeypatch):
    import asyncio
    import json
    from unittest.mock import patch
    from app.market_data import DeltaPublicStream

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.recv_count = 0
        async def send(self, payload):
            self.sent.append(json.loads(payload))
        def recv(self):
            # Return Futures rather than coroutine objects so the mocked
            # wait_for timeout path can cancel an outstanding read without
            # leaving an unawaited coroutine behind.
            loop = asyncio.get_running_loop()
            pending = loop.create_future()
            self.recv_count += 1
            if self.recv_count == 1:
                pending.set_result(json.dumps({"type": "subscriptions", "channels": [
                    {"name": "ob_l1", "symbols": ["BTCUSD"]}
                ]}))
            return pending

    class FakeConnection:
        def __init__(self):
            self.ws = FakeWebSocket()
        async def __aenter__(self):
            return self.ws
        async def __aexit__(self, *_args):
            return False

    async def run():
        connection = FakeConnection()
        disconnected = []

        async def on_quote(*_args):
            pass

        def on_disconnect():
            disconnected.append(True)

        stream = DeltaPublicStream(["BTCUSD"], on_quote, on_disconnect)

        async def force_heartbeat_timeout(awaitable, timeout):
            if timeout <= 10.0:
                return await awaitable
            assert 34.5 <= timeout <= 35.0
            awaitable.cancel()
            raise asyncio.TimeoutError

        with patch("app.market_data.websockets.connect", return_value=connection), patch(
            "app.market_data.asyncio.wait_for", side_effect=force_heartbeat_timeout
        ), patch("app.market_data.asyncio.sleep", side_effect=asyncio.CancelledError):
            try:
                await stream.run_forever()
            except asyncio.CancelledError:
                pass

        assert connection.ws.sent[0]["type"] == "subscribe"
        assert connection.ws.sent[1]["type"] == "enable_heartbeat"
        assert stream.reconnects == 1
        assert stream.connected is False
        assert disconnected == [True]

    asyncio.run(run())


def test_heartbeat_frame_is_not_parsed_as_market_quote():
    import asyncio
    from app.market_data import DeltaPublicStream

    received = []

    async def on_quote(*args):
        received.append(args)

    stream = DeltaPublicStream(["BTCUSD"], on_quote)
    asyncio.run(stream._handle_message({"type": "heartbeat"}))
    assert received == []



def test_rejected_or_incomplete_subscription_never_marks_stream_connected(monkeypatch):
    import asyncio
    import json
    from unittest.mock import patch
    from app.market_data import DeltaPublicStream

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
        async def send(self, payload):
            self.sent.append(json.loads(payload))
        async def recv(self):
            return json.dumps({"type": "subscriptions", "channels": [
                {"name": "ob_l1", "symbols": ["BTCUSD"], "error": "invalid symbol"}
            ]})

    class FakeConnection:
        def __init__(self):
            self.ws = FakeWebSocket()
        async def __aenter__(self):
            return self.ws
        async def __aexit__(self, *_args):
            return False

    async def run():
        connection = FakeConnection()
        disconnected = []
        stream = DeltaPublicStream(["BTCUSD"], lambda *_args: None, lambda: disconnected.append(True))

        with patch("app.market_data.websockets.connect", return_value=connection), patch(
            "app.market_data.asyncio.sleep", side_effect=asyncio.CancelledError
        ):
            try:
                await stream.run_forever()
            except asyncio.CancelledError:
                pass

        assert stream.connected is False
        assert stream.reconnects == 1
        assert disconnected == [True]

    asyncio.run(run())


def test_stream_tracks_disconnect_reason_and_connection_timestamps():
    from app.market_data import DeltaPublicStream

    stream = DeltaPublicStream(["BTCUSD"], lambda *_args: None)
    assert stream.last_connected_at is None
    assert stream.last_disconnect_at is None
    assert stream.last_disconnect_reason is None


def test_live_5m_candlestick_forwards_traded_ohlcv_with_bucket_timestamp():
    import asyncio
    from datetime import timedelta
    from app.market_data import DeltaPublicStream

    received = []

    async def on_quote(*_args):
        pass

    async def on_candle(symbol, candle, local_received):
        received.append((symbol, candle, local_received))

    stream = DeltaPublicStream(["BTCUSD"], on_quote, on_candle=on_candle)
    now_bucket = int(datetime.now(timezone.utc).timestamp()) // 300 * 300
    event_ts = (now_bucket + 42) * 1_000_000
    asyncio.run(stream._handle_message({
        "type": "candlestick_5m",
        "sy": "BTCUSD", "res": "5m", "ts": event_ts,
        "o": "100", "h": "103", "l": "99", "c": "102", "v": "456.5",
    }))
    assert len(received) == 1
    symbol, candle, local_received = received[0]
    assert symbol == "BTCUSD"
    assert candle.timestamp == datetime.fromtimestamp(now_bucket, timezone.utc)
    assert (candle.open, candle.high, candle.low, candle.close, candle.volume) == (100, 103, 99, 102, 456.5)
    assert local_received.tzinfo == timezone.utc


def test_live_candlestick_rejects_mark_price_without_traded_volume():
    import asyncio
    from app.market_data import DeltaPublicStream

    received = []

    async def on_candle(*args):
        received.append(args)

    stream = DeltaPublicStream(["BTCUSD"], lambda *_args: None, on_candle=on_candle)
    now_bucket = int(datetime.now(timezone.utc).timestamp()) // 300 * 300
    asyncio.run(stream._handle_message({
        "type": "candlestick_5m", "sy": "BTCUSD", "res": "5m",
        "ts": now_bucket * 1_000_000, "o": 100, "h": 101, "l": 99, "c": 100,
    }))
    assert received == []


def test_stream_subscribes_to_l1_and_live_5m_candles_when_callback_enabled():
    import asyncio
    import json
    from unittest.mock import patch
    from app.market_data import DeltaPublicStream

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.read = 0
        async def send(self, payload):
            self.sent.append(json.loads(payload))
        async def recv(self):
            self.read += 1
            if self.read == 1:
                return json.dumps({"type": "subscriptions", "channels": [
                    {"name": "ob_l1", "symbols": ["BTCUSD"]},
                    {"name": "candlestick_5m", "symbols": ["BTCUSD"]},
                ]})
            raise StopAsyncIteration

    class FakeConnection:
        def __init__(self):
            self.ws = FakeWebSocket()
        async def __aenter__(self):
            return self.ws
        async def __aexit__(self, *_args):
            return False

    async def run():
        connection = FakeConnection()
        stream = DeltaPublicStream(["BTCUSD"], lambda *_args: None, on_candle=lambda *_args: None)
        with patch("app.market_data.websockets.connect", return_value=connection), patch(
            "app.market_data.asyncio.sleep", side_effect=asyncio.CancelledError
        ):
            try:
                await stream.run_forever()
            except asyncio.CancelledError:
                pass
        subscribed = connection.ws.sent[0]["payload"]["channels"]
        assert {item["name"] for item in subscribed} == {"ob_l1", "candlestick_5m"}
        assert stream.connected is False

    asyncio.run(run())


def test_live_perpetual_product_discovery_paginates_and_filters():
    import asyncio
    from types import SimpleNamespace
    from app.market_data import DeltaRestClient

    async def run():
        client = DeltaRestClient("https://api.india.delta.exchange")
        calls = []
        pages = [
            {
                "success": True,
                "result": [
                    {"symbol": "BTCUSD", "contract_type": "perpetual_futures", "state": "live", "trading_status": "operational"},
                    {"symbol": "ETHUSD", "contract_type": "perpetual_futures", "state": "live", "trading_status": "disrupted_post_only"},
                    {"symbol": "BTC-OPT", "contract_type": "call_options", "state": "live", "trading_status": "operational"},
                ],
                "meta": {"after": "cursor-1"},
            },
            {
                "success": True,
                "result": [
                    {"symbol": "SOLUSD", "contract_type": "perpetual_futures", "state": "live"},
                    {"symbol": "XRPUSD", "contract_type": "perpetual_futures", "state": "upcoming"},
                ],
                "meta": {"after": None},
            },
        ]

        async def fake_get(path, params):
            calls.append((path, dict(params)))
            return SimpleNamespace(json=lambda: pages.pop(0))

        client._get_with_retry = fake_get
        symbols = await client.live_perpetual_symbols()
        assert symbols == {"BTCUSD", "SOLUSD"}
        assert len(calls) == 2
        assert calls[0][1]["contract_types"] == "perpetual_futures"
        assert calls[1][1]["after"] == "cursor-1"
        await client.close()

    asyncio.run(run())



def test_rest_rate_limit_uses_delta_reset_header_in_milliseconds(monkeypatch):
    import asyncio
    import httpx
    from app.market_data import DeltaRestClient

    async def run():
        client = DeltaRestClient("https://api.india.delta.exchange")
        request = httpx.Request("GET", "https://api.india.delta.exchange/test")
        responses = [
            httpx.Response(429, headers={"X-RATE-LIMIT-RESET": "2500"}, request=request),
            httpx.Response(200, json={"success": True, "result": []}, request=request),
        ]
        delays = []

        async def fake_get(_path, params=None):
            return responses.pop(0)

        async def capture_delay(delay):
            delays.append(delay)

        monkeypatch.setattr(client.client, "get", fake_get)
        monkeypatch.setattr("app.market_data.asyncio.sleep", capture_delay)
        response = await client._get_with_retry("/test", {})
        assert response.status_code == 200
        assert delays == [2.5]
        await client.close()

    asyncio.run(run())


def test_rest_transport_warning_identifies_error_type_without_leaking_exception(monkeypatch, caplog):
    import asyncio
    import httpx
    import logging
    from app.market_data import DeltaRestClient

    async def run():
        client = DeltaRestClient("https://api.india.delta.exchange")
        calls = 0

        async def flaky(_path, params=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectTimeout("temporary timeout")
            return httpx.Response(200, json={"ok": True},
                                  request=httpx.Request("GET", "https://api.india.delta.exchange/test"))

        async def no_wait(_delay):
            return None

        monkeypatch.setattr(client.client, "get", flaky)
        monkeypatch.setattr("app.market_data.asyncio.sleep", no_wait)
        with caplog.at_level(logging.WARNING, logger="app.market_data"):
            response = await client._get_with_retry("/test", {})
        assert response.status_code == 200
        assert "error_type=ConnectTimeout" in caplog.text
        await client.close()

    asyncio.run(run())



def test_rest_client_uses_connect_timeout_for_render_network_latency():
    from app.market_data import DeltaRestClient

    client = DeltaRestClient("https://api.india.delta.exchange")
    assert client.client.timeout.connect == 20.0
    assert client.client.timeout.read == 25.0
    import asyncio
    asyncio.run(client.close())


def test_live_candlestick_volume_feed_counts_zero_positive_and_missing_volume():
    import asyncio
    from app.market_data import DeltaPublicStream

    async def on_candle(*_args):
        pass

    stream = DeltaPublicStream(["BTCUSD"], lambda *_args: None, on_candle=on_candle)
    bucket = int(datetime.now(timezone.utc).timestamp()) // 300 * 300
    base = {
        "type": "candlestick_5m", "sy": "BTCUSD", "res": "5m",
        "ts": bucket * 1_000_000, "o": 100, "h": 101, "l": 99, "c": 100,
    }
    async def run():
        await stream._handle_message({**base, "v": 0})
        await stream._handle_message({**base, "v": 12.5})
        await stream._handle_message({key: value for key, value in base.items() if key != "ts"} )
    asyncio.run(run())
    diag = stream.candle_volume_diagnostics()
    assert diag["counts"] == {
        "messages_seen": 3, "missing_required_fields": 1,
        "invalid": 0, "stale": 0, "zero_volume": 1, "positive_volume": 1,
    }
    assert [sample["volume"] for sample in diag["recent_samples"]] == [0.0, 12.5]
