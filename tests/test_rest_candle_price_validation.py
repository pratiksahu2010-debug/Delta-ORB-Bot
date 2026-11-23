from datetime import datetime, timezone

from app.market_data import parse_rest_candles


def test_rest_parser_rejects_zero_and_negative_ohlc_prices():
    now = datetime.fromtimestamp(1800001200, timezone.utc)
    rows = [
        {"time": 1800000000, "open": 10, "high": 12, "low": 9, "close": 11, "volume": 5},
        {"time": 1800000300, "open": 0, "high": 12, "low": 0, "close": 11, "volume": 5},
        {"time": 1800000600, "open": -1, "high": 12, "low": -2, "close": 11, "volume": 5},
    ]

    result = parse_rest_candles({"success": True, "result": rows}, "5m", now)

    assert len(result) == 1
    assert result[0].open == 10
    assert result[0].close == 11
