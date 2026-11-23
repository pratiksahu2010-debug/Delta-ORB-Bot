"""Delta India public market-data adapters.

REST candle history is the source of completed OHLC bars. WebSocket L1 is used
only for live trigger quotes; it is never treated as a candle close.
"""
import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import websockets

from .models import Candle, Quote

log = logging.getLogger(__name__)
WS_URL = "wss://public-socket.india.delta.exchange"
RESOLUTION_SECONDS = {"5m": 300, "15m": 900}


def _utc_from_epoch(value: Any) -> datetime:
    stamp = float(value)
    # Delta WS timestamps are microseconds; REST candle times are seconds.
    if stamp > 10**14:
        stamp /= 1_000_000
    elif stamp > 10**11:
        stamp /= 1_000
    return datetime.fromtimestamp(stamp, tz=timezone.utc)


def parse_rest_candles(payload: Any, resolution: str, now: datetime | None = None) -> list[Candle]:
    """Normalize Delta REST candles and discard any candle not fully closed."""
    seconds = RESOLUTION_SECONDS[resolution]
    rows = payload.get("result", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Unexpected Delta candle response: result is not a list")
    now = now or datetime.now(timezone.utc)
    candles: dict[datetime, Candle] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_time = row.get("time", row.get("timestamp", row.get("t")))
        if raw_time is None:
            continue
        try:
            ts = _utc_from_epoch(raw_time)
            candle = Candle(
                timestamp=ts,
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row.get("volume") or 0),
            )
        except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
            log.warning("Skipping malformed Delta candle row: %s", exc)
            continue
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if (not all(math.isfinite(value) for value in values)
                or min(candle.open, candle.high, candle.low, candle.close) <= 0
                or candle.volume < 0):
            log.warning(
                "Skipping non-finite, non-positive-price, or negative-volume Delta candle timestamp=%s", ts
            )
            continue
        # Reject off-grid timestamps instead of creating shifted/overlapping bars.
        if int(ts.timestamp()) % seconds != 0:
            continue
        # REST 'time' is the candle start. Do not expose the forming candle.
        if (now - ts).total_seconds() < seconds:
            continue
        if candle.high < max(candle.open, candle.close, candle.low):
            continue
        if candle.low > min(candle.open, candle.close, candle.high):
            continue
        if ts in candles:
            log.warning("Duplicate Delta candle timestamp resolution=%s timestamp=%s", resolution, ts.isoformat())
        candles[ts] = candle
    ordered = [candles[k] for k in sorted(candles)]
    for previous, current in zip(ordered, ordered[1:]):
        gap = (current.timestamp - previous.timestamp).total_seconds()
        if gap != seconds:
            log.warning("Delta candle gap resolution=%s previous=%s current=%s gap_seconds=%s",
                        resolution, previous.timestamp.isoformat(), current.timestamp.isoformat(), gap)
    return ordered


class DeltaRestClient:
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(connect=20.0, read=25.0, write=10.0, pool=10.0),
            headers={"Accept": "application/json", "User-Agent": "delta-signal-bot/1.0"},
        )

    async def _get_with_retry(self, path: str, params: dict[str, Any]) -> httpx.Response:
        """Retry transient REST failures with bounded backoff; never retry permanent 4xx."""
        attempts = 4
        for attempt in range(attempts):
            try:
                response = await self.client.get(path, params=params)
            except httpx.RequestError as exc:
                if attempt == attempts - 1:
                    log.error(
                        "Delta REST request exhausted retries endpoint=%s attempts=%s error_type=%s",
                        path, attempts, type(exc).__name__,
                    )
                    raise
                delay = min(8.0, 0.5 * (2 ** attempt))
                log.warning(
                    "Delta REST transport failure endpoint=%s error_type=%s; "
                    "retrying attempt=%s/%s in %.2fs",
                    path, type(exc).__name__, attempt + 2, attempts, delay,
                )
                await asyncio.sleep(delay)
                continue

            status = response.status_code
            if status in (408, 425, 429) or status >= 500:
                if attempt == attempts - 1:
                    log.error(
                        "Delta REST HTTP retries exhausted endpoint=%s status=%s attempts=%s",
                        path, status, attempts,
                    )
                    response.raise_for_status()

                delay = min(8.0, 0.5 * (2 ** attempt))
                if status == 429:
                    # Delta returns X-RATE-LIMIT-RESET in milliseconds, not seconds.
                    reset_ms = response.headers.get("X-RATE-LIMIT-RESET")
                    retry_after = response.headers.get("Retry-After")
                    try:
                        if reset_ms is not None:
                            delay = min(300.0, max(0.0, float(reset_ms) / 1000.0))
                        elif retry_after is not None:
                            delay = min(300.0, max(0.0, float(retry_after)))
                    except (TypeError, ValueError):
                        pass
                elif response.headers.get("Retry-After"):
                    try:
                        delay = min(30.0, max(0.0, float(response.headers["Retry-After"])))
                    except (TypeError, ValueError):
                        pass

                log.warning(
                    "Delta REST transient HTTP failure endpoint=%s status=%s "
                    "retry_reset_ms=%s; retrying attempt=%s/%s in %.2fs",
                    path, status, response.headers.get("X-RATE-LIMIT-RESET"),
                    attempt + 2, attempts, delay,
                )
                await asyncio.sleep(delay)
                continue

            response.raise_for_status()
            return response
        raise RuntimeError("Delta REST request exhausted retries")

    async def candles(self, symbol: str, resolution: str, limit: int = 120) -> list[Candle]:
        seconds = RESOLUTION_SECONDS[resolution]
        end = int(time.time())
        start = end - seconds * (limit + 3)
        response = await self._get_with_retry(
            "/v2/history/candles",
            params={"resolution": resolution, "symbol": symbol, "start": start, "end": end},
        )
        payload = response.json()
        if isinstance(payload, dict) and payload.get("success") is False:
            raise RuntimeError(f"Delta candle API error: {payload.get('error')}")
        return parse_rest_candles(payload, resolution) [-limit:]

    async def tickers(self) -> list[dict[str, Any]]:
        """Fetch the current public ticker snapshot for all live perpetuals."""
        response = await self._get_with_retry(
            "/v2/tickers", {"contract_types": "perpetual_futures"}
        )
        payload = response.json()
        if isinstance(payload, dict) and payload.get("success") is False:
            raise RuntimeError(f"Delta ticker API error: {payload.get('error')}")
        rows = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("Unexpected Delta ticker response: result is not a list")
        return rows

    async def live_perpetual_symbols(self) -> set[str]:
        """Return all currently live, operational perpetual-futures symbols."""
        params: dict[str, Any] = {
            "contract_types": "perpetual_futures",
            "states": "live",
            "page_size": 100,
        }
        symbols: set[str] = set()
        seen_cursors: set[str] = set()
        for _page in range(20):
            response = await self._get_with_retry("/v2/products", params)
            payload = response.json()
            if isinstance(payload, dict) and payload.get("success") is False:
                raise RuntimeError(f"Delta products API error: {payload.get('error')}")
            rows = payload.get("result") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError("Unexpected Delta products response: result is not a list")
            for product in rows:
                if not isinstance(product, dict):
                    continue
                symbol = str(product.get("symbol") or "").strip().upper()
                if (symbol and product.get("contract_type") == "perpetual_futures"
                        and product.get("state") == "live"
                        and product.get("trading_status") in (None, "operational")):
                    symbols.add(symbol)
            meta = payload.get("meta") or {}
            cursor = meta.get("after") if isinstance(meta, dict) else None
            if not cursor:
                return symbols
            cursor = str(cursor)
            if cursor in seen_cursors:
                raise RuntimeError("Delta products pagination returned a repeated cursor")
            seen_cursors.add(cursor)
            params["after"] = cursor
        raise RuntimeError("Delta products pagination exceeded 20 pages")

    async def close(self) -> None:
        await self.client.aclose()


class DeltaPublicStream:
    """Reconnectable L1 + optional live 5m OHLCV stream."""
    def __init__(self, symbols: list[str], on_quote, on_disconnect=None, on_candle=None):
        self.set_symbols(symbols)
        self.on_quote = on_quote
        self.on_disconnect = on_disconnect
        self.on_candle = on_candle
        self.required_channels = ["ob_l1"] + (["candlestick_5m"] if on_candle else [])
        self.connected = False
        self.last_message_at: datetime | None = None
        self.last_heartbeat_at: datetime | None = None
        self.reconnects = 0
        self.last_connected_at: datetime | None = None
        self.last_disconnect_at: datetime | None = None
        self.last_disconnect_reason: str | None = None
        self.candlestick_updates = 0
        self.heartbeat_messages_received = 0
        self.last_candlestick_at: datetime | None = None
        self.last_candlestick_by_symbol: dict[str, datetime] = {}
        self.candle_volume_counts = {
            "messages_seen": 0, "missing_required_fields": 0,
            "invalid": 0, "stale": 0, "zero_volume": 0, "positive_volume": 0,
        }
        self.candle_volume_samples: list[dict] = []

    def set_symbols(self, symbols: list[str]) -> None:
        """Update the subscription universe and its hot-path membership index."""
        self.symbols = list(symbols)
        self.symbol_set = frozenset(self.symbols)

    def candle_volume_diagnostics(self) -> dict:
        return {
            "counts": dict(self.candle_volume_counts),
            "recent_samples": list(self.candle_volume_samples[-20:]),
        }

    async def run_forever(self) -> None:
        delay = 1.0
        while True:
            connection_open = False
            try:
                async with websockets.connect(
                    WS_URL, ping_interval=None, ping_timeout=None, close_timeout=10,
                    open_timeout=45, max_queue=2048,
                ) as ws:
                    connection_open = True
                    self.last_heartbeat_at = None
                    channels = [{"name": "ob_l1", "symbols": self.symbols}]
                    if self.on_candle is not None:
                        channels.append({"name": "candlestick_5m", "symbols": self.symbols})
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "payload": {"channels": channels},
                    }))
                    # Delta's documented application heartbeat is independent
                    # of WebSocket protocol ping/pong. Enable it only after the
                    # server confirms every required subscription; sending it
                    # before the subscription acknowledgement can be ignored.
                    last_activity_monotonic = time.monotonic()
                    subscription_deadline = last_activity_monotonic + 10.0
                    subscription_confirmed = False
                    while True:
                        now_monotonic = time.monotonic()
                        activity_remaining = 60.0 - (now_monotonic - last_activity_monotonic)
                        if activity_remaining <= 0:
                            raise TimeoutError("Delta WebSocket inactive for 60 seconds")
                        timeout = activity_remaining
                        if not subscription_confirmed:
                            subscription_remaining = subscription_deadline - now_monotonic
                            if subscription_remaining <= 0:
                                raise TimeoutError("Delta WebSocket ob_l1 subscription acknowledgement missing for 10 seconds")
                            timeout = min(timeout, subscription_remaining)
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        except asyncio.TimeoutError as exc:
                            if not subscription_confirmed:
                                raise TimeoutError("Delta WebSocket ob_l1 subscription acknowledgement missing for 10 seconds") from exc
                            raise TimeoutError("Delta WebSocket inactive for 60 seconds") from exc
                        try:
                            msg = json.loads(raw)
                            if isinstance(msg, dict) and msg.get("type") == "heartbeat":
                                last_activity_monotonic = time.monotonic()
                                self.last_heartbeat_at = datetime.now(timezone.utc)
                                self.last_message_at = self.last_heartbeat_at
                                self.heartbeat_messages_received += 1
                                if self.heartbeat_messages_received == 1 or self.heartbeat_messages_received % 10 == 0:
                                    log.info(
                                        "Delta application heartbeat received count=%s",
                                        self.heartbeat_messages_received,
                                    )
                                continue
                            if isinstance(msg, dict) and msg.get("type") == "subscriptions":
                                channels = msg.get("channels")
                                for required_name in self.required_channels:
                                    confirmed = next(
                                        (channel for channel in channels or []
                                         if isinstance(channel, dict) and channel.get("name") == required_name),
                                        None,
                                    )
                                    if confirmed is None:
                                        raise RuntimeError(
                                            f"Delta WebSocket did not confirm {required_name} subscription"
                                        )
                                    if confirmed.get("error"):
                                        raise RuntimeError(
                                            f"Delta {required_name} subscription rejected: {confirmed['error']}"
                                        )
                                    confirmed_symbols = set(confirmed.get("symbols") or [])
                                    missing_symbols = set(self.symbols) - confirmed_symbols
                                    if missing_symbols:
                                        raise RuntimeError(
                                            f"Delta {required_name} subscription missing symbols: "
                                            + ",".join(sorted(missing_symbols))
                                        )
                                await ws.send(json.dumps({"type": "enable_heartbeat"}))
                                last_activity_monotonic = time.monotonic()
                                subscription_confirmed = True
                                self.connected = True
                                self.last_connected_at = datetime.now(timezone.utc)
                                delay = 1.0
                                self.last_message_at = datetime.now(timezone.utc)
                                log.info("Delta public WebSocket subscription confirmed channel=ob_l1 symbol_count=%s", len(self.symbols))
                                continue
                            if not subscription_confirmed:
                                # Do not use pre-ack market data for trigger checks.
                                continue
                            last_activity_monotonic = time.monotonic()
                            self.last_message_at = datetime.now(timezone.utc)
                            await self._handle_message(msg)
                        except (ValueError, TypeError, KeyError) as exc:
                            log.warning("Ignoring malformed Delta WS message: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.reconnects += 1
                self.last_disconnect_at = datetime.now(timezone.utc)
                self.last_disconnect_reason = f"{type(exc).__name__}: {str(exc)[:180]}"
                log.exception("Delta WebSocket disconnected; retry in %.1fs", delay)
            finally:
                was_connected = self.connected or connection_open
                if was_connected:
                    previous_disconnect_at = self.last_disconnect_at
                    self.last_disconnect_at = datetime.now(timezone.utc)
                    if (
                        self.last_disconnect_reason is None
                        or (
                            self.last_connected_at is not None
                            and (previous_disconnect_at is None or previous_disconnect_at <= self.last_connected_at)
                        )
                    ):
                        self.last_disconnect_reason = "connection_closed_or_cancelled"
                self.connected = False
                if was_connected and self.on_disconnect is not None:
                    try:
                        result = self.on_disconnect()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        log.exception("WebSocket disconnect cleanup callback failed")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)

    async def _handle_candle_item(self, item: dict) -> None:
        """Validate and forward one live traded-price 5m candle update."""
        if self.on_candle is None:
            return
        self.candle_volume_counts["messages_seen"] += 1
        symbol = item.get("sy") or item.get("symbol")
        if symbol not in self.symbol_set or str(symbol).startswith("MARK:"):
            return
        if item.get("res") not in (None, "5m"):
            return
        raw_ts = item.get("ts") or item.get("t") or item.get("time")
        volume_key = "v" if "v" in item else ("volume" if "volume" in item else None)
        raw_volume = item.get(volume_key) if volume_key else None
        # Mark-price candles omit traded volume and are not valid MEX input.
        if raw_ts is None or raw_volume is None:
            self.candle_volume_counts["missing_required_fields"] += 1
            return
        try:
            exchange_time = _utc_from_epoch(raw_ts)
            candle_epoch = int(exchange_time.timestamp()) // 300 * 300
            candle_time = datetime.fromtimestamp(candle_epoch, tz=timezone.utc)
            candle = Candle(
                timestamp=candle_time,
                open=float(item.get("o", item.get("open"))),
                high=float(item.get("h", item.get("high"))),
                low=float(item.get("l", item.get("low"))),
                close=float(item.get("c", item.get("close"))),
                volume=float(raw_volume),
            )
        except (ValueError, TypeError, OverflowError, OSError):
            self.candle_volume_counts["invalid"] += 1
            return
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if (not all(math.isfinite(value) for value in values) or candle.volume < 0
                or min(candle.open, candle.high, candle.low, candle.close) <= 0
                or candle.high < max(candle.open, candle.close, candle.low)
                or candle.low > min(candle.open, candle.close, candle.high)):
            self.candle_volume_counts["invalid"] += 1
            return
        received = datetime.now(timezone.utc)
        age = (received - candle.timestamp).total_seconds()
        if age < -5.0 or age > 600.0:
            self.candle_volume_counts["stale"] += 1
            return
        self.candle_volume_counts["zero_volume" if candle.volume == 0 else "positive_volume"] += 1
        self.candle_volume_samples.append({
            "symbol": symbol, "candle_at": candle.timestamp.isoformat(),
            "volume": round(candle.volume, 8), "volume_field": volume_key,
            "age_seconds": round(age, 2),
        })
        del self.candle_volume_samples[:-20]
        self.candlestick_updates += 1
        self.last_candlestick_at = received
        self.last_candlestick_by_symbol[symbol] = candle.timestamp
        result = self.on_candle(symbol, candle, received)
        if asyncio.iscoroutine(result):
            await result

    async def _handle_message(self, msg: dict) -> None:
        # Delta public channel payloads use compact keys; tolerate wrapper arrays.
        items = msg.get("d") if isinstance(msg, dict) else None
        if not isinstance(items, list):
            items = [msg]
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                message_type = item.get("type") or msg.get("type")
                if message_type == "candlestick_5m":
                    await self._handle_candle_item(item)
                    continue
                symbol = item.get("sy") or item.get("symbol")
                if symbol not in self.symbol_set:
                    continue
                bid, ask = item.get("bp"), item.get("ap")
                bid_value = float(bid) if bid is not None else None
                ask_value = float(ask) if ask is not None else None
                if ((bid_value is not None and (not math.isfinite(bid_value) or bid_value <= 0))
                        or (ask_value is not None and (not math.isfinite(ask_value) or ask_value <= 0))):
                    continue
                if bid_value is not None and ask_value is not None:
                    if ask_value < bid_value:
                        continue
                    price = (bid_value + ask_value) / 2.0
                else:
                    raw_price = item.get("p") or item.get("price") or item.get("last_price")
                    if raw_price is None:
                        continue
                    price = float(raw_price)
                if not math.isfinite(price) or price <= 0:
                    continue
                raw_ts = item.get("ts") or item.get("lts") or item.get("t")
                received = datetime.now(timezone.utc)
                quote_time = _utc_from_epoch(raw_ts) if raw_ts is not None else received
                # L1 exchange timestamp can be older than the local receive time.
                await self.on_quote(
                    symbol, Quote(price=price, received_at=quote_time, bid=bid_value, ask=ask_value), received
                )
            except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
                # One malformed item in a batched frame must not discard other
                # symbols' valid quote updates from the same WebSocket message.
                log.warning("Ignoring malformed Delta WS quote item: %s", exc)
                continue
