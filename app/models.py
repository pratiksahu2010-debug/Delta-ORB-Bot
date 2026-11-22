from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class Quote:
    price: float
    received_at: datetime
    bid: float | None = None
    ask: float | None = None
    source: str = "websocket"


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Literal["LONG", "SHORT"]
    candle_time: datetime
    entry_reference: float
    stop_loss: float
    take_profit_1: float
    atr: float
    reason: str
