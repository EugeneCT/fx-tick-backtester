"""Pure domain types shared by every engine. No I/O, no broker imports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    ts: datetime  # tz-aware UTC
    bid: float
    ask: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: datetime  # bar OPEN time, tz-aware UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Order:
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    order_id: str | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    ts: datetime  # tz-aware UTC
    commission: float = 0.0


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float = 0.0  # signed: + long, - short
    avg_price: float = 0.0
