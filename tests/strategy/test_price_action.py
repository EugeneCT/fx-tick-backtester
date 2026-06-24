# tests/strategy/test_price_action.py
from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, Side
from fxtrader.strategy.examples.price_action import PriceActionTrend
from fxtrader.strategy.exits import FixedPips

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _drive(strategy, ohlc, *, symbol="EURUSD"):
    """Feed (open, high, low, close) bars; fills execute at bar.close into the portfolio."""
    pf = Portfolio(100_000.0)
    clock = SimClock()
    orders: list[Order] = []

    def sink(order: Order) -> None:
        orders.append(order)
        pf.apply_fill(Fill("x", order.symbol, order.side, order.quantity,
                           _drive.price, clock.now(), 0.0))

    strategy.bind(clock, pf, sink)
    strategy.on_start()
    for i, (o, h, l, c) in enumerate(ohlc):
        ts = T0 + timedelta(minutes=i)
        clock.set_time(ts)
        _drive.price = c
        strategy.on_bar(Bar(symbol, ts, o, h, l, c, 1.0))
    return pf, orders


# Uptrend warmup (4 bars, rising closes) then a bullish PIN on the 5th bar.
_UPTREND_PIN = [
    (1.0000, 1.0010, 0.9990, 1.0000),
    (1.0000, 1.0020, 1.0000, 1.0010),
    (1.0010, 1.0030, 1.0010, 1.0020),
    (1.0020, 1.0040, 1.0020, 1.0030),
    (1.0035, 1.0040, 0.9990, 1.0038),  # bullish pin; close 1.0038 > trend MA → long
]


def test_rejects_bad_pattern():
    with pytest.raises(ValueError):
        PriceActionTrend("EURUSD", pattern="nope", trend=4, size=10_000)


def test_no_trades_before_warmup():
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    _pf, orders = _drive(s, _UPTREND_PIN[:3])  # < trend bars
    assert orders == []


def test_pin_in_uptrend_goes_long():
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    pf, orders = _drive(s, _UPTREND_PIN)
    assert pf.position("EURUSD").quantity > 0
    assert len(orders) == 1 and orders[0].side == Side.BUY


def test_bullish_pin_against_downtrend_blocked():
    # Falling closes → trend down; a bullish pin must be suppressed by the gate.
    series = [
        (1.0030, 1.0040, 1.0020, 1.0030),
        (1.0020, 1.0030, 1.0010, 1.0020),
        (1.0010, 1.0020, 1.0000, 1.0010),
        (1.0000, 1.0010, 0.9990, 1.0000),
        (0.9995, 1.0000, 0.9950, 0.9998),  # bullish pin, but close < trend MA → blocked
    ]
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    pf, orders = _drive(s, series)
    assert pf.position("EURUSD").quantity == 0
    assert orders == []


def test_exit_first_stop_flattens():
    # Enter long on bar 5 at close 1.0038; FixedPips stop_dist=0.0020 → stop level 1.0018.
    s = PriceActionTrend(
        "EURUSD", pattern="pin", trend=4, size=10_000,
        exit_policy=FixedPips(stop_pips=20, tp_pips=200, pip_size=0.0001),
    )
    stop_bar = (1.0030, 1.0035, 1.0010, 1.0020)  # low 1.0010 <= 1.0018 → stop
    pf, orders = _drive(s, _UPTREND_PIN + [stop_bar])
    assert pf.position("EURUSD").quantity == 0
    assert orders[-1].side == Side.SELL
    assert s._last_exit_reason == "stop"


def test_max_hold_bars_flattens():
    # No exit policy; max_hold_bars=2 → time exit two bars after entry.
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000, max_hold_bars=2)
    hold1 = (1.0038, 1.0045, 1.0030, 1.0040)
    hold2 = (1.0040, 1.0048, 1.0035, 1.0044)  # 2nd bar held → time exit
    pf, orders = _drive(s, _UPTREND_PIN + [hold1, hold2])
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "time"


def test_ignores_other_symbols():
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    _pf, orders = _drive(s, _UPTREND_PIN, symbol="GBPUSD")
    assert orders == []


# ---------------------------------------------------------------------------
# Pattern-coverage tests: engulf / inside-break / swingbreak
# ---------------------------------------------------------------------------

def test_engulf_in_uptrend_goes_long():
    # trend=4; bars 1-4 are rising warmup.
    # Closes 1-4: 1.0000, 1.0010, 1.0020, 1.0030
    #
    # Bar 5 (bearish prev for engulf):
    #   open=1.0035, close=1.0025  → bearish candle
    #   deque=[1.0010,1.0020,1.0030,1.0025], MA=1.0021, close=1.0025>MA → trend_dir=+1
    #   engulfing(bar4,bar5): bar4 is bull, bar5 is bear; need bar5.close<=bar4.open(1.0020)
    #     → 1.0025>1.0020 → NOT engulfing. No entry.
    #
    # Bar 6 (bullish engulf):
    #   open=1.0020, close=1.0040
    #   engulfing(bar5,bar6): bar5 bear, bar6 bull;
    #     bar6.close(1.0040) >= bar5.open(1.0035) ✓
    #     bar6.open(1.0020) <= bar5.close(1.0025) ✓  → returns +1
    #   deque=[1.0020,1.0030,1.0025,1.0040], MA=1.0029, close=1.0040>MA → trend_dir=+1
    #   signal=+1 == trend_dir → LONG
    series = [
        (1.0000, 1.0010, 0.9990, 1.0000),  # bar1 warmup
        (1.0000, 1.0020, 1.0000, 1.0010),  # bar2 warmup
        (1.0010, 1.0030, 1.0010, 1.0020),  # bar3 warmup
        (1.0020, 1.0040, 1.0020, 1.0030),  # bar4 warmup
        (1.0035, 1.0040, 1.0015, 1.0025),  # bar5 bearish prev
        (1.0020, 1.0045, 1.0018, 1.0040),  # bar6 bullish engulf → long
    ]
    s = PriceActionTrend("EURUSD", pattern="engulf", trend=4, size=10_000)
    pf, orders = _drive(s, series)
    assert pf.position("EURUSD").quantity > 0
    assert orders[-1].side == Side.BUY


def test_inside_break_in_uptrend_goes_long():
    # trend=4, inside_break_bars=3
    # bars 1-3 warmup; bar4 is the mother bar (high=1.0040, low=1.0020).
    #
    # At bar4: deque=[1.0000,1.0010,1.0020,1.0030], MA=1.0015, trend_dir=+1.
    #   _inside_break: _mother=None; is_inside_bar(bar3,bar4): bar4.high=1.0040>bar3.high=1.0030
    #   → not inside → _mother stays None. No entry.
    #
    # Bar 5 (inside bar): high=1.0035<=1.0040, low=1.0022>=1.0020.
    #   deque=[1.0010,1.0020,1.0030,1.0028], MA=1.0022, close=1.0028>MA → trend_dir=+1.
    #   _inside_break: _mother=None; is_inside_bar(bar4,bar5) ✓ → _mother=(1.0040,1.0020,3).
    #   sig=0. No entry.
    #
    # Bar 6 (break bar): close=1.0045 > mother.high=1.0040.
    #   deque=[1.0020,1.0030,1.0028,1.0045], MA=1.0031, close=1.0045>MA → trend_dir=+1.
    #   _inside_break: _mother set; bar6.close(1.0045)>1.0040 → sig=1, _mother=None. → LONG
    series = [
        (1.0000, 1.0010, 0.9990, 1.0000),  # bar1 warmup
        (1.0000, 1.0020, 1.0000, 1.0010),  # bar2 warmup
        (1.0010, 1.0030, 1.0010, 1.0020),  # bar3 warmup
        (1.0020, 1.0040, 1.0020, 1.0030),  # bar4 mother bar
        (1.0025, 1.0035, 1.0022, 1.0028),  # bar5 inside bar
        (1.0038, 1.0050, 1.0035, 1.0045),  # bar6 break above mother.high → long
    ]
    s = PriceActionTrend("EURUSD", pattern="inside", trend=4, size=10_000, inside_break_bars=3)
    pf, orders = _drive(s, series)
    assert pf.position("EURUSD").quantity > 0
    assert orders[-1].side == Side.BUY


def test_swingbreak_in_uptrend_goes_long():
    # trend=4, swing_lookback=3 → _bars maxlen=max(3+1,2)=4
    # swing_break fires when len(bars)>=lookback+1=4 AND close > swing_high of prior lookback bars.
    #
    # Bars 1-4 warmup (rising):
    #   At bar4: _bars=[bar1,bar2,bar3,bar4]; window=bars[-4:-1]=[bar1,bar2,bar3]
    #   swing_high=max(bar1.high=1.0010, bar2.high=1.0020, bar3.high=1.0030)=1.0030
    #   bar4.close=1.0030 > 1.0030? → False. No entry.
    #
    # Bar 5: close=1.0045; _bars=[bar2,bar3,bar4,bar5] (maxlen=4 drops bar1)
    #   window=bars[-4:-1]=[bar2,bar3,bar4]
    #   swing_high=max(bar2.high=1.0020, bar3.high=1.0030, bar4.high=1.0040)=1.0040
    #   bar5.close=1.0045 > 1.0040 ✓ → +1
    #   deque=[1.0010,1.0020,1.0030,1.0045], MA=1.0026, close=1.0045>MA → trend_dir=+1 → LONG
    series = [
        (1.0000, 1.0010, 0.9990, 1.0000),  # bar1 warmup
        (1.0000, 1.0020, 1.0000, 1.0010),  # bar2 warmup
        (1.0010, 1.0030, 1.0010, 1.0020),  # bar3 warmup
        (1.0020, 1.0040, 1.0020, 1.0030),  # bar4 warmup (swing_break checks but no break yet)
        (1.0030, 1.0050, 1.0030, 1.0045),  # bar5 close > prior swing high 1.0040 → long
    ]
    s = PriceActionTrend("EURUSD", pattern="swingbreak", trend=4, size=10_000, swing_lookback=3)
    pf, orders = _drive(s, series)
    assert pf.position("EURUSD").quantity > 0
    assert orders[-1].side == Side.BUY
