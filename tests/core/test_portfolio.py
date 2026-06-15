from datetime import datetime, timezone

from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Fill, Side


def _fill(side, qty, price, commission=0.0):
    return Fill(order_id="o", symbol="EURUSD", side=side, quantity=qty,
                price=price, ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
                commission=commission)


def test_starting_cash_and_flat_equity():
    p = Portfolio(starting_cash=100_000)
    assert p.cash == 100_000
    assert p.equity({"EURUSD": 1.10}) == 100_000


def test_buy_reduces_cash_and_opens_long():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    assert p.cash == 100_000 - 11_000
    assert p.position("EURUSD").quantity == 10_000
    assert p.position("EURUSD").avg_price == 1.1000


def test_equity_marks_to_market():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    # price rises to 1.1010 -> +10 quote-ccy on 10k units
    assert round(p.equity({"EURUSD": 1.1010}), 2) == 100_010.00


def test_commission_reduces_cash():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000, commission=0.5))
    assert p.cash == 100_000 - 11_000 - 0.5


def test_close_realizes_pnl():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    p.apply_fill(_fill(Side.SELL, 10_000, 1.1010))
    assert p.position("EURUSD").quantity == 0
    assert round(p.realized_pnl, 2) == 10.00
    assert round(p.equity({"EURUSD": 1.1010}), 2) == 100_010.00


def test_adding_to_long_updates_avg_price():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1020))
    assert p.position("EURUSD").quantity == 20_000
    assert round(p.position("EURUSD").avg_price, 5) == 1.1010


def test_partial_close_keeps_avg_price_and_realizes_on_closed_portion():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    p.apply_fill(_fill(Side.SELL, 4_000, 1.1200))
    pos = p.position("EURUSD")
    assert pos.quantity == 6_000
    # avg_price of the remaining lot is unchanged by a partial close
    assert round(pos.avg_price, 5) == 1.1000
    # realize only on the 4_000 closed: (1.12 - 1.10) * 4_000 = 80
    assert round(p.realized_pnl, 2) == 80.00
    # cash: 100_000 - 11_000 (buy) + 4_480 (sell) = 93_480
    assert round(p.cash, 2) == 93_480.00
    assert round(p.equity({"EURUSD": 1.1200}), 2) == 100_200.00


def test_sell_flip_through_zero_rebases_avg_to_fill_price():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    p.apply_fill(_fill(Side.SELL, 15_000, 1.1200))
    pos = p.position("EURUSD")
    assert pos.quantity == -5_000
    # remaining short lot is re-based to the fill price
    assert round(pos.avg_price, 5) == 1.1200
    # realize only on the 10_000 that closed the long, not the full 15_000
    assert round(p.realized_pnl, 2) == 200.00
    # cash: 100_000 - 11_000 (buy) + 16_800 (sell) = 105_800
    assert round(p.cash, 2) == 105_800.00
    assert round(p.equity({"EURUSD": 1.1200}), 2) == 100_200.00


def test_buy_flip_through_zero_from_short_rebases_avg_to_fill_price():
    # symmetric short-side flip: SELL opens short, larger BUY flips to long
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.SELL, 10_000, 1.1200))
    p.apply_fill(_fill(Side.BUY, 15_000, 1.1000))
    pos = p.position("EURUSD")
    assert pos.quantity == 5_000
    assert round(pos.avg_price, 5) == 1.1000
    # short closed 10_000 lower than entry -> profit: (1.12 - 1.10) * 10_000 = 200
    assert round(p.realized_pnl, 2) == 200.00
    # cash: 100_000 + 11_200 (sell) - 16_500 (buy) = 94_700
    assert round(p.cash, 2) == 94_700.00


def test_partial_close_of_short_keeps_avg_price():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.SELL, 10_000, 1.1200))
    p.apply_fill(_fill(Side.BUY, 4_000, 1.1000))
    pos = p.position("EURUSD")
    assert pos.quantity == -6_000
    assert round(pos.avg_price, 5) == 1.1200
    # covered 4_000 of the short at a lower price -> profit (1.12 - 1.10)*4_000 = 80
    assert round(p.realized_pnl, 2) == 80.00


def test_adding_to_short_updates_avg_price():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.SELL, 10_000, 1.1000))
    p.apply_fill(_fill(Side.SELL, 10_000, 1.1020))
    pos = p.position("EURUSD")
    assert pos.quantity == -20_000
    assert round(pos.avg_price, 5) == 1.1010
    assert p.realized_pnl == 0.0
