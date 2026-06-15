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
