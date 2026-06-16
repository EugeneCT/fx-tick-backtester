import math
from datetime import datetime, timedelta, timezone

from fxtrader.core.types import Fill, Side
from fxtrader.metrics.performance import compute_metrics, trade_pnls


def _curve(values, start=datetime(2024, 1, 2, tzinfo=timezone.utc), step_hours=1):
    return [(start + timedelta(hours=i * step_hours), v) for i, v in enumerate(values)]


def test_total_return_and_final_equity():
    m = compute_metrics(_curve([100_000, 101_000, 102_000]), [], periods_per_year=8760)
    assert round(m["total_return"], 5) == round(102_000 / 100_000 - 1, 5)
    assert m["final_equity"] == 102_000


def test_max_drawdown():
    # peak 110, trough 99 -> dd = (99-110)/110 = -0.1
    m = compute_metrics(_curve([100, 110, 99, 105]), [], periods_per_year=252)
    assert round(m["max_drawdown"], 5) == round(-11 / 110, 5)


def test_sharpe_positive_for_steady_gains():
    m = compute_metrics(_curve([100, 101, 102, 103, 104]), [], periods_per_year=252)
    assert m["sharpe"] > 0
    assert math.isfinite(m["sharpe"])


def test_flat_curve_has_zero_sharpe_and_drawdown():
    m = compute_metrics(_curve([100, 100, 100]), [], periods_per_year=252)
    assert m["sharpe"] == 0.0
    assert m["max_drawdown"] == 0.0


def test_trade_pnls_from_round_trip_fills():
    ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
    fills = [
        Fill("1", "EURUSD", Side.BUY, 10_000, 1.1000, ts),
        Fill("2", "EURUSD", Side.SELL, 10_000, 1.1010, ts),   # +10 win
        Fill("3", "EURUSD", Side.BUY, 10_000, 1.2000, ts),
        Fill("4", "EURUSD", Side.SELL, 10_000, 1.1990, ts),   # -10 loss
    ]
    pnls = trade_pnls(fills)
    assert [round(p, 4) for p in pnls] == [10.0, -10.0]


def test_trade_stats_win_rate_and_profit_factor():
    ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
    fills = [
        Fill("1", "EURUSD", Side.BUY, 10_000, 1.1000, ts),
        Fill("2", "EURUSD", Side.SELL, 10_000, 1.1010, ts),   # +10
        Fill("3", "EURUSD", Side.BUY, 10_000, 1.2000, ts),
        Fill("4", "EURUSD", Side.SELL, 10_000, 1.1990, ts),   # -10
        Fill("5", "EURUSD", Side.BUY, 10_000, 1.3000, ts),
        Fill("6", "EURUSD", Side.SELL, 10_000, 1.3030, ts),   # +30
    ]
    m = compute_metrics(_curve([100_000, 100_030]), fills, periods_per_year=8760)
    assert m["num_trades"] == 3
    assert round(m["win_rate"], 4) == round(2 / 3, 4)
    # profit factor = (10+30) / 10 = 4.0
    assert round(m["profit_factor"], 4) == 4.0


def test_empty_curve_is_safe():
    m = compute_metrics([], [], periods_per_year=252)
    assert m["num_trades"] == 0
    assert m["sharpe"] == 0.0
    assert m["total_return"] == 0.0
    assert m["ruined"] is False


def test_metrics_robust_to_account_ruin():
    # Equity blows through zero. Risk-adjusted ratios must NOT come out positive
    # (negative denominators flip the sign of step returns), drawdown floors at
    # -100%, and the run is flagged ruined.
    m = compute_metrics(_curve([100_000, 50_000, -10_000, 5_000]), [], periods_per_year=252)
    assert m["ruined"] is True
    assert m["sharpe"] == 0.0
    assert m["sortino"] == 0.0
    assert m["max_drawdown"] == -1.0
    assert round(m["total_return"], 4) == round(5_000 / 100_000 - 1, 4)  # -0.95


def test_healthy_curve_not_flagged_ruined():
    m = compute_metrics(_curve([100_000, 101_000, 102_000]), [], periods_per_year=8760)
    assert m["ruined"] is False
    assert m["sharpe"] > 0
