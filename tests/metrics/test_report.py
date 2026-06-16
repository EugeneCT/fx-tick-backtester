import json
from datetime import datetime, timezone

from fxtrader.config import BacktestConfig
from fxtrader.core.types import Fill, Side
from fxtrader.engine.backtest import BacktestResult
from fxtrader.metrics.performance import compute_metrics
from fxtrader.metrics.report import build_results, write_results_json


def _result():
    start = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    curve = [(start, 100_000.0), (datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc), 100_050.0)]
    fills = [
        Fill("1", "EURUSD", Side.BUY, 10_000, 1.1000, start),
        Fill("2", "EURUSD", Side.SELL, 10_000, 1.1050, start),
    ]
    return BacktestResult(equity_curve=curve, fills=fills, final_equity=100_050.0)


def _config():
    return BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-01-03")


def test_build_results_structure():
    result = _result()
    metrics = compute_metrics(result.equity_curve, result.fills, periods_per_year=8760)
    out = build_results(_config(), result, metrics)

    assert out["config"]["symbol"] == "EURUSD"
    assert out["metrics"]["num_fills"] == 2
    assert out["num_fills"] == 2
    # equity curve serialized as [iso, value] pairs
    assert out["equity_curve"][0] == ["2024-01-02T09:00:00+00:00", 100_000.0]
    # one round-trip trade, +50
    assert [round(p, 2) for p in out["trades"]] == [50.0]


def test_build_results_is_json_serializable():
    result = _result()
    metrics = compute_metrics(result.equity_curve, result.fills, periods_per_year=8760)
    out = build_results(_config(), result, metrics)
    # must round-trip through json with no custom encoder
    assert json.loads(json.dumps(out))["config"]["symbol"] == "EURUSD"


def test_write_results_json(tmp_path):
    result = _result()
    metrics = compute_metrics(result.equity_curve, result.fills, periods_per_year=8760)
    path = tmp_path / "results.json"
    write_results_json(path, _config(), result, metrics)
    loaded = json.loads(path.read_text())
    assert loaded["metrics"]["num_trades"] == 1
    assert loaded["config"]["bar_minutes"] == 1


def _all_wins_result():
    ts = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    # two winning round trips, zero losses -> profit_factor = inf
    fills = [
        Fill("1", "EURUSD", Side.BUY, 10_000, 1.1000, ts),
        Fill("2", "EURUSD", Side.SELL, 10_000, 1.1010, ts),
        Fill("3", "EURUSD", Side.BUY, 10_000, 1.1000, ts),
        Fill("4", "EURUSD", Side.SELL, 10_000, 1.1020, ts),
    ]
    curve = [(ts, 100_000.0), (datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc), 100_030.0)]
    return BacktestResult(equity_curve=curve, fills=fills, final_equity=100_030.0)


def test_infinite_profit_factor_becomes_null_and_stays_strict_json(tmp_path):
    result = _all_wins_result()
    metrics = compute_metrics(result.equity_curve, result.fills, periods_per_year=8760)
    assert metrics["profit_factor"] == float("inf")  # the in-memory metric is inf

    out = build_results(_config(), result, metrics)
    assert out["metrics"]["profit_factor"] is None  # sanitized for the JSON contract

    path = tmp_path / "results.json"
    write_results_json(path, _config(), result, metrics)
    # strict parse: no `Infinity`/`NaN` tokens (parse_constant raises if present)
    text = path.read_text()
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert "Infinity" not in text
