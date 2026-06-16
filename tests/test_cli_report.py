import json
from datetime import datetime, timedelta, timezone

from fxtrader.config import BacktestConfig
from fxtrader.cli import run_backtest_config
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore
from fxtrader.metrics.report import write_html_report, write_results_json


def _price(i: int) -> float:
    # Minute closes (1 tick/sec, so 60 ticks per 1-min bar) form a small dip then
    # a clear uptrend, so the fast MA starts at/below the slow MA and crosses
    # *above* it on 1-min bars. The dip/turn must span multiple bars: a series
    # that is monotonic at bar granularity never produces an upward cross.
    if i < 60:
        return 1.100
    if i < 120:
        return 1.099
    if i < 180:
        return 1.100 + (i - 120) * 0.00001
    return 1.100 + (i - 180) * 0.001


def _seed(tmp_path):
    store = TickStore(tmp_path)
    start = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticks = []
    # dip then uptrend so an MA cross fires
    for i in range(240):
        p = _price(i)
        ticks.append(Tick("EURUSD", start + timedelta(seconds=i), p, p + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_run_backtest_config_and_write_reports(tmp_path):
    _seed(tmp_path)
    cfg = BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-01-03",
                         fast=2, slow=3, bar_minutes=1)
    result, metrics = run_backtest_config(cfg, data_dir=tmp_path)
    assert metrics["num_fills"] >= 1

    jpath = tmp_path / "results.json"
    hpath = tmp_path / "report.html"
    write_results_json(jpath, cfg, result, metrics)
    write_html_report(hpath, cfg, result, metrics)

    assert json.loads(jpath.read_text())["config"]["symbol"] == "EURUSD"
    assert "data:image/png;base64," in hpath.read_text()
