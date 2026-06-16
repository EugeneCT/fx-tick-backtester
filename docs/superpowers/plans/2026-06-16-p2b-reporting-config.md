# P2b — Reporting + Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make backtests reproducible (YAML/pydantic config) and consumable — write a machine-readable `results.json` (for the P4 research agent) and a human HTML report with an equity-curve plot — wired into `fxtrader backtest`.

**Architecture:** A pydantic `BacktestConfig` (loadable from YAML) captures every run parameter and maps to `run_backtest`. A `report` module turns a `BacktestResult` + metrics into (a) a structured `results.json` and (b) a self-contained HTML report with a base64-embedded matplotlib equity-curve PNG. The `backtest` CLI gains `--config`, `--json`, and `--html`.

**Tech Stack:** Python 3.12 (`.venv`), pydantic v2 + PyYAML (config), matplotlib (Agg backend, headless plot), stdlib json. Builds on the P2a backtester (`run_backtest`, `BacktestResult`, `compute_metrics`, `trade_pnls`).

**Conventions:**
- All timestamps tz-aware UTC. Use `.venv/bin/pytest` / `.venv/bin/python` (shell `source` doesn't persist). pytest uses `--import-mode=importlib`.
- `results.json` is **deterministic** (no wall-clock timestamp inside) so tests can assert it and the agent gets stable output.
- Commit after each task. No placeholders.

---

## File structure

```
pyproject.toml                          # Tasks 1 & 3 — add deps (MODIFY)
src/fxtrader/config.py                  # Task 1 — BacktestConfig + load_config
src/fxtrader/metrics/report.py          # Task 2 — build_results / write_results_json
                                        # Task 3 — write_html_report (adds matplotlib)
src/fxtrader/cli.py                     # Task 4 — --config/--json/--html (MODIFY)
tests/test_config.py                    # Task 1
tests/metrics/test_report.py            # Tasks 2 & 3
tests/test_cli_report.py                # Task 4
config/ma_eurusd.yaml                   # Task 4 — example config
```

---

## Task 1: BacktestConfig (pydantic + YAML)

**Files:**
- Modify: `pyproject.toml` (add `pydantic>=2`, `pyyaml>=6`)
- Create: `src/fxtrader/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Add deps and install**

Edit `pyproject.toml` `[project].dependencies` to add `"pydantic>=2"` and `"pyyaml>=6"` (keep existing `polars`, `typer`):
```toml
dependencies = [
    "polars>=1.0",
    "typer>=0.12",
    "pydantic>=2",
    "pyyaml>=6",
]
```
Run: `.venv/bin/pip install -e ".[dev]" -q`
Expected: installs pydantic + pyyaml.

- [ ] **Step 2: Write the failing test `tests/test_config.py`**

```python
from datetime import date

import pytest

from fxtrader.config import BacktestConfig, load_config


def test_minimal_config_has_defaults():
    cfg = BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-02-01")
    assert cfg.symbol == "EURUSD"
    assert cfg.start == date(2024, 1, 2)
    assert cfg.fast == 20 and cfg.slow == 50
    assert cfg.bar_minutes == 1
    assert cfg.daily_loss_limit is None  # disabled by default


def test_to_run_kwargs_maps_dates_and_unbounded_limits():
    cfg = BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-02-01",
                         fast=10, slow=30, daily_loss_limit=None, max_position=None)
    kw = cfg.to_run_kwargs()
    from datetime import datetime, timezone
    assert kw["symbol"] == "EURUSD"
    assert kw["start"] == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert kw["end"] == datetime(2024, 2, 1, tzinfo=timezone.utc)
    assert kw["fast"] == 10 and kw["slow"] == 30
    assert kw["daily_loss_limit"] == float("inf")  # None -> inf
    assert kw["max_position"] == float("inf")


def test_explicit_limits_pass_through():
    cfg = BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-02-01",
                         daily_loss_limit=5_000, max_position=50_000)
    kw = cfg.to_run_kwargs()
    assert kw["daily_loss_limit"] == 5_000
    assert kw["max_position"] == 50_000


def test_rejects_unknown_field():
    with pytest.raises(Exception):
        BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-02-01", bogus=1)


def test_load_config_from_yaml(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(
        "symbol: GBPUSD\nstart: 2023-06-01\nend: 2024-06-01\n"
        "fast: 15\nslow: 40\nbar_minutes: 5\nsize: 25000\n"
    )
    cfg = load_config(p)
    assert cfg.symbol == "GBPUSD"
    assert cfg.fast == 15 and cfg.slow == 40
    assert cfg.bar_minutes == 5
    assert cfg.size == 25_000
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.config'`

- [ ] **Step 4: Write `src/fxtrader/config.py`**

```python
"""Typed, YAML-loadable backtest run configuration."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class BacktestConfig(BaseModel):
    """A reproducible backtest specification. Maps to cli.run_backtest via to_run_kwargs()."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    start: date
    end: date
    fast: int = 20
    slow: int = 50
    size: float = 10_000
    bar_minutes: int = 1
    starting_cash: float = 100_000.0
    commission_rate: float = 0.00002
    min_commission: float = 2.0
    slippage: float = 0.0
    daily_loss_limit: float | None = None  # None -> unbounded
    max_position: float | None = None      # None -> unbounded

    def to_run_kwargs(self) -> dict:
        """Keyword args for cli.run_backtest (dates -> UTC datetimes; None -> inf)."""
        return {
            "symbol": self.symbol,
            "start": datetime(self.start.year, self.start.month, self.start.day, tzinfo=timezone.utc),
            "end": datetime(self.end.year, self.end.month, self.end.day, tzinfo=timezone.utc),
            "fast": self.fast,
            "slow": self.slow,
            "size": self.size,
            "bar_minutes": self.bar_minutes,
            "starting_cash": self.starting_cash,
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
            "slippage": self.slippage,
            "daily_loss_limit": self.daily_loss_limit if self.daily_loss_limit is not None else float("inf"),
            "max_position": self.max_position if self.max_position is not None else float("inf"),
        }


def load_config(path: str | Path) -> BacktestConfig:
    data = yaml.safe_load(Path(path).read_text())
    return BacktestConfig(**data)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Full suite + lint + Commit**

Run: `.venv/bin/pytest -q` → all pass.
Run: `.venv/bin/ruff check src/fxtrader/config.py tests/test_config.py` → clean.
```bash
git add pyproject.toml src/fxtrader/config.py tests/test_config.py
git commit -m "feat: BacktestConfig (pydantic + YAML run config)"
```

---

## Task 2: results.json writer

**Files:**
- Create: `src/fxtrader/metrics/report.py`
- Create: `tests/metrics/test_report.py`

`build_results` produces a deterministic, JSON-serializable dict; `write_results_json` writes it. It captures the config, metrics, the equity curve (ISO timestamps), and per-trade PnLs — the surface the P4 agent reads.

- [ ] **Step 1: Write the failing test `tests/metrics/test_report.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/metrics/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.metrics.report'`

- [ ] **Step 3: Write `src/fxtrader/metrics/report.py`**

```python
"""Backtest reporting: machine-readable results.json (and HTML, see write_html_report)."""

from __future__ import annotations

import json
from pathlib import Path

from fxtrader.config import BacktestConfig
from fxtrader.engine.backtest import BacktestResult
from fxtrader.metrics.performance import trade_pnls


def build_results(config: BacktestConfig, result: BacktestResult, metrics: dict) -> dict:
    """A deterministic, JSON-serializable summary of a backtest run."""
    return {
        "config": config.model_dump(mode="json"),
        "metrics": metrics,
        "equity_curve": [[ts.isoformat(), equity] for ts, equity in result.equity_curve],
        "trades": trade_pnls(result.fills),
        "num_fills": len(result.fills),
    }


def write_results_json(
    path: str | Path,
    config: BacktestConfig,
    result: BacktestResult,
    metrics: dict,
) -> None:
    Path(path).write_text(json.dumps(build_results(config, result, metrics), indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/metrics/test_report.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Full suite + lint + Commit**

Run: `.venv/bin/pytest -q` → all pass.
Run: `.venv/bin/ruff check src/fxtrader/metrics/report.py tests/metrics/test_report.py` → clean.
```bash
git add src/fxtrader/metrics/report.py tests/metrics/test_report.py
git commit -m "feat: results.json writer (deterministic machine-readable backtest output)"
```

---

## Task 3: HTML report with equity-curve plot

**Files:**
- Modify: `pyproject.toml` (add `matplotlib>=3.8`)
- Modify: `src/fxtrader/metrics/report.py` (add `write_html_report`)
- Modify: `tests/metrics/test_report.py` (add HTML tests)

`write_html_report` renders the equity curve to a PNG via matplotlib's Agg backend, base64-embeds it in a self-contained HTML page with a metrics table. No external files, no display.

- [ ] **Step 1: Add the dep and install**

Edit `pyproject.toml` `[project].dependencies` to add `"matplotlib>=3.8"`.
Run: `.venv/bin/pip install -e ".[dev]" -q`
Expected: installs matplotlib.

- [ ] **Step 2: Add the failing tests (append to `tests/metrics/test_report.py`)**

```python
from fxtrader.metrics.report import write_html_report


def test_write_html_report_is_self_contained(tmp_path):
    result = _result()
    metrics = compute_metrics(result.equity_curve, result.fills, periods_per_year=8760)
    path = tmp_path / "report.html"
    write_html_report(path, _config(), result, metrics)

    html = path.read_text()
    assert "<html" in html.lower()
    assert "EURUSD" in html
    assert "Sharpe" in html
    # equity plot embedded as a base64 PNG (self-contained, no external file)
    assert "data:image/png;base64," in html


def test_write_html_report_handles_empty_curve(tmp_path):
    result = BacktestResult(equity_curve=[], fills=[], final_equity=0.0)
    metrics = compute_metrics([], [], periods_per_year=8760)
    path = tmp_path / "empty.html"
    write_html_report(path, _config(), result, metrics)  # must not raise
    assert path.exists()
```

- [ ] **Step 3: Append `write_html_report` to `src/fxtrader/metrics/report.py`**

Add these imports at the top (merge with existing). Set the Agg backend BEFORE importing pyplot:
```python
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
```

Append at the end of the file:
```python
def _equity_png_base64(result: BacktestResult) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    if result.equity_curve:
        xs = [ts for ts, _ in result.equity_curve]
        ys = [equity for _, equity in result.equity_curve]
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.0)
    ax.set_title("Equity curve")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_html_report(
    path: str | Path,
    config: BacktestConfig,
    result: BacktestResult,
    metrics: dict,
) -> None:
    png = _equity_png_base64(result)
    rows = "".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in metrics.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Backtest — {config.symbol}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; }} td {{ border: 1px solid #ddd; padding: 4px 10px; }}
td:first-child {{ color: #555; }}
</style></head>
<body>
<h1>Backtest — {config.symbol}</h1>
<p>MA({config.fast}/{config.slow}) · {config.bar_minutes}m bars · {config.start} → {config.end}</p>
<img src="data:image/png;base64,{png}" alt="Equity curve">
<h2>Metrics</h2>
<table>{rows}</table>
</body></html>
"""
    Path(path).write_text(html)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/metrics/test_report.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Full suite + lint + Commit**

Run: `.venv/bin/pytest -q` → all pass.
Run: `.venv/bin/ruff check src/fxtrader/metrics/report.py` → clean (the `# noqa: E402` keeps the post-`matplotlib.use` import lint-clean).
```bash
git add pyproject.toml src/fxtrader/metrics/report.py tests/metrics/test_report.py
git commit -m "feat: self-contained HTML backtest report with equity-curve plot"
```

---

## Task 4: CLI `--config` / `--json` / `--html` + example config

**Files:**
- Modify: `src/fxtrader/cli.py`
- Create: `tests/test_cli_report.py`
- Create: `config/ma_eurusd.yaml`

`run_backtest_config(config, data_dir)` runs a config; the `backtest` command gains `--config` (supersedes flags), `--json`, and `--html` outputs. `symbol/start/end` become optional (required only when no `--config`).

- [ ] **Step 1: Write the failing test `tests/test_cli_report.py`**

```python
import json
from datetime import datetime, timedelta, timezone

from fxtrader.config import BacktestConfig
from fxtrader.cli import run_backtest_config
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore
from fxtrader.metrics.report import write_html_report, write_results_json


def _seed(tmp_path):
    store = TickStore(tmp_path)
    start = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticks = []
    # dip then uptrend so an MA cross fires
    for i in range(240):
        p = 1.10 - min(i, 30) * 0.0002 + max(0, i - 30) * 0.0003
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_backtest_config'`

- [ ] **Step 3: Modify `src/fxtrader/cli.py`**

Add to imports at the top (merge with existing):
```python
from fxtrader.config import BacktestConfig, load_config
from fxtrader.metrics.report import write_html_report, write_results_json
```

Add this helper after `run_backtest`:
```python
def run_backtest_config(config: BacktestConfig, data_dir: str | Path):
    """Run a backtest from a BacktestConfig. Returns (BacktestResult, metrics)."""
    return run_backtest(data_dir=data_dir, **config.to_run_kwargs())
```

REPLACE the existing `backtest` command with this version (adds `--config`, `--json`, `--html`; makes symbol/start/end optional):
```python
@app.command()
def backtest(
    symbol: str = typer.Option(None, help="Single symbol, e.g. EURUSD (or use --config)"),
    start: str = typer.Option(None, help="Inclusive start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="Exclusive end date YYYY-MM-DD"),
    data_dir: str = typer.Option("data", help="Parquet store root"),
    fast: int = typer.Option(20, help="Fast MA window (bars)"),
    slow: int = typer.Option(50, help="Slow MA window (bars)"),
    size: float = typer.Option(10_000, help="Position size (base units)"),
    bar_minutes: int = typer.Option(1, help="Bar interval in minutes"),
    max_position: float = typer.Option(0.0, help="Max abs position per symbol (0 = unlimited)"),
    daily_loss_limit: float = typer.Option(0.0, help="Daily-loss kill switch (0 = disabled)"),
    config: str = typer.Option(None, "--config", help="YAML run config (supersedes the flags above)"),
    json_out: str = typer.Option(None, "--json", help="Write results.json to this path"),
    html_out: str = typer.Option(None, "--html", help="Write an HTML report to this path"),
) -> None:
    """Backtest the MA-crossover strategy over stored ticks."""
    if config is not None:
        cfg = load_config(config)
    else:
        if not (symbol and start and end):
            raise typer.BadParameter("provide --config, or --symbol/--start/--end")
        cfg = BacktestConfig(
            symbol=symbol, start=start, end=end, fast=fast, slow=slow, size=size,
            bar_minutes=bar_minutes,
            daily_loss_limit=daily_loss_limit or None,
            max_position=max_position or None,
        )

    result, metrics = run_backtest_config(cfg, data_dir=data_dir)

    typer.echo(f"symbol={cfg.symbol}  bars={cfg.bar_minutes}m  MA({cfg.fast}/{cfg.slow})")
    typer.echo(f"  trades        : {metrics['num_trades']}")
    typer.echo(f"  win rate      : {metrics['win_rate']:.1%}")
    typer.echo(f"  total return  : {metrics['total_return']:.2%}")
    typer.echo(f"  CAGR          : {metrics['cagr']:.2%}")
    typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
    typer.echo(f"  Sortino       : {metrics['sortino']:.2f}")
    typer.echo(f"  max drawdown  : {metrics['max_drawdown']:.2%}")
    typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
    typer.echo(f"  ruined        : {metrics['ruined']}")
    typer.echo(f"  final equity  : {metrics['final_equity']:,.2f}")

    if json_out:
        write_results_json(json_out, cfg, result, metrics)
        typer.echo(f"wrote {json_out}")
    if html_out:
        write_html_report(html_out, cfg, result, metrics)
        typer.echo(f"wrote {html_out}")
```

Note: the old `backtest` body imported `run_backtest` directly; that helper still exists and is now called via `run_backtest_config`. Do not delete `run_backtest`.

- [ ] **Step 4: Run the CLI test + full suite + lint**

Run: `.venv/bin/pytest tests/test_cli_report.py -v` → 1 passed
Run: `.venv/bin/pytest -q` → all pass
Run: `.venv/bin/ruff check src/fxtrader` → clean

- [ ] **Step 5: Write `config/ma_eurusd.yaml`**

```yaml
symbol: EURUSD
start: 2024-01-02
end: 2024-02-01
fast: 20
slow: 50
bar_minutes: 5
size: 10000
daily_loss_limit: 5000
max_position: 50000
```

- [ ] **Step 6: Real-data end-to-end — produce actual report files**

Run: `.venv/bin/fxtrader backtest --config config/ma_eurusd.yaml --data-dir data --json out/eurusd.json --html out/eurusd.html`
(Create `out/` first if needed: `mkdir -p out`. `out/` is build output — add `out/` to `.gitignore`.)
Expected: prints the metrics summary AND writes `out/eurusd.json` + `out/eurusd.html`. Confirm `out/eurusd.json` parses and contains `metrics`, and `out/eurusd.html` contains a base64 PNG. Report the metrics + that both files were written.

- [ ] **Step 7: Commit**

```bash
echo "out/" >> .gitignore
git add src/fxtrader/cli.py tests/test_cli_report.py config/ma_eurusd.yaml .gitignore
git commit -m "feat: backtest --config/--json/--html + example config"
```

---

## Self-review

**Spec coverage (P2b portion of design §5.6/§5.7/§10):** machine-readable `results.json` for the agent (Task 2), HTML report with equity-curve plot (Task 3), reproducible YAML/pydantic config (Task 1), CLI wiring `--config/--json/--html` (Task 4). ✓ Still deferred (later cycles): limit/stop orders (P3 or when a strategy needs them), walk-forward (folds into P4's optimizer).

**Placeholder scan:** none — all code/tests complete. The matplotlib Agg backend is set before pyplot import; `# noqa: E402` documented.

**Type/signature consistency:** `BacktestConfig(...).to_run_kwargs()`; `load_config(path) -> BacktestConfig`; `build_results(config, result, metrics) -> dict`; `write_results_json(path, config, result, metrics)`; `write_html_report(path, config, result, metrics)`; `run_backtest_config(config, data_dir) -> (result, metrics)`. `run_backtest` (P2a) is reused unchanged. Consistent across tasks.

**Scope:** Produces reproducible, reportable backtests with agent-readable output. Limit/stop orders and walk-forward remain future cycles.
