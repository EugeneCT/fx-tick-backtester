# fxtrader

Tick-driven FX backtesting and live trading platform.

## Setup

```bash
pyenv install -s 3.12.7
pyenv local 3.12.7
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
fxtrader download --symbols EURUSD,GBPUSD,USDJPY --start 2024-01-01 --end 2024-01-31
```

Data is written to a partitioned Parquet store under `./data` (override with `--data-dir`).
