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

## Live trading on Interactive Brokers (paper first)

The live path needs IB Gateway or TWS running and an IB account (a free paper account works).

1. Install the live extra: `pip install -e ".[live]"`.
2. Download **IB Gateway** (lighter) or **TWS**, log in to your **paper** account.
3. In Gateway/TWS: Configure → API → Settings → enable *ActiveX and Socket Clients*; note the socket port (TWS paper `7497`, Gateway paper `4002`); add `127.0.0.1` to trusted IPs.
4. Dry run first (computes orders, sends nothing):

       fxtrader live --symbol EURUSD --mode dry-run --port 7497

5. Then paper:

       fxtrader live --symbol EURUSD --mode paper --port 7497 --daily-loss-limit 500 --max-position 50000

6. `LIVE` (real money) is gated: it requires `--mode live --i-understand-live`. Do not enable it until you've validated on paper.

Notes: live FX quotes may require an IB market-data subscription; paper accounts mirror the linked live account's subscriptions. Engine state is written to `live_state.json`. To run the opt-in connection smoke test: `FXTRADER_IB_TESTS=1 FXTRADER_IB_PORT=7497 pytest tests/integration/test_ib_live.py`.
