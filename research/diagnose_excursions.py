"""Run the trend strategy (no exits) on in-sample EURUSD and print the MAE/MFE summary."""
from datetime import datetime, timedelta, timezone
import json

from fxtrader.cli import run_backtest
from fxtrader.data.feed import HistoricalTickFeed, Resampler
from fxtrader.data.store import TickStore
from fxtrader.metrics.excursions import summarize_excursions, trade_excursions
from fxtrader.core.instruments import pip_size

# Window and strategy parameters
SYMBOL = "EURUSD"
BAR_MIN = 30
FAST, SLOW, TREND = 20, 100, 200

# Smoke window (short for fast testing)
# For real study, widen to in-sample: 2023-06-01 → 2025-06-01
START = datetime(2025, 1, 6, tzinfo=timezone.utc)
END = datetime(2025, 2, 1, tzinfo=timezone.utc)

# Run backtest: trend strategy with no exits
result, _metrics = run_backtest(
    symbol=SYMBOL, start=START, end=END, data_dir="data",
    fast=FAST, slow=SLOW, size=10_000, bar_minutes=BAR_MIN,
    strategy_name="trend", trend=TREND, exit_kind="none",
)

# Rebuild the same bars the engine saw, for excursion measurement
res = Resampler(timedelta(minutes=BAR_MIN))
bars = []
for tick in HistoricalTickFeed(TickStore("data"), [SYMBOL], START, END):
    bar = res.update(tick)
    if bar:
        bars.append(bar)
last = res.flush()
if last:
    bars.append(last)

# Compute MAE/MFE excursions
recs = trade_excursions(result.fills, bars, pip_size(SYMBOL))
print(json.dumps(summarize_excursions(recs), indent=2))
