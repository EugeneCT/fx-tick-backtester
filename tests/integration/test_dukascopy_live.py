import os
from datetime import datetime, timezone

import pytest

from fxtrader.data.dukascopy import fetch_hour

pytestmark = pytest.mark.skipif(
    os.environ.get("FXTRADER_NETWORK_TESTS") != "1",
    reason="set FXTRADER_NETWORK_TESTS=1 to run live Dukascopy download",
)


def test_fetch_one_real_eurusd_hour():
    # A Tuesday mid-session hour that reliably has ticks.
    hour = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    ticks = fetch_hour("EURUSD", hour)
    assert len(ticks) > 100
    assert all(t.ask >= t.bid for t in ticks)
    assert all(1.0 < t.bid < 1.3 for t in ticks)  # sane EUR/USD range
