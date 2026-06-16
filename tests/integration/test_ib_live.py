import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FXTRADER_IB_TESTS") != "1",
    reason="set FXTRADER_IB_TESTS=1 with IB Gateway/TWS running (paper) to run",
)


def test_connect_and_request_positions_paper():
    # Requires IB Gateway/TWS running on the paper port (default 7497 TWS / 4002 Gateway).
    from fxtrader.broker.ib import IBBroker

    broker = IBBroker(host="127.0.0.1", port=int(os.environ.get("FXTRADER_IB_PORT", "7497")),
                      client_id=99, symbols=["EURUSD"])
    broker.connect()
    try:
        positions = broker.positions()
        assert isinstance(positions, dict)
    finally:
        broker.disconnect()
