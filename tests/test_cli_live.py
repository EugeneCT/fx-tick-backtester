import pytest
import typer

from fxtrader.cli import build_live_engine
from fxtrader.engine.live import TradingMode


def test_build_live_engine_paper():
    eng = build_live_engine(
        symbol="EURUSD", mode=TradingMode.PAPER,
        fast=20, slow=50, size=10_000, host="127.0.0.1", port=7497, client_id=1,
        daily_loss_limit=5_000, max_position=100_000, state_path=None,
    )
    assert eng.mode is TradingMode.PAPER
    assert eng.broker.port == 7497
    assert eng.broker.symbols == ["EURUSD"]


def test_live_command_blocks_live_without_flag():
    from fxtrader.cli import _resolve_mode
    assert _resolve_mode("paper", confirm_live=False) is TradingMode.PAPER
    assert _resolve_mode("dry-run", confirm_live=False) is TradingMode.DRY_RUN
    with pytest.raises(typer.BadParameter):
        _resolve_mode("live", confirm_live=False)
    assert _resolve_mode("live", confirm_live=True) is TradingMode.LIVE


def test_resolve_mode_rejects_bogus_mode_cleanly():
    from fxtrader.cli import _resolve_mode
    # A bad mode string must surface a clean typer.BadParameter, not a raw ValueError.
    with pytest.raises(typer.BadParameter):
        _resolve_mode("bogus", confirm_live=False)
