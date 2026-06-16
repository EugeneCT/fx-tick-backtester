from fxtrader.broker.simulated import SimulatedBroker


def test_default_live_methods_are_safe_noops():
    b = SimulatedBroker()
    # These default methods must exist and not raise on the simulated broker.
    b.connect()
    b.disconnect()
    b.cancel(object())
    assert b.positions() == {}
    assert b.account_value() is None


def test_tick_callback_emits():
    b = SimulatedBroker()
    got = []
    b.set_tick_callback(got.append)
    b._emit_tick("a-tick")
    assert got == ["a-tick"]


def test_tick_callback_unset_is_safe():
    b = SimulatedBroker()
    b._emit_tick("ignored")  # no callback set -> must not raise
