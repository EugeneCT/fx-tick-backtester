import pytest

from fxtrader.broker.base import Broker


def test_broker_is_abstract():
    with pytest.raises(TypeError):
        Broker()  # abstract methods not implemented


def test_concrete_subclass_can_register_and_invoke_fill_callback():
    class Dummy(Broker):
        def update_market(self, tick):
            pass

        def submit(self, order):
            self._emit_fill("a-fill")

    got = []
    b = Dummy()
    b.set_fill_callback(got.append)
    b.submit(object())
    assert got == ["a-fill"]
