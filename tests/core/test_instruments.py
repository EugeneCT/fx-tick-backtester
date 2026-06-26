from fxtrader.core.instruments import pip_size


def test_pip_size_default_is_one_ten_thousandth():
    assert pip_size("EURUSD") == 0.0001
    assert pip_size("gbpusd") == 0.0001


def test_pip_size_jpy_pairs_are_one_hundredth():
    assert pip_size("USDJPY") == 0.01
    assert pip_size("eurjpy") == 0.01


def test_pip_size_gold_is_one_tenth():
    assert pip_size("XAUUSD") == 0.1
    assert pip_size("xauusd") == 0.1


def test_pip_size_audusd_is_default():
    assert pip_size("AUDUSD") == 0.0001
    assert pip_size("audusd") == 0.0001
