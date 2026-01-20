import pandas as pd

from momentum_strategy.signals import generate_crossover_signals


def test_generate_crossover_signals_basic():
    prices = [1, 1, 1, 2, 3, 4, 3, 2, 1]
    df = pd.DataFrame({"close": prices, "high": prices, "low": prices})

    signals = generate_crossover_signals(
        df,
        short_ma=2,
        long_ma=3,
        trend_filter=False,
        atr_period=2,
        atr_stop_mult=0,
        rsi_period=2,
        rsi_entry=1,
        rsi_exit=0,
    )

    assert 1 in signals.values
    assert -1 in signals.values
