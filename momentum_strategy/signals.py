from __future__ import annotations

import pandas as pd


def generate_crossover_signals(
    df: pd.DataFrame,
    short_ma: int,
    long_ma: int,
    trend_filter: bool = True,
    atr_period: int = 14,
    atr_stop_mult: float = 2.0,
) -> pd.Series:
    """Return 1 for long, -1 for exit, 0 otherwise using SMA crossovers and ATR stop."""
    required = {"close", "high", "low"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame must contain columns: {sorted(missing)}")
    if short_ma <= 0 or long_ma <= 0:
        raise ValueError("Moving average periods must be positive.")
    if short_ma >= long_ma:
        raise ValueError("short_ma must be smaller than long_ma.")
    if atr_period <= 0:
        raise ValueError("atr_period must be positive.")
    if atr_stop_mult < 0:
        raise ValueError("atr_stop_mult cannot be negative.")

    short_sma = df["close"].rolling(window=short_ma, min_periods=short_ma).mean()
    long_sma = df["close"].rolling(window=long_ma, min_periods=long_ma).mean()
    crossover = (short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1))
    crossunder = (short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1))

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()

    signals = pd.Series(0, index=df.index)
    position = 0
    stop_price = None

    for idx in df.index:
        if position == 0:
            if crossover.loc[idx]:
                if trend_filter and df.loc[idx, "close"] <= long_sma.loc[idx]:
                    continue
                if atr_stop_mult > 0 and pd.isna(atr.loc[idx]):
                    continue
                position = 1
                signals.loc[idx] = 1
                if atr_stop_mult > 0:
                    stop_price = df.loc[idx, "close"] - atr.loc[idx] * atr_stop_mult
        else:
            if atr_stop_mult > 0 and not pd.isna(atr.loc[idx]):
                new_stop = df.loc[idx, "close"] - atr.loc[idx] * atr_stop_mult
                stop_price = new_stop if stop_price is None else max(stop_price, new_stop)
                if df.loc[idx, "close"] < stop_price:
                    position = 0
                    signals.loc[idx] = -1
                    stop_price = None
                    continue
            if crossunder.loc[idx]:
                position = 0
                signals.loc[idx] = -1
                stop_price = None

    return signals
