from __future__ import annotations

import time
from dataclasses import asdict

import pandas as pd
import yfinance as yf

from .config import BacktestConfig
from .signals import generate_crossover_signals


def _fetch_intraday(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    period = f"{lookback_days}d"
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No data returned for live mode.")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


def run_live(config: BacktestConfig, interval: str = "5m", lookback_days: int = 30, refresh_sec: int = 60) -> None:
    print("Starting live signal monitor")
    print(asdict(config))
    print("Press Ctrl+C to stop.\n")

    last_signal = 0
    while True:
        df = _fetch_intraday(config.symbol, interval, lookback_days)
        signals = generate_crossover_signals(
            df,
            config.short_ma,
            config.long_ma,
            trend_filter=config.trend_filter,
            atr_period=config.atr_period,
            atr_stop_mult=config.atr_stop_mult,
            rsi_period=config.rsi_period,
            rsi_entry=config.rsi_entry,
            rsi_exit=config.rsi_exit,
            atr_vol_min=config.atr_vol_min,
        )
        latest = signals.iloc[-1]
        if latest != 0 and latest != last_signal:
            action = "BUY" if latest == 1 else "SELL"
            print(f"{df.index[-1]} {action} signal at {df['close'].iloc[-1]:.2f}")
            last_signal = latest
        else:
            print(f"{df.index[-1]} no new signal; last close {df['close'].iloc[-1]:.2f}")

        time.sleep(refresh_sec)
