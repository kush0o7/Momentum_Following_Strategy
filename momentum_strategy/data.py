from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS: Iterable[str] = ("Open", "High", "Low", "Close", "Volume")


def fetch_yfinance_data(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch and normalize OHLCV data from Yahoo Finance."""
    df = yf.download(symbol, start=start.isoformat(), end=end.isoformat(), progress=False)
    if df.empty:
        raise ValueError("No data retrieved from Yahoo Finance.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df.index = pd.to_datetime(df.index)
    df = df[list(REQUIRED_COLUMNS)]
    df.columns = ["open", "high", "low", "close", "volume"]
    return df
