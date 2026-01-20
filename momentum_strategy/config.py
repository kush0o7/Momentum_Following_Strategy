from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    start: date
    end: date
    short_ma: int = 50
    long_ma: int = 200
    trend_filter: bool = True
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    risk_per_trade: float = 0.01
    slippage_perc: float = 0.0005
    rsi_period: int = 14
    rsi_entry: float = 55.0
    rsi_exit: float = 45.0
    atr_vol_min: float = 0.0
    cash: float = 10000.0
    commission: float = 0.001
    plot: bool = False
    log_level: str = "INFO"

    def validate(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Symbol must be provided.")
        if self.start >= self.end:
            raise ValueError("Start date must be before end date.")
        if self.short_ma <= 0 or self.long_ma <= 0:
            raise ValueError("Moving average periods must be positive.")
        if self.short_ma >= self.long_ma:
            raise ValueError("short_ma must be smaller than long_ma.")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive.")
        if self.atr_stop_mult < 0:
            raise ValueError("atr_stop_mult cannot be negative.")
        if not (0 < self.risk_per_trade <= 1):
            raise ValueError("risk_per_trade must be between 0 and 1.")
        if self.slippage_perc < 0:
            raise ValueError("slippage_perc cannot be negative.")
        if self.rsi_period <= 0:
            raise ValueError("rsi_period must be positive.")
        if not (0 <= self.rsi_exit <= 100) or not (0 <= self.rsi_entry <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100.")
        if self.rsi_exit >= self.rsi_entry:
            raise ValueError("rsi_exit must be less than rsi_entry.")
        if self.atr_vol_min < 0:
            raise ValueError("atr_vol_min cannot be negative.")
        if self.cash <= 0:
            raise ValueError("cash must be positive.")
        if self.commission < 0:
            raise ValueError("commission cannot be negative.")
