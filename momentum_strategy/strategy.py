from __future__ import annotations

import math

import backtrader as bt


class MomentumStrategy(bt.Strategy):
    params = (
        ("short_ma", 50),
        ("long_ma", 200),
        ("trend_filter", True),
        ("atr_period", 14),
        ("atr_stop_mult", 2.0),
        ("risk_per_trade", 0.01),
    )

    def __init__(self) -> None:
        self.sma_short = bt.indicators.SimpleMovingAverage(period=self.p.short_ma)
        self.sma_long = bt.indicators.SimpleMovingAverage(period=self.p.long_ma)
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        self.stop_price = None

    def next(self) -> None:
        if not self.position:
            if self.crossover > 0:
                if self.p.trend_filter and self.data.close[0] <= self.sma_long[0]:
                    return
                if self.p.atr_stop_mult > 0 and math.isnan(self.atr[0]):
                    return
                size = 1
                if self.p.atr_stop_mult > 0:
                    risk_amount = self.broker.getvalue() * self.p.risk_per_trade
                    stop_dist = self.atr[0] * self.p.atr_stop_mult
                    if stop_dist > 0:
                        size = int(risk_amount / stop_dist)
                max_size = int(self.broker.getcash() / self.data.close[0])
                size = min(size, max_size)
                if size <= 0:
                    return
                self.buy(size=size)
                if self.p.atr_stop_mult > 0:
                    self.stop_price = self.data.close[0] - (self.atr[0] * self.p.atr_stop_mult)
        else:
            if self.p.atr_stop_mult > 0 and not math.isnan(self.atr[0]):
                new_stop = self.data.close[0] - (self.atr[0] * self.p.atr_stop_mult)
                if self.stop_price is None:
                    self.stop_price = new_stop
                else:
                    self.stop_price = max(self.stop_price, new_stop)
                if self.data.close[0] < self.stop_price:
                    self.sell()
                    self.stop_price = None
                    return
            if self.crossover < 0:
                self.sell()
                self.stop_price = None
