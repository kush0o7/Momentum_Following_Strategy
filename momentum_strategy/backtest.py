from __future__ import annotations

from dataclasses import asdict
import logging

import backtrader as bt

from .config import BacktestConfig
from .data import fetch_yfinance_data
from .strategy import MomentumStrategy

logger = logging.getLogger(__name__)


def run_backtest(config: BacktestConfig) -> tuple[bt.Cerebro, bt.Strategy, "pd.DataFrame"]:
    config.validate()
    logger.info("Starting backtest for %s", config.symbol)

    df = fetch_yfinance_data(config.symbol, config.start, config.end)
    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(
        MomentumStrategy,
        short_ma=config.short_ma,
        long_ma=config.long_ma,
        trend_filter=config.trend_filter,
        atr_period=config.atr_period,
        atr_stop_mult=config.atr_stop_mult,
        risk_per_trade=config.risk_per_trade,
        rsi_period=config.rsi_period,
        rsi_entry=config.rsi_entry,
        rsi_exit=config.rsi_exit,
        atr_vol_min=config.atr_vol_min,
    )
    cerebro.broker.setcash(config.cash)
    cerebro.broker.setcommission(commission=config.commission)
    if config.slippage_perc > 0:
        cerebro.broker.set_slippage_perc(
            config.slippage_perc,
            slip_open=True,
            slip_limit=True,
            slip_match=True,
            slip_out=True,
        )

    logger.debug("Backtest config: %s", asdict(config))
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    return cerebro, results[0], df
