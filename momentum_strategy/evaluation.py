from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from .signals import generate_crossover_signals
from .config import BacktestConfig
from .data import fetch_yfinance_data


@dataclass(frozen=True)
class MetricResult:
    total_return: float
    sharpe: float
    max_drawdown: float
    trades: int
    cagr: float


def compute_equity_curve(df: pd.DataFrame, signals: pd.Series, starting_cash: float) -> pd.Series:
    returns = df["close"].pct_change().fillna(0.0)
    position = signals.replace({1: 1, -1: 0}).ffill().fillna(0)
    strategy_returns = position.shift(1).fillna(0) * returns
    equity = (1 + strategy_returns).cumprod() * starting_cash
    return equity


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min() * -100.0)


def _sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return float("nan")
    return float(np.sqrt(252) * returns.mean() / returns.std())


def _cagr(equity: pd.Series, index: pd.Index) -> float:
    if equity.empty:
        return float("nan")
    days = (index[-1] - index[0]).days
    if days <= 0:
        return float("nan")
    years = days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def compute_metrics(df: pd.DataFrame, signals: pd.Series, starting_cash: float) -> MetricResult:
    equity = compute_equity_curve(df, signals, starting_cash)
    returns = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    return MetricResult(
        total_return=total_return,
        sharpe=_sharpe(returns),
        max_drawdown=_max_drawdown(equity),
        trades=int((signals == 1).sum()),
        cagr=_cagr(equity, equity.index),
    )


def buy_hold_metrics(df: pd.DataFrame, starting_cash: float) -> MetricResult:
    equity = starting_cash * (df["close"] / df["close"].iloc[0])
    returns = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    return MetricResult(
        total_return=total_return,
        sharpe=_sharpe(returns),
        max_drawdown=_max_drawdown(equity),
        trades=1,
        cagr=_cagr(equity, equity.index),
    )


def run_grid_search(
    df: pd.DataFrame,
    starting_cash: float,
    short_list: Iterable[int],
    long_list: Iterable[int],
    trend_filter_list: Iterable[bool],
    atr_period_list: Iterable[int],
    atr_stop_mult_list: Iterable[float],
    max_combos: int = 200,
) -> pd.DataFrame:
    rows = []
    combos = list(product(short_list, long_list, trend_filter_list, atr_period_list, atr_stop_mult_list))
    for short_ma, long_ma, trend_filter, atr_period, atr_stop_mult in combos[:max_combos]:
        if short_ma >= long_ma:
            continue
        signals = generate_crossover_signals(
            df,
            short_ma=short_ma,
            long_ma=long_ma,
            trend_filter=trend_filter,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
        )
        metrics = compute_metrics(df, signals, starting_cash)
        rows.append(
            {
                "short_ma": short_ma,
                "long_ma": long_ma,
                "trend_filter": trend_filter,
                "atr_period": atr_period,
                "atr_stop_mult": atr_stop_mult,
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "trades": metrics.trades,
            }
        )

    return pd.DataFrame(rows).sort_values(by=["sharpe", "total_return"], ascending=False)


def walk_forward_report(
    config: BacktestConfig, train_days: int, test_days: int
) -> pd.DataFrame:
    df = fetch_yfinance_data(config.symbol, config.start, config.end)
    results = []
    start_idx = df.index[0]
    end_idx = df.index[-1]
    current_start = start_idx

    while current_start < end_idx:
        train_end = current_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        train_df = df.loc[current_start:train_end]
        test_df = df.loc[train_end:test_end]
        if len(train_df) < max(config.long_ma, config.atr_period) or len(test_df) == 0:
            break

        signals = generate_crossover_signals(
            train_df,
            short_ma=config.short_ma,
            long_ma=config.long_ma,
            trend_filter=config.trend_filter,
            atr_period=config.atr_period,
            atr_stop_mult=config.atr_stop_mult,
        )
        train_metrics = compute_metrics(train_df, signals, config.cash)

        test_signals = generate_crossover_signals(
            test_df,
            short_ma=config.short_ma,
            long_ma=config.long_ma,
            trend_filter=config.trend_filter,
            atr_period=config.atr_period,
            atr_stop_mult=config.atr_stop_mult,
        )
        test_metrics = compute_metrics(test_df, test_signals, config.cash)

        results.append(
            {
                "train_start": train_df.index[0].date(),
                "train_end": train_df.index[-1].date(),
                "test_start": test_df.index[0].date(),
                "test_end": test_df.index[-1].date(),
                "train_sharpe": train_metrics.sharpe,
                "train_cagr": train_metrics.cagr,
                "test_sharpe": test_metrics.sharpe,
                "test_cagr": test_metrics.cagr,
                "test_max_drawdown": test_metrics.max_drawdown,
                "test_trades": test_metrics.trades,
            }
        )

        current_start = current_start + pd.Timedelta(days=test_days)

    return pd.DataFrame(results)
