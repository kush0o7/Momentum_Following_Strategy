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


def compute_equity_curve(
    df: pd.DataFrame,
    signals: pd.Series,
    starting_cash: float,
    commission_perc: float = 0.0,
    slippage_perc: float = 0.0,
) -> pd.Series:
    returns = df["close"].pct_change().fillna(0.0)
    position = signals.replace({1: 1, -1: 0}).ffill().fillna(0)
    turnover = position.diff().abs().fillna(0.0)
    cost_rate = commission_perc + slippage_perc
    strategy_returns = position.shift(1).fillna(0) * returns - cost_rate * turnover
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


def compute_metrics(
    df: pd.DataFrame,
    signals: pd.Series,
    starting_cash: float,
    commission_perc: float = 0.0,
    slippage_perc: float = 0.0,
) -> MetricResult:
    equity = compute_equity_curve(
        df,
        signals,
        starting_cash,
        commission_perc=commission_perc,
        slippage_perc=slippage_perc,
    )
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
    commission_perc: float = 0.0,
    slippage_perc: float = 0.0,
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
        metrics = compute_metrics(
            df,
            signals,
            starting_cash,
            commission_perc=commission_perc,
            slippage_perc=slippage_perc,
        )
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
        train_metrics = compute_metrics(
            train_df,
            signals,
            config.cash,
            commission_perc=config.commission,
            slippage_perc=config.slippage_perc,
        )

        test_signals = generate_crossover_signals(
            test_df,
            short_ma=config.short_ma,
            long_ma=config.long_ma,
            trend_filter=config.trend_filter,
            atr_period=config.atr_period,
            atr_stop_mult=config.atr_stop_mult,
        )
        test_metrics = compute_metrics(
            test_df,
            test_signals,
            config.cash,
            commission_perc=config.commission,
            slippage_perc=config.slippage_perc,
        )

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


def walk_forward_optimize(
    config: BacktestConfig,
    train_days: int,
    test_days: int,
    short_list: Iterable[int],
    long_list: Iterable[int],
    atr_stop_list: Iterable[float],
    trend_filter_list: Iterable[bool],
    atr_period_list: Iterable[int],
    max_combos: int = 200,
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

        leaderboard = run_grid_search(
            train_df,
            config.cash,
            short_list,
            long_list,
            trend_filter_list,
            atr_period_list,
            atr_stop_list,
            max_combos=max_combos,
            commission_perc=config.commission,
            slippage_perc=config.slippage_perc,
        )
        if leaderboard.empty:
            break

        best = leaderboard.iloc[0]
        test_signals = generate_crossover_signals(
            test_df,
            short_ma=int(best["short_ma"]),
            long_ma=int(best["long_ma"]),
            trend_filter=bool(best["trend_filter"]),
            atr_period=int(best["atr_period"]),
            atr_stop_mult=float(best["atr_stop_mult"]),
        )
        test_metrics = compute_metrics(
            test_df,
            test_signals,
            config.cash,
            commission_perc=config.commission,
            slippage_perc=config.slippage_perc,
        )

        results.append(
            {
                "train_start": train_df.index[0].date(),
                "train_end": train_df.index[-1].date(),
                "test_start": test_df.index[0].date(),
                "test_end": test_df.index[-1].date(),
                "best_short": int(best["short_ma"]),
                "best_long": int(best["long_ma"]),
                "best_trend_filter": bool(best["trend_filter"]),
                "best_atr_period": int(best["atr_period"]),
                "best_atr_stop": float(best["atr_stop_mult"]),
                "test_sharpe": test_metrics.sharpe,
                "test_cagr": test_metrics.cagr,
                "test_max_drawdown": test_metrics.max_drawdown,
                "test_trades": test_metrics.trades,
            }
        )

        current_start = current_start + pd.Timedelta(days=test_days)

    return pd.DataFrame(results)


def simulate_portfolio(
    data_map: dict[str, pd.DataFrame],
    config: BacktestConfig,
    rebalance_days: int = 21,
    asset_costs: dict[str, dict[str, float]] | None = None,
) -> pd.Series:
    aligned = {
        symbol: df[["close", "high", "low"]].copy()
        for symbol, df in data_map.items()
        if not df.empty
    }
    if not aligned:
        raise ValueError("No data available for portfolio simulation.")

    closes = pd.DataFrame({s: df["close"] for s, df in aligned.items()}).dropna()
    if closes.empty:
        raise ValueError("No overlapping dates across symbols.")

    returns = closes.pct_change().fillna(0.0)
    signals = {}
    for symbol, df in aligned.items():
        df = df.loc[closes.index]
        signals[symbol] = generate_crossover_signals(
            df,
            short_ma=config.short_ma,
            long_ma=config.long_ma,
            trend_filter=config.trend_filter,
            atr_period=config.atr_period,
            atr_stop_mult=config.atr_stop_mult,
        ).reindex(closes.index, fill_value=0)

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    rebalance_dates = closes.index[::rebalance_days]

    last_weights = pd.Series(0.0, index=closes.columns)
    for date in closes.index:
        if date in rebalance_dates:
            active = [s for s in closes.columns if signals[s].loc[date] == 1]
            if active:
                equal_weight = 1.0 / len(active)
                new_weights = pd.Series(0.0, index=closes.columns)
                new_weights[active] = equal_weight
            else:
                new_weights = pd.Series(0.0, index=closes.columns)
            last_weights = new_weights
        weights.loc[date] = last_weights

    portfolio_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)

    if asset_costs is None:
        asset_costs = {}
    cost_rates = {
        symbol: (asset_costs.get(symbol, {}).get("commission", config.commission)
        + asset_costs.get(symbol, {}).get("slippage", config.slippage_perc))
        for symbol in closes.columns
    }

    turnover = weights.diff().abs().fillna(0.0)
    cost_series = pd.Series(0.0, index=closes.index)
    for symbol in closes.columns:
        cost_series += turnover[symbol] * cost_rates[symbol]

    portfolio_returns = portfolio_returns - cost_series
    equity = (1 + portfolio_returns).cumprod() * config.cash
    return equity


def portfolio_metrics(equity: pd.Series) -> MetricResult:
    returns = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    return MetricResult(
        total_return=total_return,
        sharpe=_sharpe(returns),
        max_drawdown=_max_drawdown(equity),
        trades=0,
        cagr=_cagr(equity, equity.index),
    )
