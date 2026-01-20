from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from .config import BacktestConfig
from .evaluation import compute_equity_curve
from .signals import generate_crossover_signals


def _metrics_from_analyzers(strategy) -> Dict[str, float]:
    sharpe = strategy.analyzers.sharpe.get_analysis().get("sharperatio")
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()
    trades = strategy.analyzers.trades.get_analysis()

    return {
        "sharpe": sharpe if sharpe is not None else float("nan"),
        "max_drawdown": drawdown.get("max", {}).get("drawdown", float("nan")),
        "cumulative_return": returns.get("rtot", float("nan")),
        "total_trades": trades.get("total", {}).get("total", 0),
    }


def get_metrics(strategy) -> Dict[str, float]:
    return _metrics_from_analyzers(strategy)


def print_metrics(config: BacktestConfig, strategy) -> None:
    metrics = _metrics_from_analyzers(strategy)
    config_dict = asdict(config)
    print("\nBacktest Summary")
    print("Symbol:", config_dict["symbol"])
    print("Date Range:", f"{config_dict['start']} -> {config_dict['end']}")
    print("Params:", f"short_ma={config_dict['short_ma']} long_ma={config_dict['long_ma']}")
    print("Sharpe:", f"{metrics['sharpe']:.2f}")
    print("Max Drawdown (%):", f"{metrics['max_drawdown']:.2f}")
    print("Cumulative Return:", f"{metrics['cumulative_return']:.2f}")
    print("Total Trades:", metrics["total_trades"])


def build_report_figure(df: pd.DataFrame, config: BacktestConfig) -> plt.Figure:
    signals = generate_crossover_signals(
        df,
        config.short_ma,
        config.long_ma,
        trend_filter=config.trend_filter,
        atr_period=config.atr_period,
        atr_stop_mult=config.atr_stop_mult,
    )
    equity = compute_equity_curve(df, signals, config.cash)

    short_sma = df["close"].rolling(window=config.short_ma).mean()
    long_sma = df["close"].rolling(window=config.long_ma).mean()

    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        f"Momentum Strategy Report - {config.symbol}",
        fontsize=16,
        fontweight="bold",
    )

    grid = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.15)
    ax_price = fig.add_subplot(grid[0])
    ax_volume = fig.add_subplot(grid[1], sharex=ax_price)
    ax_equity = fig.add_subplot(grid[2], sharex=ax_price)

    ax_price.plot(df.index, df["close"], color="#1f2937", linewidth=1.6, label="Close")
    ax_price.plot(df.index, short_sma, color="#2563eb", linewidth=1.2, label=f"SMA {config.short_ma}")
    ax_price.plot(df.index, long_sma, color="#f97316", linewidth=1.2, label=f"SMA {config.long_ma}")

    buys = signals[signals == 1].index
    sells = signals[signals == -1].index
    ax_price.scatter(buys, df.loc[buys, "close"], marker="^", color="#16a34a", s=60, label="Buy")
    ax_price.scatter(sells, df.loc[sells, "close"], marker="v", color="#dc2626", s=60, label="Sell")

    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", frameon=False, ncol=3)
    ax_price.grid(alpha=0.3)

    ax_volume.bar(df.index, df["volume"], color="#94a3b8", width=1.0)
    ax_volume.set_ylabel("Volume")
    ax_volume.grid(alpha=0.2)

    ax_equity.plot(df.index, equity, color="#0f766e", linewidth=1.6, label="Equity")
    ax_equity.set_ylabel("Equity")
    ax_equity.legend(loc="upper left", frameon=False)
    ax_equity.grid(alpha=0.3)

    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_volume.get_xticklabels(), visible=False)
    return fig


def render_report(df: pd.DataFrame, config: BacktestConfig) -> None:
    fig = build_report_figure(df, config)
    plt.show()
