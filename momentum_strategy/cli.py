from __future__ import annotations

import argparse
from datetime import datetime

from .backtest import run_backtest
from .config import BacktestConfig
from .logging_utils import configure_logging
from .report import print_metrics, render_report


def _parse_date(value: str) -> datetime.date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must be YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Momentum strategy backtester")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument("--short-ma", type=int, default=50)
    parser.add_argument("--long-ma", type=int, default=200)
    parser.add_argument("--trend-filter", action="store_true", default=True)
    parser.add_argument("--no-trend-filter", action="store_false", dest="trend_filter")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--atr-stop-mult", type=float, default=2.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--slippage-perc", type=float, default=0.0005)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--wf-train-days", type=int, default=365)
    parser.add_argument("--wf-test-days", type=int, default=180)
    parser.add_argument("--walk-forward-opt", action="store_true")
    parser.add_argument("--opt-short-min", type=int, default=10)
    parser.add_argument("--opt-short-max", type=int, default=30)
    parser.add_argument("--opt-long-min", type=int, default=80)
    parser.add_argument("--opt-long-max", type=int, default=150)
    parser.add_argument("--opt-atr-stop-min", type=float, default=1.5)
    parser.add_argument("--opt-atr-stop-max", type=float, default=3.0)
    parser.add_argument("--opt-max-combos", type=int, default=200)
    parser.add_argument("--portfolio", action="store_true")
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA,TSLA")
    parser.add_argument("--rebalance-days", type=int, default=21)
    parser.add_argument("--costs-file", default=None)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    config = BacktestConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        short_ma=args.short_ma,
        long_ma=args.long_ma,
        trend_filter=args.trend_filter,
        atr_period=args.atr_period,
        atr_stop_mult=args.atr_stop_mult,
        risk_per_trade=args.risk_per_trade,
        slippage_perc=args.slippage_perc,
        cash=args.cash,
        commission=args.commission,
        plot=args.plot,
        log_level=args.log_level,
    )

    if args.walk_forward_opt:
        from .evaluation import walk_forward_optimize

        short_list = list(range(args.opt_short_min, args.opt_short_max + 1, 2))
        long_list = list(range(args.opt_long_min, args.opt_long_max + 1, 5))
        atr_stop_list = [
            round(args.opt_atr_stop_min, 2),
            round((args.opt_atr_stop_min + args.opt_atr_stop_max) / 2, 2),
            round(args.opt_atr_stop_max, 2),
        ]
        results = walk_forward_optimize(
            config,
            train_days=args.wf_train_days,
            test_days=args.wf_test_days,
            short_list=short_list,
            long_list=long_list,
            atr_stop_list=atr_stop_list,
            trend_filter_list=[True, False],
            atr_period_list=[config.atr_period],
            max_combos=args.opt_max_combos,
        )
        print(results)
    elif args.walk_forward:
        from .evaluation import walk_forward_report

        results = walk_forward_report(
            config,
            train_days=args.wf_train_days,
            test_days=args.wf_test_days,
        )
        print(results)
    elif args.portfolio:
        from .data import fetch_yfinance_data
        from .evaluation import portfolio_metrics, simulate_portfolio

        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        data_map = {sym: fetch_yfinance_data(sym, config.start, config.end) for sym in symbols}

        asset_costs = None
        if args.costs_file:
            import csv

            asset_costs = {}
            with open(args.costs_file, "r", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    symbol = row.get("symbol", "").strip().upper()
                    if not symbol:
                        continue
                    asset_costs[symbol] = {
                        "commission": float(row.get("commission", config.commission)),
                        "slippage": float(row.get("slippage", config.slippage_perc)),
                    }

        equity = simulate_portfolio(
            data_map,
            config,
            rebalance_days=args.rebalance_days,
            asset_costs=asset_costs,
        )
        metrics = portfolio_metrics(equity)
        print("Portfolio Summary")
        print(f"Sharpe: {metrics.sharpe:.2f}")
        print(f"Max Drawdown (%): {metrics.max_drawdown:.2f}")
        print(f"CAGR: {metrics.cagr:.2%}")
    else:
        cerebro, strategy, df = run_backtest(config)
        print_metrics(config, strategy)
        if config.plot:
            try:
                import matplotlib  # noqa: F401
            except ImportError as exc:
                raise SystemExit(
                    "Plotting requires matplotlib. Install it with: pip install matplotlib"
                ) from exc
            render_report(df, config)


if __name__ == "__main__":
    main()
