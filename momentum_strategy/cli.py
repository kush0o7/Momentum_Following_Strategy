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

    if args.walk_forward:
        from .evaluation import walk_forward_report

        results = walk_forward_report(
            config,
            train_days=args.wf_train_days,
            test_days=args.wf_test_days,
        )
        print(results)
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
