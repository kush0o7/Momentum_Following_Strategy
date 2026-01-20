from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import streamlit as st
import yfinance as yf

from momentum_strategy.ai import train_ai_model
from momentum_strategy.backtest import run_backtest
from momentum_strategy.config import BacktestConfig
from momentum_strategy.data import fetch_yfinance_data
from momentum_strategy.evaluation import (
    buy_hold_metrics,
    compute_equity_curve,
    compute_metrics,
    run_grid_search,
    simulate_portfolio,
    portfolio_metrics,
    walk_forward_optimize,
)
from momentum_strategy.report import build_report_figure, get_metrics
from momentum_strategy.signals import generate_crossover_signals

st.set_page_config(page_title="Momentum Strategy Lab", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #0b0f1a;
  --panel: #121826;
  --accent: #38bdf8;
  --accent-2: #f97316;
  --text: #e2e8f0;
  --muted: #94a3b8;
}

.stApp {
  background: radial-gradient(circle at 10% 10%, #1f2937, #0b0f1a 45%, #0b0f1a 100%);
  color: var(--text);
}

.block-container {
  padding-top: 2rem;
}

.kpi-card {
  background: var(--panel);
  border: 1px solid rgba(148,163,184,0.2);
  border-radius: 14px;
  padding: 1rem 1.2rem;
  box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.hero {
  padding: 1.25rem 1.5rem;
  border-radius: 18px;
  background: linear-gradient(120deg, rgba(56,189,248,0.2), rgba(249,115,22,0.15));
  border: 1px solid rgba(56,189,248,0.25);
}

.badge {
  display: inline-block;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  background: rgba(56,189,248,0.2);
  color: var(--text);
  font-size: 0.8rem;
  margin-right: 0.5rem;
}

h1, h2, h3, h4 {
  color: var(--text);
}
</style>
""",
    unsafe_allow_html=True,
)

if "reset_widgets" not in st.session_state:
    st.session_state["reset_widgets"] = False

if st.sidebar.button("Reset Widgets", width="stretch"):
    for key in list(st.session_state.keys()):
        if key != "reset_widgets":
            del st.session_state[key]
    st.session_state["reset_widgets"] = True
    st.rerun()

st.markdown(
    """
<div class="hero">
  <span class="badge">Momentum Strategy Lab</span>
  <span class="badge">Backtesting</span>
  <span class="badge">Optimizer</span>
  <h1>Momentum Strategy Lab</h1>
  <p style="color:#cbd5f5">Scan, test, and benchmark momentum setups with a premium, recruiter-ready dashboard.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Primary Run")
    symbol = st.text_input("Symbol", value="AAPL")
    start = st.date_input("Start", value=dt.date(2020, 1, 1))
    end = st.date_input("End", value=dt.date(2023, 1, 1))
    short_ma = st.number_input("Short MA", min_value=2, max_value=200, value=20)
    long_ma = st.number_input("Long MA", min_value=10, max_value=400, value=100)
    trend_filter = st.checkbox("Trend Filter", value=True)
    atr_period = st.number_input("ATR Period", min_value=2, max_value=100, value=14)
    atr_stop_mult = st.number_input("ATR Stop Mult", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    rsi_period = st.number_input("RSI Period", min_value=2, max_value=50, value=14)
    rsi_entry = st.number_input("RSI Entry", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
    rsi_exit = st.number_input("RSI Exit", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
    atr_vol_min = st.number_input("ATR Vol Min", min_value=0.0, max_value=0.2, value=0.0, step=0.01)
    risk_per_trade = st.number_input("Risk Per Trade", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
    slippage_perc = st.number_input("Slippage (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    cash = st.number_input("Starting Cash", min_value=1000.0, value=10000.0, step=1000.0)
    commission = st.number_input("Commission", min_value=0.0, value=0.001, step=0.0005, format="%.4f")
    run = st.button("Run Backtest", width="stretch")

    st.divider()
    st.header("Optimizer")
    short_range = st.slider("Short MA Range", 5, 60, (10, 30), step=2)
    long_range = st.slider("Long MA Range", 50, 250, (80, 150), step=5)
    atr_stop_range_raw = st.slider(
        "ATR Stop Mult Range (x0.1)",
        0,
        50,
        (15, 30),
        step=1,
        key="atr_stop_range_raw_v2",
    )
    atr_stop_range = (atr_stop_range_raw[0] / 10, atr_stop_range_raw[1] / 10)
    max_combos = st.number_input("Max Combos", min_value=20, max_value=1000, value=200, step=20)
    wf_train_days = st.number_input("WFO Train Days", min_value=90, max_value=720, value=365, step=30)
    wf_test_days = st.number_input("WFO Test Days", min_value=60, max_value=360, value=180, step=30)
    run_opt = st.button("Run Optimizer", width="stretch")
    run_wfo = st.button("Run Walk-Forward Optimizer", width="stretch")

    st.divider()
    st.header("Watchlist")
    watchlist_raw = st.text_area("Symbols (comma-separated)", value="AAPL, MSFT, NVDA, TSLA")
    optimize_watchlist = st.checkbox("Optimize Each", value=False)
    run_watchlist = st.button("Run Watchlist", width="stretch")
    st.subheader("Portfolio")
    rebalance_days = st.number_input("Rebalance Days", min_value=5, max_value=120, value=21, step=1)
    costs_file = st.file_uploader(
        "Per-Asset Costs CSV (symbol,commission,slippage as decimal)",
        type=["csv"],
    )
    run_portfolio = st.button("Run Portfolio", width="stretch")


def _build_config() -> BacktestConfig:
    return BacktestConfig(
        symbol=symbol.strip().upper(),
        start=start,
        end=end,
        short_ma=int(short_ma),
        long_ma=int(long_ma),
        trend_filter=trend_filter,
        atr_period=int(atr_period),
        atr_stop_mult=float(atr_stop_mult),
        risk_per_trade=float(risk_per_trade),
        slippage_perc=float(slippage_perc) / 100.0,
        rsi_period=int(rsi_period),
        rsi_entry=float(rsi_entry),
        rsi_exit=float(rsi_exit),
        atr_vol_min=float(atr_vol_min),
        cash=float(cash),
        commission=float(commission),
    )


def _kpi(label: str, value: str, delta: str | None = None) -> None:
    st.markdown(
        f"""
<div class="kpi-card">
  <div style="color:#94a3b8;font-size:0.8rem">{label}</div>
  <div style="font-size:1.6rem;font-weight:700">{value}</div>
  <div style="color:#38bdf8;font-size:0.8rem">{delta or ''}</div>
</div>
""",
        unsafe_allow_html=True,
    )


col_left, col_right = st.columns([2, 1])
with col_right:
    st.subheader("What You Get")
    st.write("- Custom report with signals & equity")
    st.write("- Optimizer leaderboard")
    st.write("- Watchlist batch scan")
    st.write("- Benchmark vs buy & hold")

if run:
    try:
        config = _build_config()
        cerebro, strategy, df = run_backtest(config)
        metrics = get_metrics(strategy)
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
        equity = compute_equity_curve(
            df,
            signals,
            config.cash,
            commission_perc=config.commission,
            slippage_perc=config.slippage_perc,
        )
        benchmark = config.cash * (df["close"] / df["close"].iloc[0])
        bench_metrics = buy_hold_metrics(df, config.cash)

        st.subheader("Backtest Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _kpi("Sharpe", f"{metrics['sharpe']:.2f}", f"B&H {bench_metrics.sharpe:.2f}")
        with c2:
            _kpi("Max Drawdown (%)", f"{metrics['max_drawdown']:.2f}", f"B&H {bench_metrics.max_drawdown:.2f}")
        with c3:
            _kpi("Cumulative Return", f"{metrics['cumulative_return']:.2f}", f"B&H {bench_metrics.total_return:.2f}")
        with c4:
            _kpi("Total Trades", f"{metrics['total_trades']}", f"B&H {bench_metrics.trades}")

        st.subheader("Momentum Report")
        fig = build_report_figure(df, config)
        st.pyplot(fig, width="stretch")

        st.subheader("AI Research Signal")
        if st.button("Train AI Model", width="stretch"):
            ai_result = train_ai_model(df)
            st.write(f"Model: {ai_result.model_name}")
            st.write(f"Accuracy: {ai_result.accuracy:.2f}")
            st.write(f"Latest Up-Probability: {ai_result.latest_score:.2f}")
            st.caption(ai_result.message)

        st.subheader("Live Snapshot")
        if st.button("Fetch Live Snapshot", width="stretch"):
            try:
                live_df = yf.download(
                    config.symbol,
                    period="5d",
                    interval="5m",
                    progress=False,
                )
                live_df = live_df[["Open", "High", "Low", "Close", "Volume"]]
                live_df.columns = ["open", "high", "low", "close", "volume"]
                live_signals = generate_crossover_signals(
                    live_df,
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
                latest = live_signals.iloc[-1]
                action = "BUY" if latest == 1 else "SELL" if latest == -1 else "HOLD"
                st.write(f"Latest bar: {live_df.index[-1]}")
                st.write(f"Last close: {live_df['close'].iloc[-1]:.2f}")
                st.write(f"Signal: {action}")
            except Exception as exc:
                st.error(f"Live snapshot error: {exc}")

        st.subheader("Equity vs Buy & Hold")
        fig2 = build_report_figure(df, config)
        ax = fig2.axes[-1]
        ax.plot(benchmark.index, benchmark, color="#facc15", linewidth=1.6, label="Buy & Hold")
        ax.legend(loc="upper left", frameon=False)
        st.pyplot(fig2, width="stretch")

        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")
        st.download_button("Download Report (PNG)", png_buf.getvalue(), file_name="report.png")

        pdf_buf = io.BytesIO()
        fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
        st.download_button("Download Report (PDF)", pdf_buf.getvalue(), file_name="report.pdf")

        with st.expander("Data Preview"):
            st.dataframe(df.tail(50))
    except Exception as exc:
        st.error(f"Error: {exc}")

if run_opt:
    try:
        config = _build_config()
        df = fetch_yfinance_data(config.symbol, config.start, config.end)
        short_list = list(range(short_range[0], short_range[1] + 1, 2))
        long_list = list(range(long_range[0], long_range[1] + 1, 5))
        atr_stop_list = [
            round(atr_stop_range[0], 2),
            round((atr_stop_range[0] + atr_stop_range[1]) / 2, 2),
            round(atr_stop_range[1], 2),
        ]
        results = run_grid_search(
            df,
            config.cash,
            short_list,
            long_list,
            [True, False],
            [config.atr_period],
            atr_stop_list,
            max_combos=int(max_combos),
            commission_perc=config.commission,
            slippage_perc=config.slippage_perc,
            rsi_period_list=[config.rsi_period],
            rsi_entry_list=[config.rsi_entry],
            rsi_exit_list=[config.rsi_exit],
            atr_vol_min_list=[config.atr_vol_min],
        )
        st.subheader("Optimizer Leaderboard")
        st.dataframe(results.head(20), width="stretch")
    except Exception as exc:
        st.error(f"Optimizer error: {exc}")

if run_wfo:
    try:
        config = _build_config()
        short_list = list(range(short_range[0], short_range[1] + 1, 2))
        long_list = list(range(long_range[0], long_range[1] + 1, 5))
        atr_stop_list = [
            round(atr_stop_range[0], 2),
            round((atr_stop_range[0] + atr_stop_range[1]) / 2, 2),
            round(atr_stop_range[1], 2),
        ]
        wfo = walk_forward_optimize(
            config,
            train_days=int(wf_train_days),
            test_days=int(wf_test_days),
            short_list=short_list,
            long_list=long_list,
            atr_stop_list=atr_stop_list,
            trend_filter_list=[True, False],
            atr_period_list=[config.atr_period],
            max_combos=int(max_combos),
        )
        st.subheader("Walk-Forward Optimizer")
        st.dataframe(wfo, width="stretch")
    except Exception as exc:
        st.error(f"Walk-forward error: {exc}")

if run_watchlist:
    symbols = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]
    rows = []
    for sym in symbols:
        try:
            df = fetch_yfinance_data(sym, start, end)
            if optimize_watchlist:
                results = run_grid_search(
                    df,
                    cash,
                    list(range(10, 31, 2)),
                    list(range(80, 151, 5)),
                    [True, False],
                    [atr_period],
                    [atr_stop_mult],
                    max_combos=120,
                    commission_perc=float(commission),
                    slippage_perc=float(slippage_perc) / 100.0,
                    rsi_period_list=[int(rsi_period)],
                    rsi_entry_list=[float(rsi_entry)],
                    rsi_exit_list=[float(rsi_exit)],
                    atr_vol_min_list=[float(atr_vol_min)],
                )
                best = results.iloc[0].to_dict() if not results.empty else {}
                rows.append({"symbol": sym, **best})
            else:
                signals = generate_crossover_signals(
                    df,
                    short_ma=int(short_ma),
                    long_ma=int(long_ma),
                    trend_filter=trend_filter,
                    atr_period=int(atr_period),
                    atr_stop_mult=float(atr_stop_mult),
                    rsi_period=int(rsi_period),
                    rsi_entry=float(rsi_entry),
                    rsi_exit=float(rsi_exit),
                    atr_vol_min=float(atr_vol_min),
                )
                metrics = compute_metrics(
                    df,
                    signals,
                    cash,
                    commission_perc=float(commission),
                    slippage_perc=float(slippage_perc) / 100.0,
                )
                rows.append(
                    {
                        "symbol": sym,
                        "total_return": metrics.total_return,
                        "cagr": metrics.cagr,
                        "sharpe": metrics.sharpe,
                        "max_drawdown": metrics.max_drawdown,
                        "trades": metrics.trades,
                    }
                )
        except Exception as exc:
            rows.append({"symbol": sym, "error": str(exc)})

    st.subheader("Watchlist Results")
    st.dataframe(rows, width="stretch")

if run_portfolio:
    symbols = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]
    data_map = {sym: fetch_yfinance_data(sym, start, end) for sym in symbols}
    slippage_rate = float(slippage_perc) / 100.0
    asset_costs = None
    if costs_file is not None:
        try:
            costs_df = pd.read_csv(costs_file)
            asset_costs = {
                row["symbol"].strip().upper(): {
                    "commission": float(row.get("commission", commission)),
                    "slippage": float(row.get("slippage", slippage_rate)),
                }
                for _, row in costs_df.iterrows()
            }
        except Exception as exc:
            st.error(f"Cost file error: {exc}")
            asset_costs = None

    try:
        config = _build_config()
        equity = simulate_portfolio(
            data_map,
            config,
            rebalance_days=int(rebalance_days),
            asset_costs=asset_costs,
        )
        metrics = portfolio_metrics(equity)
        st.subheader("Portfolio Equity")
        st.line_chart(equity)
        c1, c2, c3 = st.columns(3)
        with c1:
            _kpi("Portfolio Sharpe", f"{metrics.sharpe:.2f}")
        with c2:
            _kpi("Portfolio Max DD (%)", f"{metrics.max_drawdown:.2f}")
        with c3:
            _kpi("Portfolio CAGR", f"{metrics.cagr:.2%}")
    except Exception as exc:
        st.error(f"Portfolio error: {exc}")

if not (run or run_opt or run_wfo or run_watchlist or run_portfolio):
    st.info("Set parameters and run a backtest, optimizer, or watchlist scan.")
