# Momentum Strategy Lab

Production-grade momentum research platform built on Backtrader + yfinance. This repo provides a clean CLI, a premium Streamlit dashboard, walk-forward validation, portfolio simulation, and optional AI scoring to help you evaluate momentum strategies with realistic costs and risk controls.

## Highlights
- Modular architecture (data, strategy, evaluation, reporting, CLI, dashboard)
- Risk controls: trend filter, ATR trailing stop, RSI confirmation, volatility filter
- Execution realism: commission + slippage, ATR-based position sizing
- Walk-forward analysis and optimizer (train/test windows)
- Portfolio simulator with per-asset cost models
- Premium report charts + export to PNG/PDF
- Optional AI scoring (Logistic Regression for research signals)

## Project Structure
```
Momentum_Following_Strategy/
  momentum_strategy/
    __init__.py
    __main__.py
    ai.py
    backtest.py
    cli.py
    config.py
    data.py
    evaluation.py
    logging_utils.py
    realtime.py
    report.py
    signals.py
    strategy.py
  tests/
    test_signals.py
  app.py
  run_backtest.py
  requirements.txt
  pyproject.toml
  README.md
```

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m momentum_strategy --symbol AAPL --start 2020-01-01 --end 2023-01-01 --plot
```

## CLI Usage
```bash
python -m momentum_strategy --help
```

### Core Backtest
```bash
python -m momentum_strategy --symbol AAPL --start 2020-01-01 --end 2023-01-01 \
  --short-ma 20 --long-ma 100 --rsi-entry 55 --rsi-exit 45 --atr-vol-min 0.01 --plot
```

### Walk-Forward (Out-of-Sample)
```bash
python -m momentum_strategy --symbol AAPL --start 2018-01-01 --end 2023-12-31 \
  --short-ma 20 --long-ma 100 --walk-forward --wf-train-days 365 --wf-test-days 180
```

### Walk-Forward Optimizer
```bash
python -m momentum_strategy --symbol AAPL --start 2018-01-01 --end 2023-12-31 \
  --walk-forward-opt --opt-short-min 10 --opt-short-max 30 --opt-long-min 80 --opt-long-max 150
```

### Portfolio Mode (CLI)
```bash
python -m momentum_strategy --portfolio --symbols AAPL,MSFT,NVDA,TSLA \
  --start 2020-01-01 --end 2023-01-01 --rebalance-days 21
```

Per-asset costs CSV (decimal rates):
```csv
symbol,commission,slippage
AAPL,0.001,0.0005
MSFT,0.0008,0.0004
```

### Live Signals (Paper)
```bash
python -m momentum_strategy --symbol AAPL --start 2023-01-01 --end 2023-12-31 \
  --live --live-interval 5m --live-refresh-sec 60
```

## Dashboard (Streamlit)
```bash
streamlit run app.py
```

Dashboard features:
- KPI cards and equity benchmark vs buy-and-hold
- Optimizer leaderboard and walk-forward optimizer
- Multi-symbol watchlist scan and portfolio mode
- AI scoring and live snapshot
- Export report as PNG/PDF

## Configuration Reference
Key parameters (CLI or dashboard):
- `short_ma`, `long_ma`: SMA crossover windows
- `trend_filter`: only trade when price > long SMA
- `atr_period`, `atr_stop_mult`: trailing stop and volatility measure
- `rsi_period`, `rsi_entry`, `rsi_exit`: momentum confirmation + exit
- `atr_vol_min`: minimum ATR/price to avoid low-volatility chop
- `risk_per_trade`: risk-based position sizing
- `commission`, `slippage_perc`: execution costs

## AI Scoring (Optional)
The AI module uses a simple Logistic Regression model trained on engineered features (returns, SMA slope, RSI, ATR%) to output a probability of a positive next-day return. It is intended for research only, not as a trading signal.

To enable:
```bash
pip install scikit-learn
```
Then use the dashboard button: **Train AI Model**.

## Testing
```bash
pytest
```

## Security Notes
- No credentials are stored or logged.
- If you add broker APIs, load secrets via environment variables.
- Treat outputs as research only; validate with out-of-sample testing.

## Disclaimer
This project is for educational and research purposes only and is not financial advice.
