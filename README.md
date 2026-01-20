# Momentum / Trend-Following Strategy

Industry-style backtesting project using Backtrader + yfinance with a clean CLI, structured modules, tests, and a custom report chart.

## Highlights
- Modular design (data, strategy, backtest, CLI, report)
- Strict input validation and logging
- Testable signal generation
- Custom recruiter-friendly report (price, SMA, signals, equity)
- Risk controls: trend filter + ATR trailing stop
- Execution realism: slippage + risk-based position sizing
- Security-minded defaults (no secrets in code, no external writes)

## Project Structure
```
Momentum_Following_Strategy/
  momentum_strategy/
    __init__.py
    __main__.py
    backtest.py
    cli.py
    config.py
    data.py
    logging_utils.py
    signals.py
    strategy.py
  tests/
    test_signals.py
  run_backtest.py
  README.md
```

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m momentum_strategy --symbol AAPL --start 2022-01-01 --end 2023-01-01 --plot
python run_backtest.py --symbol AAPL --start 2022-01-01 --end 2023-01-01 --plot
```

## CLI Options
```bash
python -m momentum_strategy --help
```

## Security Notes
- No credentials are stored or logged.
- Input validation blocks invalid periods or dates.
- If you add broker APIs later, load secrets via environment variables (never hardcode them).

## Development
```bash
pytest
```

## Disclaimer
This project is for educational and research purposes only and is not financial advice.

## Dashboard (Optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard features:
- KPI cards and equity benchmark vs buy-and-hold
- Optimizer leaderboard for SMA/ATR settings
- Multi-symbol watchlist scan
- Export report as PNG/PDF

## Walk-Forward (Out-of-Sample)
```bash
python -m momentum_strategy --symbol AAPL --start 2018-01-01 --end 2023-12-31 \\
  --short-ma 20 --long-ma 100 --walk-forward --wf-train-days 365 --wf-test-days 180
```
