# 📈 Momentum/Trend-Following Strategy (Backtrader)

This project implements a **Momentum Trading Strategy** using **Backtrader** and **Yahoo Finance (`yfinance`)** for backtesting. It follows the **Golden Cross/Death Cross principle**, where:
- **Buy Signal:** When the 50-day SMA crosses above the 200-day SMA.
- **Sell Signal:** When the 50-day SMA crosses below the 200-day SMA.

## 🔥 Features
✅ **Backtesting with historical stock & crypto data**  
✅ **Visual trade signals (Buy/Sell) on price charts**  
✅ **Works with stocks, ETFs, and cryptocurrencies**  
✅ **Easy parameter customization for optimization**  
✅ **Extensible for live trading with broker APIs**  

---

## 🚀 How It Works
- **Indicators Used:** 
  - `Simple Moving Average (SMA)`
  - `Golden Cross` (Bullish)
  - `Death Cross` (Bearish)

- **Trading Logic:**
  - Enter a trade when a bullish crossover happens.
  - Exit when a bearish crossover occurs.

---

## 🛠 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/momentum_strategy.git
   cd momentum_strategy
