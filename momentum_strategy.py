import backtrader as bt
import yfinance as yf
import pandas as pd

class MomentumStrategy(bt.Strategy):
    params = (("short_ma", 50), ("long_ma", 200))

    def __init__(self):
        # Define the short (50-day) and long (200-day) moving averages
        self.sma_short = bt.indicators.SimpleMovingAverage(period=self.params.short_ma)
        self.sma_long = bt.indicators.SimpleMovingAverage(period=self.params.long_ma)

    def next(self):
        if not self.position:  # No open position
            if self.sma_short[0] > self.sma_long[0]:  # Golden Cross
                self.buy()
        elif self.sma_short[0] < self.sma_long[0]:  # Death Cross (Exit Signal)
            self.sell()

# Initialize Backtest
cerebro = bt.Cerebro()

# Fetch historical data from Yahoo Finance
symbol = "AAPL"  # Change this to "BTC-USD" for Bitcoin
df = yf.download(symbol, start="2022-01-01", end="2023-01-01", progress=False)

# Ensure DataFrame is valid and formatted for Backtrader
if df.empty:
    raise ValueError("Error: No data retrieved from Yahoo Finance. Check symbol and date range.")

df.index = pd.to_datetime(df.index)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.columns = ['open', 'high', 'low', 'close', 'volume']

# Load Data into Backtrader
data = bt.feeds.PandasData(dataname=df)

# Add data and strategy to backtest engine
cerebro.adddata(data)
cerebro.addstrategy(MomentumStrategy)

# Run Backtest and Plot Results
cerebro.run()
cerebro.plot()
