import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
data = pd.read_csv("BTCUSDT_15minutes.csv")
data['Time'] = pd.to_datetime(data['Time'])
data.set_index('Time', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# ATR Calculation
def calculate_atr(df, period=14):
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)

# MACD Calculation
def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal

# Strategy Class
class TradingStrategy:
    def __init__(self, data, initial_capital=1000000):
        self.data = data
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size = 0
        self.position = None
        self.stop_loss = None
        self.trades = []
        self.max_drawdown = 0
        self.risk_percent = 0.01
        self.drawdown = [] 

    def atr_based_position_sizing(self, price, atr):
        # Calculate position size based on ATR stop-loss
        stop_loss_distance = atr * price  # Stop loss distance in dollars
        if stop_loss_distance / price > 0.01:
            return self.capital / stop_loss_distance
        else:
            return self.capital

    def adjust_stop_loss(self, open_price, close_price, atr, price):
        return (open_price + close_price) / (atr / price)

    def run(self, atr_multiplier=1.5):
        for i in range(1, len(self.data)):
            row = self.data.iloc[i]
            price = row['Close']
            atr = row['ATR']
            macd = row['MACD']
            macd_signal = row['MACD_Signal']

            equity = self.capital + (self.position['size'] * (price - self.position['entry_price']) if self.position else 0)
            drawdown = (self.initial_capital - equity) / self.initial_capital
            self.drawdown.append(drawdown)

            # Long Entry
            if self.position is None and macd > macd_signal:
                position_size = self.atr_based_position_sizing(price, atr)
                self.stop_loss = price - atr * atr_multiplier
                self.position = {'entry_price': price, 'size': position_size}
                self.capital -= price * position_size  # Deduct capital for buying

            # Short Entry
            elif self.position is None and macd < macd_signal:
                position_size = self.atr_based_position_sizing(price, atr)
                self.stop_loss = price + atr * atr_multiplier
                self.position = {'entry_price': price, 'size': position_size, 'short': True}
                self.capital += price * position_size  # Add capital for shorting

            # Manage existing position
            if self.position is not None:
                if not self.position.get('short', False):  # Long position management
                    if price > self.position['entry_price']:
                        self.stop_loss = self.adjust_stop_loss(row['Open'], row['Close'], atr, price)
                    if price <= self.stop_loss or macd < macd_signal:
                        profit = (price - self.position['entry_price']) * self.position['size']
                        self.capital += price * self.position['size']  # Add profit/loss to capital
                        self.trades.append(profit)
                        self.position = None
                else:  # Short position management
                    if price < self.position['entry_price']:
                        self.stop_loss = self.adjust_stop_loss(row['Open'], row['Close'], atr, price)
                    if price >= self.stop_loss or macd > macd_signal:
                        profit = (self.position['entry_price'] - price) * self.position['size']
                        self.capital -= price * self.position['size']  # Add profit/loss to capital
                        self.trades.append(profit)
                        self.position = None

            # Max drawdown
            equity = self.capital + (self.position['size'] * (price - self.position['entry_price']) if self.position else 0)
            self.max_drawdown = min(self.max_drawdown, self.initial_capital - equity)

    def report(self):
        total_trades = len(self.trades)
        winning_trades = [trade for trade in self.trades if trade > 0]
        losing_trades = [trade for trade in self.trades if trade < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        total_profit = np.sum(self.trades)

        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Total Profit: {total_profit:.2f}")
        print(f"Final Capital: {self.capital:.2f}")
        print(f"Max Drawdown: {self.max_drawdown:.2f}")

# Function to optimize individual parameters
def optimize_parameters(params, data):
    try:
        macd_fast, macd_slow, macd_signal, atr_period = params
        calculate_macd(data, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
        calculate_atr(data, period=atr_period)

        # Run the strategy
        strategy = TradingStrategy(data)
        strategy.run()
        total_profit = np.sum(strategy.trades)

        return (total_profit, params)
    except Exception as e:
        logging.error(f"Error with parameters {params}: {e}")  # Log the error message
        return (0, params)  # Return a default value to indicate failure

# Optimization using grid search
def grid_search(data, initial_capital=1000000):
    best_result = None
    best_params = {}

    # Parameter ranges for MACD and ATR
    macd_fast_range = range(8, 10)
    macd_slow_range = range(21, 24)
    macd_signal_range = range(5, 7)
    atr_period_range = range(13, 16)

    param_combinations = [(macd_fast, macd_slow, macd_signal, atr_period)
                          for macd_fast in macd_fast_range
                          for macd_slow in macd_slow_range
                          for macd_signal in macd_signal_range
                          for atr_period in atr_period_range]

    start_time = time.time()

    with ThreadPoolExecutor() as executor:  # Use threads instead of processes
        futures = {executor.submit(optimize_parameters, params, data): params for params in param_combinations}

        for future in as_completed(futures):
            result, params = future.result()
            elapsed_time = time.time() - start_time
            
            if best_result is None or result > best_result:
                best_result = result
                best_params = params

            logging.info(f"Processed params: {params}, Profit: {result:.2f}, Time elapsed: {elapsed_time:.2f}s")

    logging.info(f"Best Result: {best_result}")
    logging.info(f"Best Params: {best_params}")
    return best_params

# Run optimization
best_params = grid_search(data)

# Run the final strategy with the best parameters
best_macd_fast, best_macd_slow, best_macd_signal, best_atr_period = best_params
calculate_macd(data, fast_period=best_macd_fast, slow_period=best_macd_slow, signal_period=best_macd_signal)
calculate_atr(data, period=best_atr_period)
strategy = TradingStrategy(data)
strategy.run()

# Report the results
strategy.report()

# Plot equity curve
equity_curve = [strategy.initial_capital + np.cumsum(strategy.trades)]
plt.figure(figsize=(14, 7))
plt.plot(equity_curve)
plt.title('Equity Curve')
plt.xlabel('Trades')
plt.ylabel('Equity')
plt.grid()
plt.show()

# Plot drawdown
plt.figure(figsize=(14, 7))
plt.plot(strategy.drawdown, label='Drawdown', color='orange')
plt.title('Drawdown Over Time')
plt.xlabel('Trades')
plt.ylabel('Drawdown')
plt.axhline(0, color='black', lw=1, ls='--')
plt.legend()
plt.grid()
plt.show()

# Plot indicator signals
plt.figure(figsize=(14, 7))
plt.subplot(3, 1, 1)
plt.plot(data.index, data['Close'], label='Price', color='blue')
plt.title('Price Chart')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data.index, data['MACD'], label='MACD', color='green')
plt.plot(data.index, data['MACD_Signal'], label='MACD Signal', color='red')
plt.title('MACD Indicator')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(data.index, data['ATR'], label='ATR', color='purple')
plt.title('Average True Range (ATR)')
plt.legend()

plt.tight_layout()
plt.show()