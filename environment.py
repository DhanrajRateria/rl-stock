# environment.py

import yfinance as yf
import numpy as np

class StockMarketEnv:
    def __init__(self, stock_symbol, start_date, end_date, initial_cash=10000):
        self.stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        self.stock_data.reset_index(inplace=True)
        self.initial_cash = initial_cash
        self.current_step = 0
        self.current_cash = initial_cash
        self.stock_owned = 0
        self.done = False

    def reset(self):
        """Resets the environment to the initial state."""
        self.current_step = 0
        self.current_cash = self.initial_cash
        self.stock_owned = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Returns the current state of the environment (price, portfolio)."""
        price = self.stock_data['Close'].iloc[self.current_step]
        return [price, self.current_cash, self.stock_owned]

    def step(self, action):
        """Executes the action: Buy (0), Sell (1), or Hold (2)."""
        current_price = self.stock_data['Close'].iloc[self.current_step]
        reward = 0

        if action == 0:  # Buy
            if self.current_cash >= current_price:
                self.stock_owned += 1
                self.current_cash -= current_price
        elif action == 1:  # Sell
            if self.stock_owned > 0:
                self.stock_owned -= 1
                self.current_cash += current_price
                reward = current_price  # Profit from selling
        else:  # Hold
            reward = 0

        self.current_step += 1
        if self.current_step >= len(self.stock_data) - 1:
            self.done = True

        next_state = self._get_state()
        return next_state, reward, self.done

    def render(self):
        """Displays the portfolio value and stock price at the current step."""
        price = self.stock_data['Close'].iloc[self.current_step]
        total_value = self.current_cash + (self.stock_owned * price)
        print(f"Step: {self.current_step}, Stock Price: {price}, Cash: {self.current_cash}, "
              f"Stocks Owned: {self.stock_owned}, Total Portfolio Value: {total_value}")