# main.py

from environment import StockMarketEnv
from model import QLearningAgent
import matplotlib.pyplot as plt

# Initialize the environment and agent
env = StockMarketEnv(stock_symbol='AAPL', start_date='2015-01-01', end_date='2023-01-01')
agent = QLearningAgent(state_size=len(env.stock_data), action_size=3)

# Train the agent
agent.train(env, episodes=1000)

# Simulate the trading
portfolio_values = []
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state[0])
    next_state, reward, done = env.step(action)
    env.render()
    portfolio_value = state[1] + (state[2] * state[0])  # Cash + (stock_owned * price)
    portfolio_values.append(portfolio_value)
    state = next_state

# Plot portfolio value over time
plt.plot(env.stock_data['Date'], portfolio_values)
plt.title('Trading Agent Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.show()