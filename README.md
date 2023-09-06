## Reinforcement Learning Stock Trader

Notebook summarizing the results after running the RL trader.\
The RL trader aims to maximize the profit by doing a daily rebalancing of the portfolio.

- The model trains on the first half of the stock prices and tests on the second half.
- The model predicts what action to take: buy/sell/hold based on historical data.
- An epsilon-greedy policy is used to allow for exploration.

Prediction Model:

- Linear Regression using Gradient Descent with Momentum

