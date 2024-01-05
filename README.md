# Reinforcement Learning Stock Trader

The RL trader aims to maximize the profit by doing a daily rebalancing of the portfolio

- The model trains on the first half of the stock prices and tests on the second half.
- The model predicts what action to take: buy/sell/hold based on historical data.
- An epsilon-greedy policy is used to allow for exploration.

Prediction Model:

- Linear Regression using Gradient Descent with Momentum


## Results
This is an example of how the trader acts on one of the stocks (JPM).\
Green points is when the trader buys, while red points are when the trader sells.

![buy_sell](https://github.com/alexisdpc/RL-Stock-Trader/assets/124795834/c7b14119-690e-4576-bb61-c52dc21c159e)


We also compare the returns on the 5 years of test data with two other portfolios:

- Portfolio with equally-weighted stocks: 30.14 %

- Portfolio with random actions (epsilon=1): 23.57 %

- Portfolio with Reinforcement Learning agent: 188.53 %

