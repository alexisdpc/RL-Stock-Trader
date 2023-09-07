## Reinforcement Learning Stock Trader

Notebook summarizing the results after running the RL trader.\
The RL trader aims to maximize the profit by doing a daily rebalancing of the portfolio.

- The model trains on the first half of the stock prices and tests on the second half.
- The model predicts what action to take: buy/sell/hold based on historical data.
- An epsilon-greedy policy is used to allow for exploration.

Prediction Model:

- Linear Regression using Gradient Descent with Momentum



This is an example on one of the stocks (JPM). Green points is when the trader buys, while red points are when the trader sells.

![buy_sell](https://github.com/alexisdpc/RL-Stock-Trader/assets/124795834/c7b14119-690e-4576-bb61-c52dc21c159e)
