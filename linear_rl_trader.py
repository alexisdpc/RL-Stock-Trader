# ========================== Reinforcement Learning Trader ====================================
#
# We implement a Reinforcement Learning algorithm for stock trading.
# The aim is to train an Agent that will buy and sell different stocks to maximize the profit.
# Model: Linear Regression with gradient descent and momentum. 
#
# =============================================================================================

import numpy as np
import pandas as pd
import itertools
import argparse
import os
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import StandardScaler

'''We read the data and create a dataframe that contains the stock prices stock_prices.csv:
January 1, 2010 - December 31, 2019
Chevron (CVX), Ford (FORD), Google (GOOG), JP MOrgan (JPM), Microsoft (MSFT), Walmart (WMT)'''

def get_data():
  '''Returns: Array with the prices of the selected stocks.
              Each column corresponds to a different stock.'''
  df = pd.read_csv('data/stock_prices.csv') 
  return df.values

def get_scaler(env):
  '''Returns: scaler object to scale the states
  StandardScaler: Standardize features by removing the mean and scaling to unit variance.

  In order to get the right parameters for the scaler we need some data
  To get the data, we play an episode randomly and store each of the states we encounter
  There is no need to have an agent, because such agent will not be trained anyway
  To choose an action, we choose randomly from the action_space
  This can be improved if we run over multiple episodes'''

  states = []
  for _ in range(env.n_step):
    action = np.random.choice(env.action_space)
    state, reward, done, port_val = env.step(action) 
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler  

def maybe_make_dir(directory):
  '''If directory does not exist, then create it'''
  if not os.path.exists(directory):
    os.makedirs(directory)    

class LinearModel: 
  "Linear regression model"
  def __init__(self, input_dim, n_action):
    # input_dim = 2*n_stocks + 1 
    # n_action = 3^n_stocks
    # W is a matrix of size [input_dim, n_action] 
    self.W = np.random.randn(input_dim, n_action)/np.sqrt(input_dim)
    self.b = np.zeros(n_action)  

    self.vW = 0
    self.vb = 0

    self.losses = []

  def predict(self, X):
    '''Check that the vector is 2D
    Argument: X is size (1,7)
    Return: Y is size (1,27) Value of the portfolio for each of the 27 actions'''
    assert(len(X.shape) == 2)  # Remove this line and all the asserts?
    return X.dot(self.W) + self.b

  def sgd(self, X, Y, learning_rate = 0.01, momentum = 0.9):
    '''Implementation of stochastic?? gradient descent with Momentum
    First calculate the momentum term. 
    Next update the parameters.'''
    assert(len(X.shape) == 2)      

    # The total number of values n,m where Y is an n,m matrix
    num_values = np.prod(Y.shape)

    # Y: Value of the portfolio for each of the 27 actions
    # Y: Prediction from Model, but for entry 'action' it has reward+gamma*prediction
    # Yhat: Prediction from Model
    # (Yhat-Y) has only 1 non-zero entry

    Yhat = self.predict(X)
    # Derivative of MSE function with resepct to W, gW is a 7,27 matrix
    gW = 2.*X.T.dot(Yhat-Y)/num_values
    # Derivative of MSE function with resepct to b, gb has shape 27,1 
    gb = 2.*(Yhat-Y).sum(axis=0)/num_values

    # The value of 'momentum' determines the contributions from previous gradients
    # We update the momentum terms:
    self.vW = momentum*self.vW - learning_rate*gW
    self.vb = momentum*self.vb - learning_rate*gb

    # We update the parameters:
    self.W += self.vW
    self.b += self.vb
    
    # Mean squared error, which is just a number
    mse = np.mean((Yhat-Y)**2.)
    self.losses.append(mse)

  def load_weights(self, filepath):
    '''Load the weights of the trained model'''
    npz = np.load(filepath)
    self.W = npz['W']
    self.b = npz['b']

  def save_weights(self, filepath):
    '''Save arrays into filepath.npz'''
    np.savez(filepath, W=self.W, b=self.b)    

class MultiStockEnv:
  '''Returns: state, reward, done, info '''

  def __init__(self, data, initial_investment=2000):         
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    self.initial_investment = initial_investment 
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    
    # Create the action_space. We have 3^n_stock possibilities
    self.action_space = np.arange(3**self.n_stock)
    self.action_list = list(map(list, itertools.product([0,1,2], repeat=self.n_stock)))
    self.state_dim = 2*self.n_stock + 1
    self.reset()

  def reset(self):
    ''' Goes back to the initial state
    Returns: Observation state'''
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.cur_step]
    self.cash_in_hand = self.initial_investment   

    return self._get_obs()     

  def step(self, action):
    '''This function performs the action and updates the state
       Argument: action 
       Return: state, reward, done, current_value'''

    # Ensure that the action is part of the action_space    
    assert action in self.action_space

    # Obtain the current value before performing the action
    prev_val = self._get_val() 

    self.cur_step += 1
    self.stock_price = self.stock_price_history[self.cur_step]
    self._trade(action) 

    # TODO: Check this one out: print(f'\n {action} \n')

    # Obtain the current value
    cur_val = self._get_val()

    # Reward (Gain in portfolio value)
    reward = cur_val - prev_val

    # Check if we have reached the end of the time series
    done = (self.cur_step == self.n_step-1)

    # Save the current value of the portfolio
    port_val = {'cur_val': cur_val}

    return self._get_obs(), reward, done, port_val

  # ======= Functions used internally: ========
  # Get observation, this is equivalent to getting the 'state'
  def _get_obs(self):
    '''Returns a vector with 7 entries, for example:
    [10,15,20,150,100,200,10000]
    - Therefore, we own 10,15,20 share of stock 1,2,3 respectively
    - Price of the stocks is 150,100,200
    - We have 10000 cash in hand'''

    obs = np.empty(self.state_dim)  # Why not use np.zeros(self.state_dim)?
    # First n_stock entries contain the number of stocks owned:
    obs[:self.n_stock] = self.stock_owned  
    # The next n_stock entries contain the stock prices:
    obs[self.n_stock:2*self.n_stock] = self.stock_price 
    # The last entry contains the cash in hand:
    obs[-1] = self.cash_in_hand 
    return obs

  def _get_val(self):    
    '''Returns the current value of the Portfolio = stocks*stockprice + cash '''
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

  def _trade(self, action):   
    '''Argument: action (integer with the index between 0-26)
       Return: vector of size n_stocks with the actions for each stock
    0: sell 
    1: hold
    2: buy'''

    # Obtain the action vector of size n_stocks
    action_vec = self.action_list[action]
    
    if ep == num_episodes-1:
      for i in range(n_stocks):
        actions_list[i].append(action_vec[i])

    # We sell before buying
    sell_index = []
    buy_index = []
    for i, a in enumerate(action_vec):
      # Sell stock:
      if a == 0:
        sell_index.append(i)
      # Buy stock:
      elif a == 2:
        buy_index.append(i)  
  
    # Sell all the shares in the stock we want to sell
    if sell_index:
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i]*self.stock_owned[i] 
        self.stock_owned[i] = 0 
    
    # Buy shaers (one-by-one) for each stock, until there is no more cash in hand
    if buy_index:
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False      
  # ================================================

class DQNAgent(object):
  ''' Trading Agent '''
  
  def __init__(self, state_size, action_size):
    ''' Constructor
    state_size: input of NN
    action_size output of NN '''
    self.state_size = state_size
    self.action_size = action_size
    self.gamma = 0.95   # Discount rate
    self.epsilon = 1.0  # Exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = LinearModel(state_size, action_size)  # Call the LinearModel constructor

  def act(self, state):
    ''' Epsilon greedy function to choose action.
        Returns: Index of the action that needs to be taken, integer in range [0, 3^n_stocks]'''

    # epsilon-greedy policy:
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    
    # 'state' is an 2*n_stock+1 vector
    # 'act_values' is an n_stock^3 vector with the portfolio value for all possible actions
    act_values = self.model.predict(state)  

    # ====================
    #print(f'\n state: {state} \n\n')
    #print(f'\n act_values: {act_values} \n\n')
    #print(act_values[0])
    # ====================

    # Choose the action that maximizes the value. Recturns the index of action.
    max_action = np.argmax(act_values[0])
    return max_action
  
  def train(self, state, action, reward, next_state, done):
    ''' Training function. '''

    # WE ONLY COMPUTE TARGET FOR STATE PREDICTED
    # COUNT NON_ZERO MATRIX ENTRIES
    if done:
      target = reward 
    else:   
      # Update value function maximizing over predictions (Bellmann Equation?) (one step into the future)
      target = reward + self.gamma*np.amax(self.model.predict(next_state), axis=1)
    
    # Construct the Value function for all possible states
    target_full = self.model.predict(state)
    target_full[0, action] = target

    # Run gradient descent and update the weights
    # X = state, Y = target_full
    self.model.sgd(state, target_full)
    
    # Decrease the value of epsilon, so that there is less exploration with time
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)

def play_one_episode(agent, env, is_train):
  '''Returns: current value of the portfolio '''

  # Reset the environment and transform the state
  state = env.reset()
  state = scaler.transform([state])
  done = False   

  while not done:
    action = agent.act(state)
    next_state, reward, done, port_val = env.step(action)
    next_state = scaler.transform([next_state])

    # Only in 'train' mode we make the prediction, compute the loss and update the weights
    if is_train == 'train':
      agent.train(state, action, reward, next_state, done)
    state = next_state

  return port_val['cur_val']   


#ep = 0
if __name__ == '__main__':

  # Set-up and onfiguration 
  ep = 0
  models_folder = 'linear_rl_trader_models'
  rewards_folder = 'linear_rl_trader_rewards'
  num_episodes = 200 #10 #2000
  batch_size = 32  # I NEVER USE THIS VARIABLE!
  initial_investment = 20000

  # Run with command arguments from the terminal -m train or -m test
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
  args = parser.parse_args()

  # Create the directories in case they do not exist 
  maybe_make_dir(models_folder)
  maybe_make_dir(rewards_folder)

  # Get the time series with the stock prices
  data = get_data()
  n_timesteps, n_stocks = data.shape 
  n_train = n_timesteps//2
  actions_list = [[] for x in range(n_stocks)]
  
  # We train on the first half of the data and test on the other half
  train_data = data[:n_train]
  test_data = data[n_train:]
  
  # Initialize the environment with the Train data and variables
  env = MultiStockEnv(train_data, initial_investment)
  state_size = env.state_dim
  action_size = len(env.action_space)
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)
  
  # Initialize variable for portfolio value
  portfolio_value = []
  
  # If 'test' mode then load the saved scaler and weights before running the episodes
  if args.mode == 'test':
    # Load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)
    
    # Remake the environment with the 'test' data
    env = MultiStockEnv(test_data, initial_investment)

    # Initial epsilon (determines amount of exploration)
    # epislon=0 in 'test' mode then we always gives the same results
    # epsilon=1 always takes random actions
    agent.epsilon = 0.01

    # Load the weights
    agent.load(f'{models_folder}/linear.npz')

  # We run all the episodes
  for ep in range(num_episodes):
    t0 = datetime.now()   
    val = play_one_episode(agent, env, args.mode)
    dt = datetime.now() - t0
    print(f"Episode: {ep+1}/{num_episodes},  Episode end value: {val:.2f},  Duration: {dt} ")
    portfolio_value.append(val) 

  # If 'train' mode we save weights and the scaler after the training
  if args.mode == 'train':
    agent.save(f'{models_folder}/linear.npz')
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)

    # Plot losses
    plt.plot(agent.model.losses)
    plt.title('Losses')
    plt.show()  

  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)  

  # We only save the actions of the last episode
  actions = pd.DataFrame()
  for i in range(n_stocks):
    actions['Actions'+str(i)] = actions_list[i]
  actions.to_csv(f'{rewards_folder}/actions.csv')