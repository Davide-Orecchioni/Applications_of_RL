import numpy as np
import random
import pandas as pd

import contextlib
from concurrent.futures import ThreadPoolExecutor
from numba import njit, types
from numba.experimental import jitclass
from multiprocessing import get_context


# MDP implementations for the standard models (looping model
# with episodic (Loop) or incremental (Delta) reward). Actions
# are bit flips and can be restricted or unrestricted

@jitclass([("n", types.int64), ("actions", types.int64),
           ("f", types.FunctionType(types.float64(types.float64[::1]))),
           ("state", types.Tuple((types.float64[::1], types.int64)))])
class Loop:
  def __init__(self, n, restricted, f, start):
        self.n = n
        self.actions = 2 if restricted else n+1
        self.f = f
        self.state = start

  def reset(self):
        x = np.zeros(self.n)
        for i in range(self.n):
          x[i] = random.randint(0,1)
        self.state = (x, 1)
        return (x, 1)

  def step(self, action):
      x = oplus(self.state, action, self.actions)
      t = self.state[1]
      # Time passing, Episodic reward
      if (t < self.n):
        t = t+1
        r=0
      else:
        t = 1
        r = self.f(x)
      # State Transition
      next_state = (x, t)
      self.state = next_state
      return next_state, r


@jitclass([("n", types.int64), ("actions", types.int64),
           ("f", types.FunctionType(types.float64(types.float64[::1]))),
           ("state", types.Tuple((types.float64[::1], types.int64)))])
class Delta:
  def __init__(self, n, restricted, f, start):
        self.n = n
        self.actions = 2 if restricted else n+1
        self.f = f
        self.state = start

  def reset(self):
        x = np.zeros(self.n)
        for i in range(self.n):
          x[i] = random.randint(0,1)
        self.state = (x, 1)
        return (x, 1)

  def step(self, action):
      # Time step
      t = self.state[1]
      if (t < self.n):
        t = t+1
      else:
        t = 1
      # Incremental Reward
      x_old = self.state[0].copy()
      x_new = oplus(self.state, action, self.actions)
      r = self.f(x_new) - self.f(x_old)
      # State Transition
      next_state = (x_new, t)
      self.state = next_state
      return next_state, r



# MDP implementations for the monotonic models, still looping,
# with episodic (Loop) or incremental (Delta) reward). Actions
# set bits to 1 and can be restricted or unrestricted

@jitclass([("n", types.int64), ("actions", types.int64),
           ("f", types.FunctionType(types.float64(types.float64[::1]))),
           ("state", types.Tuple((types.float64[::1], types.int64)))])
class Mono_Episodic:
  def __init__(self, n, restricted, f, start):
        self.n = n
        self.actions = 2 if restricted else n+1
        self.f = f
        self.state = start

  def reset(self):
        x = np.zeros(self.n)
        self.state = (x, 1)
        return (x, 1)

  def step(self, action):
      t = self.state[1]
      x = monoplus(self.state, action, self.actions)

      if (t < self.n):
        t = t+1
        r = 0
      else:
        t = 1
        r = self.f(x)
      
      next_state = (x, t)
      self.state = next_state
      return next_state, r


@jitclass([("n", types.int64), ("actions", types.int64),
           ("f", types.FunctionType(types.float64(types.float64[::1]))),
           ("state", types.Tuple((types.float64[::1], types.int64)))])
class Mono_Incremental:
  def __init__(self, n, restricted, f, start):
        self.n = n
        self.actions = 2 if restricted else n+1
        self.f = f
        self.state = start

  def reset(self):
        x = np.zeros(self.n)
        self.state = (x, 1)
        return (x, 1)

  def step(self, action):
      # Incremental reward
      x_old = self.state[0].copy()
      x_new = monoplus(self.state, action, self.actions)
      r = self.f(x_new) - self.f(x_old)
      # Time step
      t = self.state[1]
      t = t+1 if (t < self.n) else 1
      # State transition
      next_state = (x_new, t)
      self.state = next_state
      return next_state, r


# Dynamics (oplus operation), epsilon-greedy policy operator
# and linear features phi (normal and for monotonic MDPs)

@njit
def oplus(state, action, a):
  # If actions == 2, action is 0 for skip and 1 for flip_t;
  # if actions == n+1, action is a number from 0 to n where
  # k = 0 is for skip, k = 1...n are for flip_1...flip_n
  # (the index is shifted because x[0] is the first bit)
  x = state[0].copy()
  t = state[1]
  if (a == 2 and action == 1):
    x[t-1] = 1 - x[t-1]
  elif (action > 0):
    x[action-1] = 1 - x[action-1]
  return x

@njit
def monoplus(state, action, a):
  # If actions == 2, action is 0 for skip and 1 for set_1^t;
  # if actions == n+1, action is a number from 0 to n where
  # k = 0 is for skip, k = 1...n are for set_1^1...set_1^n
  # (the index is shifted because x[0] is the first bit)
  x = state[0].copy()
  t = state[1]
  if (a == 2 and action == 1):
    x[t-1] = 1
  elif (action > 0):
    x[action-1] = 1
  return x

# Notice that for epsilon = 0 the policy is greedy
@njit
def weighted_epsilon_greedy_policy(w, n, a, state, epsilon, Phi, masking = False):
  if random.uniform(0, 1) < epsilon:
      return random.randint(0, a-1)
  else:
      q_values = np.zeros(a)
      for j in range(a):
        q_values[j] = Phi(state, j, n, a, masking) @ w
      max_q = np.max(q_values)
      best_actions = np.where(q_values == max_q)[0]
      return np.random.choice(best_actions)

@njit
def Phi(state, action, n, a, masking = False):
  x_1 = oplus(state, action, a)
  x_2 = np.zeros(n)
  t = state[1]
  # One-Hot encoding of time
  x_2[t-1]=1
  # Masking if requested
  if masking:
    for i in range(t, n):
      x_1[i]=0
  x = np.concatenate((x_1, x_2))
  return x

@njit
def Mono_Phi(state, action, n, a, masking = False):
  x_1 = monoplus(state, action, a)
  x_2 = np.zeros(n)
  t = state[1]
  # One-Hot encoding of time
  x_2[t-1]=1
  # Masking if requested
  if masking:
    for i in range(t, n):
      x_1[i]=0
  x = np.concatenate((x_1, x_2))
  return x


# Test function


def Test(model, method, n, restricted, max_it, grid, seeds, masking : bool = False):

  seed_grid =  [(theta, alpha, rate, epsilon, seed)
        for theta, alpha, rate, epsilon in grid
        for seed in seeds]

  a = [(0.0, 0.0, 0.0, 0.0, 0.0) for (theta, alpha, rate, epsilon, seed) in seed_grid]

  A = pd.DataFrame(a,
                   index = list(seed_grid),
                   columns=['a_d', 'r_d', 'w_b', 'n_it', 'div'])

  for seed in seeds:
    np.random.seed(int(seed))
    random.seed(int(seed))

    obj = np.random.normal(loc = 0.0, scale = 1/(2*n), size=2*n)
    u = obj[0:n]
    opt = np.zeros(n)
    for i in range(n):
      opt[i] = 1 if u[i] > 0 else 0

    @njit(types.float64(types.float64[::1]))
    def f(x):
      return x @ u

    start = (np.zeros(n), 1)
    MDP = model(n, restricted, f, start)

    for (theta, alpha, rate, epsilon) in grid:
      try:
        W, it = method(MDP, max_it, theta, alpha, rate, epsilon, masking)
        if not np.all(np.isfinite(W)):
           raise ValueError("Non-finite weights")
        U = W[0:n]

        MDP.reset()
        if model is Mono_Episodic or model is Mono_Incremental:
          for i in range(n):
            action = weighted_epsilon_greedy_policy(W, n, MDP.actions, MDP.state, 0, Mono_Phi, masking)
            MDP.step(action)
        else:
          for i in range(n):
            action = weighted_epsilon_greedy_policy(W, n, MDP.actions, MDP.state, 0, Phi, masking)
            MDP.step(action)
        lim = MDP.state[0]

        approx_max = lim @ u
        actual_max = opt @ u
        abs_diff = round(actual_max - approx_max, 5)
        if actual_max > 0:
          rel_diff = round((actual_max - approx_max)/(actual_max), 5)
        else:
          rel_diff = np.NaN
        wrong_bits = sum(lim!=opt)

        A.at[(theta, alpha, rate, epsilon, seed), 'a_d'] = abs_diff
        A.at[(theta, alpha, rate, epsilon, seed), 'r_d'] = rel_diff
        A.at[(theta, alpha, rate, epsilon, seed), 'w_b'] = wrong_bits
        A.at[(theta, alpha, rate, epsilon, seed), 'n_it'] = it
      except:
        A.at[(theta, alpha, rate, epsilon, seed), 'div'] = 1.0

  return A


# Linear Sarsa Implementations


@njit(nogil=True)
def episodic_sarsa(MDP, max_it, theta, alpha, gamma, epsilon, masking : bool = False):

  # The weight vector is randomly initialized
  n = MDP.n
  a = MDP.actions
  w = np.random.normal(loc = 0.0, scale = 1/(2*n), size=2*n)

  # The algoritm goes on until convergence (infinity norm < theta)
  # or until the maximum number of iterations is reached
  converge = False
  it = 0

  while (converge == False and it < max_it):

    # S_0, A_0 are sampled every episode
    state_0 = MDP.reset()
    action_0 = weighted_epsilon_greedy_policy(w, n, a, state_0, epsilon, Phi, masking)

    # We keep track of w to check for convergence
    w_old = w.copy()

    # Single episode: n steps
    for k in range(n):

      # We observe S_t+1, R_t+1 and sample A_t+1
      phi_0 = Phi(state_0, action_0, n, a, masking)
      state_1, reward = MDP.step(action_0)
      action_1 = weighted_epsilon_greedy_policy(w, n, a, state_1, epsilon, Phi, masking)
      phi_1 = np.zeros(2*n) if (k == n-1) else Phi(state_1, action_1, n, a, masking) 

      # Weight Update
      delta = reward + gamma * phi_1 @ w - phi_0 @ w
      w = w + alpha * delta * phi_0

      # Time step
      state_0 = state_1
      action_0 = action_1

    # Convergence check
    if np.max(np.abs(w - w_old)) < theta:
      converge = True

    # Iteration counter
    it+=1

  return w, it

@njit(nogil=True)
def continuing_sarsa(MDP, max_it, theta, alpha, gamma, epsilon, masking : bool = False):

  # The weight vector is randomly initialized
  n = MDP.n
  a = MDP.actions
  w = np.random.normal(loc = 0.0, scale = 1/(2*n), size=2*n)

  # The algoritm goes on until convergence (infinity norm < theta)
  # or until the maximum number of iterations is reached
  converge = False
  it = 0

  # S_0, A_0 are sampled only once at the beginning
  state_0 = MDP.reset()
  action_0 = weighted_epsilon_greedy_policy(w, n, a, state_0, epsilon, Phi, masking)

  while (converge == False and it < max_it):

    # We keep track of w to check for convergence
    w_old = w.copy()

    # Single episode: n steps
    for k in range(n):

      # We observe S_t+1, R_t+1 and sample A_t+1
      phi_0 = Phi(state_0, action_0, n, a, masking)
      state_1, reward = MDP.step(action_0)
      action_1 = weighted_epsilon_greedy_policy(w, n, a, state_1, epsilon, Phi, masking)
      phi_1 = Phi(state_1, action_1, n, a, masking)

      # Weight Update
      delta = reward + gamma * phi_1 @ w - phi_0 @ w
      w = w + alpha * delta * phi_0

      # Time step
      state_0 = state_1
      action_0 = action_1

    # Convergence check
    if np.max(np.abs(w - w_old)) < theta:
      converge = True

    # Iteration counter
    it+=1

  return w, it

@njit(nogil=True)
def average_reward_sarsa(MDP, max_it, theta, alpha, beta, epsilon, masking : bool = False):

  # The weight vector is randomly initialized
  n = MDP.n
  a = MDP.actions
  w = np.random.normal(loc = 0.0, scale = 1/(2*n), size=2*n)
  R = 0

  # The algoritm goes on until convergence (infinity norm < theta)
  # or until the maximum number of iterations is reached
  converge = False
  it = 0

  # S_0, A_0 are sampled only once at the beginning
  state_0 = MDP.reset()
  action_0 = weighted_epsilon_greedy_policy(w, n, a, state_0, epsilon, Phi, masking)

  while (converge == False and it < max_it):

    # We keep track of w to check for convergence
    w_old = w.copy()

    # Single episode: n steps
    for k in range(n):

      # We observe S_t+1, R_t+1 and sample A_t+1
      phi_0 = Phi(state_0, action_0, n, a, masking)
      state_1, reward = MDP.step(action_0)
      action_1 = weighted_epsilon_greedy_policy(w, n, a, state_1, epsilon, Phi, masking)
      phi_1 = Phi(state_1, action_1, n, a, masking)

      # Weight Update
      delta = reward - R + phi_1 @ w - phi_0 @ w
      R = R + beta * delta
      w = w + alpha * delta * phi_0

      # Time step
      state_0 = state_1
      action_0 = action_1

    # Convergence check
    if np.max(np.abs(w - w_old)) < theta:
      converge = True

    # Iteration counter
    it+=1

  return w, it

@njit(nogil=True)
def monotonic_sarsa(MDP, max_it, theta, alpha, gamma, epsilon, masking : bool = False):

  # The weight vector is randomly initialized
  n = MDP.n
  a = MDP.actions
  w = np.random.normal(loc = 0.0, scale = 1/(2*n), size=2*n)

  # The algoritm goes on until convergence (infinity norm < theta)
  # or until the maximum number of iterations is reached
  converge = False
  it = 0

  while (converge == False and it < max_it):

    # S_0, A_0 are sampled every episode
    state_0 = MDP.reset()
    action_0 = weighted_epsilon_greedy_policy(w, n, a, state_0, epsilon, Mono_Phi, masking)

    # We keep track of w to check for convergence
    w_old = w.copy()

    # Single episode: n steps
    for k in range(n):

      # We observe S_t+1, R_t+1 and sample A_t+1
      phi_0 = Mono_Phi(state_0, action_0, n, a, masking)
      state_1, reward = MDP.step(action_0)
      action_1 = weighted_epsilon_greedy_policy(w, n, a, state_1, epsilon, Mono_Phi, masking)
      phi_1 = np.zeros(2*n) if (k == n-1) else Mono_Phi(state_1, action_1, n, a, masking) 

      # Weight Update
      delta = reward + gamma * phi_1 @ w - phi_0 @ w
      w = w + alpha * delta * phi_0

      # Time step
      state_0 = (state_1[0].copy(), state_1[1])
      action_0 = action_1

    # Convergence check
    if np.max(np.abs(w - w_old)) < theta:
      converge = True

    # Iteration counter
    it+=1

  return w, it


# Parallelization + optimization

@njit
def seed_numba(seed, objective_n=0):
    random.seed(seed)
    np.random.seed(seed)
    if objective_n:
        np.random.normal(0.0, 1 / (2 * objective_n), 2 * objective_n)
# RNG is advanced to ensure reproducibility of our results


def fit_grid(model, method, n, restricted, f, max_it, grid, masking, seed):
    warmup = model(n, restricted, f, (np.zeros(n), 1))
    method(warmup, 0, *grid[0], masking)

    def fit_one(parameters):
        seed_numba(seed, n)
        MDP = model(n, restricted, f, (np.zeros(n), 1))
        try:
            return method(MDP, max_it, *parameters, masking)
        except Exception as err:
            return err

    if workers == 1:
        return list(map(fit_one, grid))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fit_one, grid))


def parallel_method(model, method, grid, restricted, masking, seed):
    results = None

    def fitted_method(MDP, max_it, theta, alpha, rate, epsilon, masking):
        nonlocal results
        if results is None:
            results = iter(fit_grid(model, method, MDP.n, restricted, MDP.f,
                                    max_it, grid, masking, seed))
        result = next(results)
        if isinstance(result, Exception):
            raise result
        seed_numba(seed)
        return result

    return fitted_method


# Best performing hyperparameters from the grid search

Episodic_M_grid = [(1e-06, 0.01, 0.9, 0.2)]
Continuing_UM_grid = [(1e-06, 0.0005, 0.8, 0.2)]
Average_Reward_UM_grid = [(1e-06, 0.001, 0.0001, 0.2)]
Average_Reward_M_grid = [(1e-06, 0.005, 0.0005, 0.2)]

seed = 243
random.seed(seed)
np.random.seed(seed)
seeds = np.random.choice(100000, size=1000, replace=False).tolist()

k = 11
n = int(k*(k-1)/2)
max_it = 50000
workers = 1

def run_seed(task):
    model_name, method_name, restricted, masking, grid, seed = task
    model = globals()[model_name]
    method = globals()[method_name]

    accelerated = parallel_method(
        model, method, grid, restricted, masking, seed
    )
    return Test(
        model, accelerated, n, restricted, max_it,
        grid, seeds=[seed], masking=masking
    )

def main():
  cases = [
      ("Episodic M R", Loop, episodic_sarsa, True, True, Episodic_M_grid),
      ("Continuing UM R", Loop, continuing_sarsa, True, False, Continuing_UM_grid),
      ("Average reward UM R", Loop, average_reward_sarsa, True, False, Average_Reward_UM_grid),
      ("Average reward M R", Loop, average_reward_sarsa, True, True, Average_Reward_M_grid),
  ]
  for name, model, method, restricted, masking, method_grid in cases:
          tasks = [(model.__name__,
                    method.__name__,
                    restricted,
                    masking,
                    method_grid,
                    int(seed)) for seed in seeds]
  
          with get_context("spawn").Pool(16) as pool:
              res_list = pool.map(run_seed, tasks)
  
          # Combine all five seeds into one table for each case
          results = pd.concat(res_list)
  
          rate = "beta" if method is average_reward_sarsa else "gamma"
          results.index = pd.MultiIndex.from_tuples(
              results.index,
              names=["theta", "alpha", rate, "epsilon", "seed"]
          )
  
          filename = name.replace(" ", "_") + ".csv"
          results.to_csv(filename)
          print(f"Saved {filename}: {len(results)} rows", flush=True)
  
if __name__ == "__main__":
  main()