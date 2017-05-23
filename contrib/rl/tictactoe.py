import deepchem as dc
import numpy as np
import random
import tensorflow as tf
import time
import copy

from deepchem.models.tensorgraph.layers import Flatten, Dense, SoftMax, \
  Variable, \
  Feature, Layer, Add, BatchNorm, Conv2D
from deepchem.rl.a3c import _Worker


class TicTacToeEnvironment(dc.rl.Environment):
  X = np.array([1.0, 0.0])
  O = np.array([0.0, 1.0])
  EMPTY = np.array([0.0, 0.0])

  ILLEGAL_MOVE_PENALTY = -3.0
  LOSS_PENALTY = -3.0
  NOT_LOSS = 0.1
  DRAW_REWARD = 5.0
  WIN_REWARD = 10.0

  def __init__(self):
    super().__init__([(3, 3, 2)], 9)
    self.reset()

  def reset(self):
    self._terminated = False
    self._state = [np.zeros(shape=(3, 3, 2), dtype=np.float32)]

    # Randomize who goes first
    if random.randint(0, 1) == 1:
      move = self.get_O_move()
      self._state[0][move[0]][move[1]] = TicTacToeEnvironment.O

  def step(self, action):
    self._state = copy.deepcopy(self._state)
    row = action // 3
    col = action % 3

    # Illegal move -- the square is not empty
    if not np.all(self._state[0][row][col] == TicTacToeEnvironment.EMPTY):
      self._terminated = True
      return TicTacToeEnvironment.ILLEGAL_MOVE_PENALTY

    # Move X
    self._state[0][row][col] = TicTacToeEnvironment.X

    # Did X Win
    if self.check_winner(TicTacToeEnvironment.X):
      print("Winner")
      self._terminated = True
      return TicTacToeEnvironment.WIN_REWARD

    if self.game_over():
      self._terminated = True
      print("Draw")
      return TicTacToeEnvironment.DRAW_REWARD

    move = self.get_O_move()
    self._state[0][move[0]][move[1]] = TicTacToeEnvironment.O

    # Did O Win
    if self.check_winner(TicTacToeEnvironment.O):
      self._terminated = True
      return TicTacToeEnvironment.LOSS_PENALTY

    if self.game_over():
      self._terminated = True
      return TicTacToeEnvironment.DRAW_REWARD
    return TicTacToeEnvironment.NOT_LOSS

  def get_O_move(self):
    empty_squares = []
    for row in range(3):
      for col in range(3):
        if np.all(self._state[0][row][col] == TicTacToeEnvironment.EMPTY):
          empty_squares.append((row, col))
    return random.choice(empty_squares)

  def check_winner(self, player):
    for i in range(3):
      row = np.sum(self._state[0][i][:])
      if np.all(row == player * 3):
        return True
      col = np.sum(self._state[0][:][i])
      if np.all(col == player * 3):
        return True
    return False

  def game_over(self):
    s = set()
    for i in range(3):
      for j in range(3):
        if np.all(self._state[0][i][j] == TicTacToeEnvironment.EMPTY):
          return False
    return True

  def display(self):
    state = self._state[0]
    s = ""
    for row in range(3):
      for col in range(3):
        if np.all(state[row][col] == TicTacToeEnvironment.EMPTY):
          s += "_"
        if np.all(state[row][col] == TicTacToeEnvironment.X):
          s += "X"
        if np.all(state[row][col] == TicTacToeEnvironment.O):
          s += "O"
      s += "\n"
    return s


class TicTacToePolicy(dc.rl.Policy):

  def create_layers(self, state, **kwargs):
    d1 = Conv2D(num_outputs=64, kernel_size=3, in_layers=state)
    d1 = Flatten(in_layers=[d1])
    d2 = Dense(
        in_layers=[d1],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.nn.l2_normalize,
        normalizer_params={"dim": 1},
        out_channels=64)
    d3 = Dense(
        in_layers=[d2],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.nn.l2_normalize,
        normalizer_params={"dim": 1},
        out_channels=32)
    d4 = Dense(
        in_layers=[d3],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.nn.l2_normalize,
        normalizer_params={"dim": 1},
        out_channels=16)
    d4 = BatchNorm(in_layers=[d4])
    d5 = Dense(in_layers=[d4], activation_fn=None, out_channels=9)
    value = Dense(in_layers=[d4], activation_fn=None, out_channels=1)
    probs = SoftMax(in_layers=[d5])
    return {'action_prob': probs, 'value': value}


def main():
  env = TicTacToeEnvironment()
  policy = TicTacToePolicy()

  start = time.time()
  end = time.time()
  timeout = 60 * 60  # One Hour
  while end - start < timeout:
    a3c = dc.rl.A3C(
        env, policy, entropy_weight=0, model_dir="/home/leswing/tictactoe")
    a3c.optimizer = dc.models.tensorgraph.TFWrapper(
        tf.train.AdamOptimizer, learning_rate=0.01)
    TicTacToeEnvironment.ILLEGAL_MOVE_PENALTY = -10000.0
    try:
      a3c.restore()
    except:
      print("Restore Failed")
      pass
    a3c.fit(100000)
    env.reset()
    while not env._terminated:
      print(env.display())
      print(a3c.predict(env._state))
      action = a3c.select_action(env._state)
      print(action)
      print(env.step(action))
    print(env.display())
    end = time.time()


if __name__ == "__main__":
  main()
