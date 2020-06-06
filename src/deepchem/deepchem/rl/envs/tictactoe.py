import numpy as np
import copy
import random
import deepchem


class TicTacToeEnvironment(deepchem.rl.Environment):
  """
    Play tictactoe against a randomly acting opponent
    """
  X = np.array([1.0, 0.0])
  O = np.array([0.0, 1.0])
  EMPTY = np.array([0.0, 0.0])

  ILLEGAL_MOVE_PENALTY = -3.0
  LOSS_PENALTY = -3.0
  NOT_LOSS = 0.1
  DRAW_REWARD = 5.0
  WIN_REWARD = 10.0

  def __init__(self):
    super(TicTacToeEnvironment, self).__init__([(3, 3, 2)], 9)
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
      self._terminated = True
      return TicTacToeEnvironment.WIN_REWARD

    if self.game_over():
      self._terminated = True
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
      row = np.sum(self._state[0][i][:], axis=0)
      if np.all(row == player * 3):
        return True
      col = np.sum(self._state[0][:][i], axis=0)
      if np.all(col == player * 3):
        return True

    diag1 = self._state[0][0][0] + self._state[0][1][1] + self._state[0][2][2]
    if np.all(diag1 == player * 3):
      return True
    diag2 = self._state[0][0][2] + self._state[0][1][1] + self._state[0][2][0]
    if np.all(diag2 == player * 3):
      return True
    return False

  def game_over(self):
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
