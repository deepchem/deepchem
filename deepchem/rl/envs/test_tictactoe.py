from unittest import TestCase
import numpy as np

import deepchem.rl.envs.tictactoe


class TestTicTacToeEnvironment(TestCase):

    def test_constructor(self):
        board = deepchem.rl.envs.tictactoe.TicTacToeEnvironment()
        assert len(board.state) == 1
        assert board.state[0].shape == (3, 3, 2)
        assert np.sum(board.state[0]) == 1 or np.sum(board.state[0]) == 0

    def test_step(self):
        board = deepchem.rl.envs.tictactoe.TicTacToeEnvironment()
        X = deepchem.rl.envs.tictactoe.TicTacToeEnvironment.X
        board._state = [np.zeros(shape=(3, 3, 2), dtype=np.float32)]
        board.step(0)
        assert np.all(board.state[0][0][0] == X)

    def test_winner(self):
        board = deepchem.rl.envs.tictactoe.TicTacToeEnvironment()
        X = deepchem.rl.envs.tictactoe.TicTacToeEnvironment.X
        board.state[0][0][0] = X
        board.state[0][0][1] = X
        assert not board.check_winner(X)
        board.state[0][0][2] = X
        assert board.check_winner(X)

    def test_game_over(self):
        board = deepchem.rl.envs.tictactoe.TicTacToeEnvironment()
        X = deepchem.rl.envs.tictactoe.TicTacToeEnvironment.X
        board.state[0][0][0] = X
        board.state[0][0][1] = X
        assert not board.check_winner(X)
        board.state[0][0][2] = X
        assert board.check_winner(X)

    def test_display(self):
        board = deepchem.rl.envs.tictactoe.TicTacToeEnvironment()
        s = board.display()
        assert s.find("X") == -1

    def test_get_O_move(self):
        board = deepchem.rl.envs.tictactoe.TicTacToeEnvironment()
        empty = deepchem.rl.envs.tictactoe.TicTacToeEnvironment.EMPTY
        move = board.get_O_move()
        assert np.all(board.state[0][move[0]][move[1]] == empty)
