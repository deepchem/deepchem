from flaky import flaky

import deepchem as dc
from deepchem.models.tensorgraph.layers import Reshape, Variable, SoftMax, GRU, Dense
from deepchem.models.tensorgraph.optimizers import Adam, PolynomialDecay
import numpy as np
import tensorflow as tf
import unittest
from nose.plugins.attrib import attr


class TestMCTS(unittest.TestCase):

  @flaky
  def test_roulette(self):
    """Test training a policy for the roulette environment."""

    # This is modeled after the Roulette-v0 environment from OpenAI Gym.
    # The player can bet on any number from 0 to 36, or walk away (which ends the
    # game).  The average reward for any bet is slightly negative, so the best
    # strategy is to walk away.

    class RouletteEnvironment(dc.rl.Environment):

      def __init__(self):
        super(RouletteEnvironment, self).__init__([(1,)], 38)
        self._state = [np.array([0])]

      def step(self, action):
        if action == 37:
          self._terminated = True  # Walk away.
          return 0.0
        wheel = np.random.randint(37)
        if wheel == 0:
          if action == 0:
            return 35.0
          return -1.0
        if action != 0 and wheel % 2 == action % 2:
          return 1.0
        return -1.0

      def reset(self):
        self._terminated = False

    env = RouletteEnvironment()

    # This policy just learns a constant probability for each action, and a constant for the value.

    class TestPolicy(dc.rl.Policy):

      def create_layers(self, state, **kwargs):
        action = Variable(np.ones(env.n_actions))
        output = SoftMax(
            in_layers=[Reshape(in_layers=[action], shape=(-1, env.n_actions))])
        value = Variable([0.0])
        return {'action_prob': output, 'value': value}

    # Optimize it.

    mcts = dc.rl.MCTS(
        env,
        TestPolicy(),
        max_search_depth=5,
        n_search_episodes=200,
        optimizer=Adam(learning_rate=0.005))
    mcts.fit(10, steps_per_iteration=50, epochs_per_iteration=50)

    # It should have learned that the expected value is very close to zero, and that the best
    # action is to walk away.

    action_prob, value = mcts.predict([[0]])
    assert -0.5 < value[0] < 0.5
    assert action_prob.argmax() == 37
    assert mcts.select_action([[0]], deterministic=True) == 37

    # Verify that we can create a new MCTS object, reload the parameters from the first one, and
    # get the same result.

    new_mcts = dc.rl.MCTS(env, TestPolicy(), model_dir=mcts._graph.model_dir)
    new_mcts.restore()
    action_prob2, value2 = new_mcts.predict([[0]])
    assert value2 == value

    # Do the same thing, only using the "restore" argument to fit().

    new_mcts = dc.rl.MCTS(env, TestPolicy(), model_dir=mcts._graph.model_dir)
    new_mcts.fit(0, restore=True)
    action_prob2, value2 = new_mcts.predict([[0]])
    assert value2 == value
