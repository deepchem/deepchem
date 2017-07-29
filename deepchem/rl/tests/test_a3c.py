from flaky import flaky

import deepchem as dc
from deepchem.models.tensorgraph.layers import Reshape, Variable, SoftMax, GRU, Dense
from deepchem.models.tensorgraph.optimizers import Adam, PolynomialDecay
import numpy as np
import tensorflow as tf
import unittest
from nose.plugins.attrib import attr


class TestA3C(unittest.TestCase):

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

    a3c = dc.rl.A3C(
        env,
        TestPolicy(),
        max_rollout_length=20,
        optimizer=Adam(learning_rate=0.001))
    a3c.fit(100000)

    # It should have learned that the expected value is very close to zero, and that the best
    # action is to walk away.

    action_prob, value = a3c.predict([[0]])
    assert -0.5 < value[0] < 0.5
    assert action_prob.argmax() == 37
    assert a3c.select_action([[0]], deterministic=True) == 37

    # Verify that we can create a new A3C object, reload the parameters from the first one, and
    # get the same result.

    new_a3c = dc.rl.A3C(env, TestPolicy(), model_dir=a3c._graph.model_dir)
    new_a3c.restore()
    action_prob2, value2 = new_a3c.predict([[0]])
    assert value2 == value

    # Do the same thing, only using the "restore" argument to fit().

    new_a3c = dc.rl.A3C(env, TestPolicy(), model_dir=a3c._graph.model_dir)
    new_a3c.fit(0, restore=True)
    action_prob2, value2 = new_a3c.predict([[0]])
    assert value2 == value

  def test_recurrent_states(self):
    """Test a policy that involves recurrent layers."""

    # The environment just has a constant state.

    class TestEnvironment(dc.rl.Environment):

      def __init__(self):
        super(TestEnvironment, self).__init__((10,), 10)
        self._state = np.random.random(10)

      def step(self, action):
        self._state = np.random.random(10)
        return 0.0

      def reset(self):
        pass

    # The policy includes a single recurrent layer.

    class TestPolicy(dc.rl.Policy):

      def create_layers(self, state, **kwargs):

        reshaped = Reshape(shape=(1, -1, 10), in_layers=state)
        gru = GRU(n_hidden=10, batch_size=1, in_layers=reshaped)
        output = SoftMax(
            in_layers=[Reshape(in_layers=[gru], shape=(-1, env.n_actions))])
        value = Variable([0.0])
        return {'action_prob': output, 'value': value}

    # We don't care about actually optimizing it, so just run a few rollouts to make
    # sure fit() doesn't crash, then check the behavior of the GRU state.

    env = TestEnvironment()
    a3c = dc.rl.A3C(env, TestPolicy())
    a3c.fit(100)
    # On the first call, the initial state should be all zeros.
    prob1, value1 = a3c.predict(
        env.state, use_saved_states=True, save_states=False)
    # It should still be zeros since we didn't save it last time.
    prob2, value2 = a3c.predict(
        env.state, use_saved_states=True, save_states=True)
    # It should be different now.
    prob3, value3 = a3c.predict(
        env.state, use_saved_states=True, save_states=False)
    # This should be the same as the previous one.
    prob4, value4 = a3c.predict(
        env.state, use_saved_states=True, save_states=False)
    # Now we reset it, so we should get the same result as initially.
    prob5, value5 = a3c.predict(
        env.state, use_saved_states=False, save_states=True)
    assert np.array_equal(prob1, prob2)
    assert np.array_equal(prob1, prob5)
    assert np.array_equal(prob3, prob4)
    assert not np.array_equal(prob2, prob3)

  @attr('slow')
  def test_hindsight(self):
    """Test Hindsight Experience Replay."""

    # The environment is a plane in which the agent moves by steps until it reaches a randomly
    # positioned goal.  No reward is given until it reaches the goal.  That makes it very hard
    # to learn by standard methods, since it may take a very long time to receive any feedback
    # at all.  Using hindsight makes it much easier.

    class TestEnvironment(dc.rl.Environment):

      def __init__(self):
        super(TestEnvironment, self).__init__((4,), 4)
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

      def reset(self):
        self._state = np.concatenate([[0, 0], np.random.randint(-50, 50, 2)])
        self._terminated = False
        self.count = 0

      def step(self, action):
        new_state = self._state.copy()
        new_state[:2] += self.moves[action]
        self._state = new_state
        self.count += 1
        reward = 0
        if np.array_equal(new_state[:2], new_state[2:]):
          self._terminated = True
          reward = 1
        elif self.count == 1000:
          self._terminated = True
        return reward

      def apply_hindsight(self, states, actions, goal):
        new_states = []
        rewards = []
        goal_pos = goal[:2]
        for state, action in zip(states, actions):
          new_state = state.copy()
          new_state[2:] = goal_pos
          new_states.append(new_state)
          pos_after_action = new_state[:2] + self.moves[action]
          if np.array_equal(pos_after_action, goal_pos):
            rewards.append(1)
          else:
            rewards.append(0)
        return new_states, rewards

    # A simple policy with two hidden layers.

    class TestPolicy(dc.rl.Policy):

      def create_layers(self, state, **kwargs):

        dense1 = Dense(6, activation_fn=tf.nn.relu, in_layers=state)
        dense2 = Dense(6, activation_fn=tf.nn.relu, in_layers=dense1)
        output = Dense(
            4,
            activation_fn=tf.nn.softmax,
            biases_initializer=None,
            in_layers=dense2)
        value = Dense(1, in_layers=dense2)
        return {'action_prob': output, 'value': value}

    # Optimize it.

    env = TestEnvironment()
    learning_rate = PolynomialDecay(
        initial_rate=0.0005, final_rate=0.0002, decay_steps=2000000)
    a3c = dc.rl.A3C(
        env,
        TestPolicy(),
        use_hindsight=True,
        optimizer=Adam(learning_rate=learning_rate))
    a3c.fit(2000000)

    # Try running it a few times and see if it succeeds.

    pass_count = 0
    for i in range(5):
      env.reset()
      while not env.terminated:
        env.step(a3c.select_action(env.state))
      if np.array_equal(env.state[:2], env.state[2:]):
        pass_count += 1
    assert pass_count >= 3
