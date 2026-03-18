import unittest

import pytest
import numpy as np
from flaky import flaky
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, GRU, Reshape, Softmax
    has_tensorflow = True
except:
    has_tensorflow = False

import deepchem as dc
from deepchem.models.optimizers import Adam, PolynomialDecay


class TestA2C(unittest.TestCase):

    @flaky
    @pytest.mark.tensorflow
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

            def __init__(self):
                super(TestPolicy, self).__init__(['action_prob', 'value'])

            def create_model(self, **kwargs):

                class TestModel(tf.keras.Model):

                    def __init__(self):
                        super(TestModel, self).__init__(**kwargs)
                        self.action = tf.Variable(
                            np.ones(env.n_actions, np.float32))
                        self.value = tf.Variable([0.0], tf.float32)

                    def call(self, inputs, **kwargs):
                        prob = tf.nn.softmax(
                            tf.reshape(self.action, (-1, env.n_actions)))
                        return (prob, self.value)

                return TestModel()

        # Optimize it.

        a2c = dc.rl.A2C(env,
                        TestPolicy(),
                        max_rollout_length=20,
                        optimizer=Adam(learning_rate=0.001))
        a2c.fit(100000)

        # It should have learned that the expected value is very close to zero, and that the best
        # action is to walk away.  (To keep the test fast, we allow that to be either of the two
        # top actions).

        action_prob, value = a2c.predict([[0]])
        assert -0.5 < value[0] < 0.5
        assert action_prob.argmax() == 37
        assert 37 in np.argsort(action_prob.flatten())[-2:]
        assert a2c.select_action([[0]],
                                 deterministic=True) == action_prob.argmax()

        # Verify that we can create a new A2C object, reload the parameters from the first one, and
        # get the same result.

        new_a2c = dc.rl.A2C(env, TestPolicy(), model_dir=a2c._model.model_dir)
        new_a2c.restore()
        action_prob2, value2 = new_a2c.predict([[0]])
        assert value2 == value

        # Do the same thing, only using the "restore" argument to fit().

        new_a2c = dc.rl.A2C(env, TestPolicy(), model_dir=a2c._model.model_dir)
        new_a2c.fit(0, restore=True)
        action_prob2, value2 = new_a2c.predict([[0]])
        assert value2 == value

    @pytest.mark.tensorflow
    def test_recurrent_states(self):
        """Test a policy that involves recurrent layers."""

        # The environment just has a constant state.

        class TestEnvironment(dc.rl.Environment):

            def __init__(self):
                super(TestEnvironment, self).__init__((10,), 10)
                self._state = np.random.random(10).astype(np.float32)

            def step(self, action):
                self._state = np.random.random(10).astype(np.float32)
                return 0.0

            def reset(self):
                pass

        # The policy includes a single recurrent layer.

        class TestPolicy(dc.rl.Policy):

            def __init__(self):
                super(TestPolicy,
                      self).__init__(['action_prob', 'value', 'rnn_state'],
                                     [np.zeros(10)])

            def create_model(self, **kwargs):
                state = Input(shape=(10,))
                rnn_state = Input(shape=(10,))
                reshaped = Reshape((1, 10))(state)
                gru, rnn_final_state = GRU(10,
                                           return_state=True,
                                           return_sequences=True,
                                           time_major=True)(
                                               reshaped,
                                               initial_state=rnn_state)
                output = Softmax()(Reshape((10,))(gru))
                value = dc.models.layers.Variable([0.0])([state])
                return tf.keras.Model(inputs=[state, rnn_state],
                                      outputs=[output, value, rnn_final_state])

        # We don't care about actually optimizing it, so just run a few rollouts to make
        # sure fit() doesn't crash, then check the behavior of the GRU state.

        env = TestEnvironment()
        a2c = dc.rl.A2C(env, TestPolicy())
        a2c.fit(100)
        # On the first call, the initial state should be all zeros.
        prob1, value1 = a2c.predict(env.state,
                                    use_saved_states=True,
                                    save_states=False)
        # It should still be zeros since we didn't save it last time.
        prob2, value2 = a2c.predict(env.state,
                                    use_saved_states=True,
                                    save_states=True)
        # It should be different now.
        prob3, value3 = a2c.predict(env.state,
                                    use_saved_states=True,
                                    save_states=False)
        # This should be the same as the previous one.
        prob4, value4 = a2c.predict(env.state,
                                    use_saved_states=True,
                                    save_states=False)
        # Now we reset it, so we should get the same result as initially.
        prob5, value5 = a2c.predict(env.state,
                                    use_saved_states=False,
                                    save_states=True)
        assert np.array_equal(prob1, prob2)
        assert np.array_equal(prob1, prob5)
        assert np.array_equal(prob3, prob4)
        assert not np.array_equal(prob2, prob3)

    @flaky
    @pytest.mark.slow
    @pytest.mark.tensorflow
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
                self._state = np.concatenate([[0, 0],
                                              np.random.randint(-50, 50, 2)])
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
                        break
                    else:
                        rewards.append(0)
                return new_states, rewards

        # A simple policy with two hidden layers.

        class TestPolicy(dc.rl.Policy):

            def __init__(self):
                super(TestPolicy, self).__init__(['action_prob', 'value'])

            def create_model(self, **kwargs):
                state = Input(shape=(4,))
                dense1 = Dense(6, activation=tf.nn.relu)(state)
                dense2 = Dense(6, activation=tf.nn.relu)(dense1)
                output = Dense(4, activation=tf.nn.softmax,
                               use_bias=False)(dense2)
                value = Dense(1)(dense2)
                return tf.keras.Model(inputs=state, outputs=[output, value])

        # Optimize it.

        env = TestEnvironment()
        a2c = dc.rl.A2C(env,
                        TestPolicy(),
                        use_hindsight=True,
                        optimizer=Adam(learning_rate=0.001))
        a2c.fit(1000000)

        # Try running it a few times and see if it succeeds.

        pass_count = 0
        for i in range(5):
            env.reset()
            while not env.terminated:
                env.step(a2c.select_action(env.state))
            if np.array_equal(env.state[:2], env.state[2:]):
                pass_count += 1
        assert pass_count >= 3

    @pytest.mark.tensorflow
    def test_continuous(self):
        """Test A2C on an environment with a continous action space."""

        # The state consists of two numbers: a current value and a target value.
        # The policy just needs to learn to output the target value (or at least
        # move toward it).

        class TestEnvironment(dc.rl.Environment):

            def __init__(self):
                super(TestEnvironment, self).__init__((2,), action_shape=(1,))

            def reset(self):
                target = np.random.uniform(-50, 50)
                self._state = np.array([0, target], dtype=np.float32)
                self._terminated = False
                self.count = 0

            def step(self, action):
                target = self._state[1]
                dist = np.abs(target - action[0])
                old_dist = np.abs(target - self._state[0])
                new_state = np.array([action[0], target], dtype=np.float32)
                self._state = new_state
                self.count += 1
                reward = old_dist - dist
                self._terminated = (self.count == 10)
                return reward

        # A simple policy with no hidden layers.

        class TestPolicy(dc.rl.Policy):

            def __init__(self):
                super(TestPolicy,
                      self).__init__(['action_mean', 'action_std', 'value'])

            def create_model(self, **kwargs):

                class TestModel(tf.keras.Model):

                    def __init__(self):
                        super(TestModel, self).__init__(**kwargs)
                        self.mean = Dense(1, kernel_initializer='zeros')
                        self.std = tf.constant([10.0])
                        self.value = Dense(1)

                    def call(self, inputs, **kwargs):
                        return (self.mean(inputs[0]), self.std,
                                self.value(inputs[0]))

                return TestModel()

        # Optimize it.

        env = TestEnvironment()
        learning_rate = PolynomialDecay(initial_rate=0.005,
                                        final_rate=0.0005,
                                        decay_steps=25000)
        a2c = dc.rl.A2C(env,
                        TestPolicy(),
                        discount_factor=0,
                        optimizer=Adam(learning_rate=learning_rate))
        a2c.fit(25000)

        # Try running it and see if it reaches the target

        env.reset()
        while not env.terminated:
            env.step(a2c.select_action(env.state, deterministic=True))
        distance = np.abs(env.state[0] - env.state[1])
        tolerance = max(1.0, 0.1 * np.abs(env.state[1]))
        assert distance < tolerance
