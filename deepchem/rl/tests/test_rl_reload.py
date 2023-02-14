import deepchem as dc
import numpy as np
import pytest
from deepchem.models.optimizers import Adam
try:
    import tensorflow as tf

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

    # This policy just learns a constant probability for each action, and a constant for the value.

    class TestPolicy(dc.rl.Policy):

        def __init__(self, env):
            super(TestPolicy, self).__init__(['action_prob', 'value'])
            self.env = env

        def create_model(self, **kwargs):
            env = self.env

            class TestModel(tf.keras.Model):

                def __init__(self):
                    super(TestModel, self).__init__(**kwargs)
                    self.action = tf.Variable(np.ones(env.n_actions,
                                                      np.float32))
                    self.value = tf.Variable([0.0], tf.float32)

                def call(self, inputs, **kwargs):
                    prob = tf.nn.softmax(
                        tf.reshape(self.action, (-1, env.n_actions)))
                    return (prob, self.value)

            return TestModel()

    has_tensorflow = True
except:
    has_tensorflow = False


@pytest.mark.tensorflow
def test_a2c_reload():
    env = RouletteEnvironment()
    policy = TestPolicy(env)

    a2c = dc.rl.A2C(env,
                    policy,
                    max_rollout_length=20,
                    optimizer=Adam(learning_rate=0.001))
    a2c.fit(1000)
    action_prob, value = a2c.predict([[0]])

    new_a2c = dc.rl.A2C(env, policy, model_dir=a2c._model.model_dir)
    new_a2c.restore()
    action_prob2, value2 = new_a2c.predict([[0]])

    assert np.all(action_prob == action_prob2)
    assert value == value2


@pytest.mark.tensorflow
def test_ppo_reload():
    env = RouletteEnvironment()
    policy = TestPolicy(env)
    ppo = dc.rl.PPO(env,
                    policy,
                    max_rollout_length=20,
                    optimization_epochs=8,
                    optimizer=Adam(learning_rate=0.003))
    ppo.fit(1000)
    action_prob, value = ppo.predict([[0]])

    new_ppo = dc.rl.PPO(env, policy, model_dir=ppo._model.model_dir)
    new_ppo.restore()
    action_prob2, value2 = new_ppo.predict([[0]])

    assert np.all(action_prob == action_prob2)
    assert value == value2
