import deepchem as dc
import numpy as np
import pytest
from deepchem.models.optimizers import Adam
try:
    from deepchem.rl.torch_rl import A2C, PPO
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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

            class TestModel(nn.Module):

                def __init__(self):
                    super(TestModel, self).__init__()
                    self.action = nn.Parameter(
                        torch.ones(env.n_actions, dtype=torch.float32))
                    self.value = nn.Parameter(
                        torch.tensor([0.0], dtype=torch.float32))

                def forward(self, inputs):
                    prob = F.softmax(
                        torch.reshape(self.action, (-1, env.n_actions)))
                    return prob, self.value

            return TestModel()

    has_pytorch = True
except:
    has_pytorch = False


@pytest.mark.torch
def test_a2c_reload():
    env = RouletteEnvironment()
    policy = TestPolicy(env)

    a2c = A2C(env,
              policy,
              max_rollout_length=20,
              optimizer=Adam(learning_rate=0.001))
    a2c.fit(1000)
    action_prob, value = a2c.predict([[0]])

    new_a2c = A2C(env, policy, model_dir=a2c._model.model_dir)
    new_a2c.restore()
    action_prob2, value2 = new_a2c.predict([[0]])

    assert np.all(action_prob == action_prob2)
    assert value == value2


@pytest.mark.torch
def test_ppo_reload():
    env = RouletteEnvironment()
    policy = TestPolicy(env)
    ppo = PPO(env,
              policy,
              max_rollout_length=20,
              optimization_epochs=8,
              optimizer=Adam(learning_rate=0.003))
    ppo.fit(1000)
    action_prob, value = ppo.predict([[0]])

    new_ppo = PPO(env, policy, model_dir=ppo._model.model_dir)
    new_ppo.restore()
    action_prob2, value2 = new_ppo.predict([[0]])

    assert np.all(action_prob == action_prob2)
    assert value == value2
