import unittest
import deepchem as dc
import pytest
import numpy as np
from flaky import flaky
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from deepchem.rl.torch_rl import A2CLossDiscrete, A2CLossContinuous, A2C
    from deepchem.models.optimizers import Adam
    has_pytorch = True
except:
    has_pytorch = False


@pytest.mark.torch
def test_A2CLossDiscrete():
    outputs = [
        torch.tensor([[
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.2, 0.2, 0.2, 0.2, 0.2,
            0.2, 0.2, 0.2, 0.2
        ]]),
        torch.tensor([0.], requires_grad=True)
    ]
    labels = np.array([[
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0.
    ]],
                      dtype=np.float32)
    discount = np.array([
        -1.0203744, -0.02058018, 0.98931295, 2.009407, 1.019603, 0.01980097,
        -0.9901, 0.01, -1., 0.
    ],
                        dtype=np.float32)
    advantage = np.array([
        -1.0203744, -0.02058018, 0.98931295, 2.009407, 1.019603, 0.01980097,
        -0.9901, 0.01, -1., 0.
    ],
                         dtype=np.float32)
    loss = A2CLossDiscrete(value_weight=1.0,
                           entropy_weight=0.01,
                           action_prob_index=0,
                           value_index=1)
    loss_val = loss(outputs, [labels], [discount, advantage])
    assert round(loss_val.item(), 4) == 1.2541


@pytest.mark.torch
def test_A2CLossContinuous():
    outputs = [
        torch.tensor(
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
            dtype=torch.float32,
            requires_grad=True),
        torch.tensor([10.], dtype=torch.float32, requires_grad=True),
        torch.tensor([[27.717865], [28.860144]],
                     dtype=torch.float32,
                     requires_grad=True)
    ]
    labels = np.array(
        [[-4.897339], [3.4308329], [-4.527725], [-7.3000813], [-1.9869075],
         [20.664988], [-8.448957], [10.580486], [10.017258], [17.884674]],
        dtype=np.float32)
    discount = np.array([
        4.897339, -8.328172, 7.958559, 2.772356, -5.313174, -22.651896,
        29.113945, -19.029444, 0.56322646, -7.867417
    ],
                        dtype=np.float32)
    advantage = np.array([
        -5.681633, -20.57494, -1.4520378, -9.348538, -18.378199, -33.907513,
        25.572464, -32.485718, -6.412546, -15.034998
    ],
                         dtype=np.float32)
    loss = A2CLossContinuous(value_weight=1.0,
                             entropy_weight=0.01,
                             mean_index=0,
                             std_index=1,
                             value_index=2)
    loss_val = loss(outputs, [labels], [discount, advantage])
    assert round(loss_val.item(), 4) == 1050.2310


class TestA2C(unittest.TestCase):

    @flaky
    @pytest.mark.torch
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

        # Optimize it.

        a2c = A2C(env,
                  TestPolicy(),
                  max_rollout_length=20,
                  optimizer=Adam(learning_rate=0.001))
        a2c.fit(100000)
        action_prob, value = a2c.predict([[0]])

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

        new_a2c = A2C(env, TestPolicy(), model_dir=a2c._model.model_dir)
        new_a2c.restore()
        action_prob2, value2 = new_a2c.predict([[0]])
        assert value2 == value

        # Do the same thing, only using the "restore" argument to fit().

        new_a2c = A2C(env, TestPolicy(), model_dir=a2c._model.model_dir)
        new_a2c.fit(0, restore=True)
        action_prob2, value2 = new_a2c.predict([[0]])
        assert value2 == value

    @pytest.mark.torch
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
                                     [np.zeros(10, dtype=np.float32)])

            def create_model(self, **kwargs):

                class TestModel(nn.Module):

                    def __init__(self):
                        super(TestModel, self).__init__()
                        self.gru = nn.GRU(10, 10, batch_first=True)
                        self.action = torch.zeros(10, requires_grad=True)
                        self.value = torch.zeros(1, requires_grad=True)

                    def forward(self, inputs):
                        state = (torch.from_numpy((inputs[0])[0])).view(-1, 1)
                        rnn_state = (torch.from_numpy(inputs[1]))
                        reshaped = state.view(-1, 1, 10)
                        gru_output, rnn_final_state = self.gru(
                            reshaped, rnn_state.unsqueeze(0))
                        output = F.softmax(gru_output.view(-1, 10), dim=1)
                        value = torch.tensor([0.0])  # Create a trainable value
                        return output, value, rnn_final_state.squeeze(0)

                return TestModel()

        # We don't care about actually optimizing it, so just run a few rollouts to make
        # sure fit() doesn't crash, then check the behavior of the GRU state.

        env = TestEnvironment()
        a2c = A2C(env, TestPolicy())
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
    @pytest.mark.torch
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

                class TestModel(nn.Module):

                    def __init__(self):
                        super(TestModel, self).__init__()
                        self.dense1 = nn.Linear(4, 6)
                        self.dense2 = nn.Linear(6, 6)
                        self.output = nn.Linear(6, 4, bias=False)
                        self.value = nn.Linear(6, 1)
                        self.softmax = nn.Softmax(dim=1)

                    def forward(self, inputs):
                        # x = (torch.from_numpy(inputs)).view(1, -1)
                        x = torch.relu(
                            self.dense1(
                                torch.tensor(inputs[0], dtype=torch.float32)))
                        x = torch.relu(self.dense2(x))
                        output = self.softmax(self.output(x))
                        value = self.value(x)
                        return output, value

                return TestModel()

        # Optimize it.

        env = TestEnvironment()
        a2c = A2C(env,
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
