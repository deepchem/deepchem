import unittest
import deepchem as dc
import pytest
import numpy as np
from flaky import flaky
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from deepchem.rl.torch_rl import PPO
    from deepchem.models.optimizers import Adam
    has_pytorch = True
except:
    has_pytorch = False


class TestPPO(unittest.TestCase):

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
                        prob = F.softmax(torch.reshape(self.action,
                                                       (-1, env.n_actions)),
                                         dim=1)
                        return prob, self.value

                return TestModel()

        # Optimize it.

        ppo = PPO(env,
                  TestPolicy(),
                  max_rollout_length=20,
                  optimization_epochs=8,
                  optimizer=Adam(learning_rate=0.003))
        ppo.fit(100000)

        # It should have learned that the expected value is very close to zero, and that the best
        # action is to walk away.  (To keep the test fast, we allow that to be either of the two
        # top actions).

        action_prob, value = ppo.predict([[0]])
        assert -0.8 < value[0] < 0.5
        assert 37 in np.argsort(action_prob.flatten())[-2:]
        assert ppo.select_action([[0]],
                                 deterministic=True) == np.argmax(action_prob)

        # Verify that we can create a new PPO object, reload the parameters from the first one, and
        # get the same result.

        new_ppo = PPO(env, TestPolicy(), model_dir=ppo._model.model_dir)
        new_ppo.restore()
        action_prob2, value2 = new_ppo.predict([[0]])
        assert value2 == value

        # Do the same thing, only using the "restore" argument to fit()

        new_ppo = PPO(env, TestPolicy(), model_dir=ppo._model.model_dir)
        new_ppo.fit(0, restore=True)
        action_prob2, value2 = new_ppo.predict([[0]])
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
                        state = (torch.from_numpy((inputs[0])[0]))
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
        ppo = PPO(env, TestPolicy(), batch_size=0)
        ppo.fit(100)
        # On the first call, the initial state should be all zeros.
        prob1, value1 = ppo.predict(env.state,
                                    use_saved_states=True,
                                    save_states=False)
        # It should still be zeros since we didn't save it last time.
        prob2, value2 = ppo.predict(env.state,
                                    use_saved_states=True,
                                    save_states=True)
        # It should be different now.
        prob3, value3 = ppo.predict(env.state,
                                    use_saved_states=True,
                                    save_states=False)
        # This should be the same as the previous one.
        prob4, value4 = ppo.predict(env.state,
                                    use_saved_states=True,
                                    save_states=False)
        # Now we reset it, so we should get the same result as initially.
        prob5, value5 = ppo.predict(env.state,
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
        ppo = PPO(env,
                  TestPolicy(),
                  use_hindsight=True,
                  optimization_epochs=1,
                  batch_size=0,
                  optimizer=Adam(learning_rate=0.001))
        ppo.fit(1500000)

        # Try running it a few times and see if it succeeds.

        pass_count = 0
        for i in range(5):
            env.reset()
            while not env.terminated:
                env.step(ppo.select_action(env.state))
            if np.array_equal(env.state[:2], env.state[2:]):
                pass_count += 1
        assert pass_count >= 3
