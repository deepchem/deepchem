"""Advantage Actor-Critic (A2C) algorithm for reinforcement learning using PyTorch."""
import time
import numpy as np
import torch
import torch.nn as nn
from collections.abc import Sequence as SequenceCollection
from torch.distributions import Normal, Categorical

from deepchem.models import TorchModel
from deepchem.models.optimizers import Adam, Optimizer


class A2CLossDiscrete(nn.Module):
    """Computes the loss function for A2C with discrete action spaces."""

    def __init__(self, value_weight, entropy_weight, action_prob_index, value_index):
        super(A2CLossDiscrete, self).__init__()
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.action_prob_index = action_prob_index
        self.value_index = value_index

    def forward(self, outputs, labels, weights):
        prob = outputs[self.action_prob_index]
        value = outputs[self.value_index]
        reward, advantage = weights
        action = labels[0]

        advantage = advantage.unsqueeze(1)
        prob = prob + torch.finfo(torch.float32).eps
        log_prob = torch.log(prob)
        policy_loss = -torch.mean(advantage * torch.sum(action * log_prob, dim=1))
        value_loss = torch.mean((reward - value).pow(2))
        entropy = -torch.mean(torch.sum(prob * log_prob, dim=1))

        return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy


class A2CLossContinuous(nn.Module):
    """Computes the loss function for A2C with continuous action spaces."""

    def __init__(self, value_weight, entropy_weight, mean_index, std_index, value_index):
        super(A2CLossContinuous, self).__init__()
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.mean_index = mean_index
        self.std_index = std_index
        self.value_index = value_index

    def forward(self, outputs, labels, weights):
        mean = outputs[self.mean_index]
        std = outputs[self.std_index]
        value = outputs[self.value_index]
        reward, advantage = weights
        action = labels[0]

        # Ensure std is positive
        std = torch.nn.functional.softplus(std)
        distrib = Normal(mean, std)
        log_prob = distrib.log_prob(action).sum(dim=1)
        policy_loss = -torch.mean(advantage * log_prob)
        value_loss = torch.mean((reward - value).pow(2))
        entropy = torch.mean(distrib.entropy())

        return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy


class A2C:
    """
    Implements the Advantage Actor-Critic (A2C) algorithm using PyTorch.
    """

    def __init__(self,
                 env,
                 policy,
                 max_rollout_length=20,
                 discount_factor=0.99,
                 advantage_lambda=0.98,
                 value_weight=1.0,
                 entropy_weight=0.01,
                 optimizer=None,
                 model_dir=None,
                 use_hindsight=False,
                 device='cpu'):
        self._env = env
        self._policy = policy.to(device)
        self.max_rollout_length = max_rollout_length
        self.discount_factor = discount_factor
        self.advantage_lambda = advantage_lambda
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.use_hindsight = use_hindsight
        self._state_is_list = isinstance(env.state_shape[0], SequenceCollection)
        self.device = device

        if optimizer is None:
            self._optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        else:
            self._optimizer = optimizer
        self._optimizer = self._optimizer._create_pytorch_optimizer(self._policy.parameters())

        output_names = policy.output_names
        self.continuous = ('action_mean' in output_names)
        self._value_index = output_names.index('value')
        if self.continuous:
            self._action_mean_index = output_names.index('action_mean')
            self._action_std_index = output_names.index('action_std')
            self.loss_fn = A2CLossContinuous(value_weight, entropy_weight,
                                             self._action_mean_index,
                                             self._action_std_index, self._value_index)
        else:
            self._action_prob_index = output_names.index('action_prob')
            self.loss_fn = A2CLossDiscrete(value_weight, entropy_weight,
                                            self._action_prob_index, self._value_index)

        self._rnn_final_state_indices = [
            i for i, n in enumerate(output_names) if n == 'rnn_state'
        ]
        self._rnn_states = policy.rnn_initial_states
        self.model = TorchModel(self._policy, loss=self.loss_fn, batch_size=max_rollout_length,
                                model_dir=model_dir, optimizer=self._optimizer)
        self._checkpoint = {
            'model': self._policy.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }

    def fit(self,
            total_steps,
            max_checkpoints_to_keep=5,
            checkpoint_interval=600,
            restore=False):
        if restore:
            self.restore()
        checkpoint_time = time.time()
        self._env.reset()
        rnn_states = self._policy.rnn_initial_states

        step_count = 0
        while step_count < total_steps:
            initial_rnn_states = rnn_states
            states, actions, rewards, values, rnn_states = self._create_rollout(rnn_states)
            self._process_rollout(states, actions, rewards, values, initial_rnn_states)
            if self.use_hindsight:
                self._process_rollout_with_hindsight(states, actions, initial_rnn_states)
            step_count += len(actions)
            self.model._global_step += len(actions)

            if step_count >= total_steps or time.time() >= checkpoint_time + checkpoint_interval:
                self._save_checkpoint()
                checkpoint_time = time.time()

    def predict(self, state, use_saved_states=True, save_states=True):
        self._policy.eval()
        with torch.no_grad():
            results = self._predict_outputs(state, use_saved_states, save_states)
            if self.continuous:
                return [results[i] for i in (self._action_mean_index,
                                             self._action_std_index, self._value_index)]
            else:
                return [results[i] for i in (self._action_prob_index, self._value_index)]

    def select_action(self, state, deterministic=False, use_saved_states=True, save_states=True):
        self._policy.eval()
        with torch.no_grad():
            outputs = self._predict_outputs(state, use_saved_states, save_states)
            return self._select_action_from_outputs(outputs, deterministic)

    def restore(self):
        checkpoint = torch.load(self.model.model_dir)
        self._policy.load_state_dict(checkpoint['model'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])

    def _predict_outputs(self, state, use_saved_states, save_states):
        if not self._state_is_list:
            state = [state]
        if use_saved_states:
            state = state + list(self._rnn_states)
        else:
            state = state + list(self._policy.rnn_initial_states)
        inputs = [torch.from_numpy(np.expand_dims(s, axis=0)).float().to(self.device) for s in state]
        results = self._policy(*inputs)
        results = [r.detach().cpu().numpy() for r in results]
        if save_states:
            self._rnn_states = [
                np.squeeze(results[i], 0) for i in self._rnn_final_state_indices
            ]
        return results

    def _select_action_from_outputs(self, outputs, deterministic):
        if self.continuous:
            action_mean = outputs[self._action_mean_index]
            action_std = outputs[self._action_std_index]
            if deterministic:
                return action_mean[0]
            else:
                return np.random.normal(action_mean[0], action_std[0])
        else:
            action_prob = outputs[self._action_prob_index]
            if deterministic:
                return action_prob.argmax()
            else:
                action_prob = action_prob.flatten()
                return np.random.choice(np.arange(len(action_prob)), p=action_prob)

    def _create_rollout(self, rnn_states):
        states = []
        actions = []
        rewards = []
        values = []

        for _ in range(self.max_rollout_length):
            if self._env.terminated:
                break
            state = self._env.state
            states.append(state)
            inputs = self._create_model_inputs(state, rnn_states)
            with torch.no_grad():
                results = self._policy(*inputs)
            value = results[self._value_index].cpu().numpy()[0]
            rnn_states = [r.detach().cpu().numpy() for r in results if results.index(r) in self._rnn_final_state_indices]
            action = self._select_action_from_outputs([r.cpu().numpy() for r in results], False)
            actions.append(action)
            values.append(float(value))
            rewards.append(self._env.step(action))

        if not self._env.terminated:
            inputs = self._create_model_inputs(self._env.state, rnn_states)
            with torch.no_grad():
                final_value = self.discount_factor * self._policy(*inputs)[self._value_index].cpu().numpy()[0]
        else:
            final_value = 0.0
        values.append(final_value)
        if self._env.terminated:
            self._env.reset()
            rnn_states = self._policy.rnn_initial_states
        return states, actions, np.array(rewards, dtype=np.float32), np.array(
            values, dtype=np.float32), rnn_states

    def _process_rollout(self, states, actions, rewards, values, initial_rnn_states):
        discounted_rewards = rewards.copy()
        discounted_rewards[-1] += values[-1]
        advantages = rewards - values[:-1] + self.discount_factor * np.array(values[1:])
        for j in range(len(rewards)-1, 0, -1):
            discounted_rewards[j-1] += self.discount_factor * discounted_rewards[j]
            advantages[j-1] += self.discount_factor * self.advantage_lambda * advantages[j]

        actions_matrix = []
        if self.continuous:
            actions_matrix = torch.tensor(actions, dtype=torch.float32).to(self.device)
        else:
            n_actions = self._env.n_actions
            actions_matrix = torch.zeros((len(actions), n_actions), dtype=torch.float32).to(self.device)
            for i, a in enumerate(actions):
                actions_matrix[i, a] = 1.0

        state_arrays = self._process_states(states)
        inputs = [torch.tensor(s).float().to(self.device) for s in state_arrays] + \
                 [torch.tensor(s).unsqueeze(0).float().to(self.device) for s in initial_rnn_states]

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        self._optimizer.zero_grad()
        outputs = self._policy(*inputs)
        loss = self.loss_fn(outputs, [actions_matrix], [discounted_rewards, advantages])
        loss.backward()
        self._optimizer.step()

    def _save_checkpoint(self):
        torch.save({
            'model': self._policy.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }, self.model.model_dir)

    def _process_rollout_with_hindsight(self, states, actions, initial_rnn_states):
        hindsight_states, rewards = self._env.apply_hindsight(states, actions, states[-1])
        state_arrays = self._process_states(hindsight_states)
        inputs = [torch.tensor(s).float().to(self.device) for s in state_arrays] + \
                 [torch.tensor(s).unsqueeze(0).float().to(self.device) for s in initial_rnn_states]
        with torch.no_grad():
            outputs = self._policy(*inputs)
        values = outputs[self._value_index].cpu().numpy().flatten()
        values = np.append(values, 0.0)
        self._process_rollout(hindsight_states, actions[:len(rewards)], np.array(rewards, dtype=np.float32),
                              np.array(values, dtype=np.float32), initial_rnn_states)

    def _create_model_inputs(self, state, rnn_states):
        if not self._state_is_list:
            state = [state]
        state_tensors = [torch.from_numpy(np.array(s)).float().to(self.device) for s in state]
        rnn_tensors = [torch.from_numpy(np.array(s)).float().to(self.device) for s in rnn_states]
        return state_tensors + rnn_tensors

    def _process_states(self, states):
        if self._state_is_list:
            state_arrays = [[] for _ in range(len(self._env.state_shape))]
            for state in states:
                for j in range(len(state)):
                    state_arrays[j].append(state[j])
        else:
            state_arrays = [states]
        return [np.stack(s) for s in state_arrays]