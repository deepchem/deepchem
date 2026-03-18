"""Advantage Actor-Critic (A2C) algorithm for reinforcement learning."""
import os
import time
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import Adam, Optimizer
from collections.abc import Sequence as SequenceCollection
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    from deepchem.rl import Environment, Policy
    import torch
    import torch.nn as nn
    has_pytorch = True
except:
    has_pytorch = False


class A2CLossDiscrete(object):
    """This class computes the loss function for A2C with discrete action spaces.
    The A2C algorithm optimizes all outputs at once using a loss that is the sum of three terms:

    1. The policy loss, which seeks to maximize the discounted reward for each action.

    2. The value loss, which tries to make the value estimate match the actual discounted reward that was attained at each step.

    3. An entropy term to encourage exploration.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> import torch
    >>> import torch.nn.functional as F
    >>> outputs = [torch.tensor([[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]]), torch.tensor([0.], requires_grad = True)]
    >>> labels = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype = np.float32)
    >>> discount = np.array([-1.0203744, -0.02058018, 0.98931295, 2.009407, 1.019603, 0.01980097, -0.9901, 0.01, -1. , 0. ], dtype=np.float32)
    >>> advantage = np.array([-1.0203744 ,-0.02058018, 0.98931295, 2.009407, 1.019603, 0.01980097, -0.9901 ,0.01 ,-1. , 0.], dtype = np.float32)
    >>> loss = A2CLossDiscrete(value_weight = 1.0, entropy_weight = 0.01, action_prob_index = 0, value_index = 1)
    >>> loss_val = loss(outputs, [labels], [discount, advantage])
    >>> loss_val
    tensor(1.2541, grad_fn=<SubBackward0>)
    """

    def __init__(self, value_weight: float, entropy_weight: float,
                 action_prob_index: int, value_index: int):
        """Computes the loss function for the A2C algorithm with discrete action spaces.

        Parameters
        ----------
        value_weight: float
            a scale factor for the value loss term in the loss function
        entropy_weight: float
            a scale factor for the entropy term in the loss function
        action_prob_index: int
            Index of the action probabilities in the model's outputs.
        value_index: int
            Index of the value estimate in the model's outputs.
        """
        self.value_weight: float = value_weight
        self.entropy_weight: float = entropy_weight
        self.action_prob_index: int = action_prob_index
        self.value_index: int = value_index

    def __call__(self, outputs: List[torch.Tensor], labels: List[np.ndarray],
                 weights: List[np.ndarray]):
        prob_array: torch.Tensor = outputs[self.action_prob_index]
        value: torch.Tensor = outputs[self.value_index]
        reward, advantage_array = weights
        action: torch.Tensor = torch.from_numpy(labels[0])
        advantage: torch.Tensor = torch.unsqueeze(
            torch.from_numpy(advantage_array), 1)
        prob: torch.Tensor = prob_array + torch.finfo(torch.float32).eps
        log_prob: torch.Tensor = torch.log(prob)
        policy_loss: torch.Tensor = -torch.mean(
            advantage * torch.sum(action * log_prob, 1))
        value_loss: torch.Tensor = torch.mean(
            torch.square(torch.from_numpy(reward) - value))
        entropy: torch.Tensor = -torch.mean(torch.sum(prob * log_prob, 1))
        return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy


# Note: For continuous case, when an additional loss term correspoding to mean is calculated only then the gradients get calculated for the layers of the model.
class A2CLossContinuous(object):
    """This class computes the loss function for A2C with continuous action spaces.

    Example
    -------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> import torch
    >>> import torch.nn.functional as F
    >>> outputs = [torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]], dtype=torch.float32, requires_grad=True), torch.tensor([10.], dtype=torch.float32, requires_grad=True), torch.tensor([[27.717865],[28.860144]], dtype=torch.float32, requires_grad=True)]
    >>> labels = np.array([[-4.897339 ], [ 3.4308329], [-4.527725 ], [-7.3000813], [-1.9869075], [20.664988 ], [-8.448957 ], [10.580486 ], [10.017258 ], [17.884674 ]], dtype=np.float32)
    >>> discount = np.array([4.897339, -8.328172, 7.958559, 2.772356, -5.313174, -22.651896, 29.113945, -19.029444, 0.56322646, -7.867417], dtype=np.float32)
    >>> advantage = np.array([-5.681633, -20.57494, -1.4520378, -9.348538, -18.378199, -33.907513, 25.572464, -32.485718 , -6.412546, -15.034998], dtype=np.float32)
    >>> loss = A2CLossContinuous(value_weight = 1.0, entropy_weight = 0.01, mean_index = 0, std_index = 1, value_index = 2)
    >>> loss_val = loss(outputs, [labels], [discount, advantage])
    >>> loss_val
    tensor(1050.2310, grad_fn=<SubBackward0>)
    """

    def __init__(self, value_weight: float, entropy_weight: float,
                 mean_index: int, std_index: int, value_index: int):
        """Computes the loss function for the A2C algorithm with continuous action spaces.

        Parameters
        ----------
        value_weight: float
            a scale factor for the value loss term in the loss function
        entropy_weight: float
            a scale factor for the entropy term in the loss function
        mean_index: int
            Index of the mean of the action distribution in the model's outputs.
        std_index : int
            Index of the standard deviation of the action distribution in the model's outputs.
        value_index: int
            Index of the value estimate in the model's outputs.
        """
        self.value_weight: float = value_weight
        self.entropy_weight: float = entropy_weight
        self.mean_index: int = mean_index
        self.std_index: int = std_index
        self.value_index: int = value_index

    def __call__(self, outputs: List[torch.Tensor], labels: List[np.ndarray],
                 weights: List[np.ndarray]):
        import torch.distributions as dist
        mean: torch.Tensor = outputs[self.mean_index]
        std: torch.Tensor = outputs[self.std_index]
        value: torch.Tensor = outputs[self.value_index]
        reward, advantage = weights
        action: torch.Tensor = torch.from_numpy(labels[0])
        distrib = dist.Normal(torch.tensor(mean), torch.tensor(std))
        reduce_axes: List[int] = list(range(1, len(action.shape)))
        log_prob: torch.Tensor = torch.sum(distrib.log_prob(action),
                                           reduce_axes)
        policy_loss: torch.Tensor = -torch.mean(
            torch.from_numpy(advantage) * log_prob)
        value_loss: torch.Tensor = torch.mean(
            torch.square(torch.from_numpy(reward) - value))
        entropy: torch.Tensor = torch.mean(distrib.entropy())
        return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy


class A2C(object):
    """
    Implements the Advantage Actor-Critic (A2C) algorithm for reinforcement learning.
    The algorithm is described in Mnih et al, "Asynchronous Methods for Deep Reinforcement Learning"
    (https://arxiv.org/abs/1602.01783).  This class supports environments with both discrete and
    continuous action spaces.  For discrete action spaces, the "action" argument passed to the
    environment is an integer giving the index of the action to perform.  The policy must output
    a vector called "action_prob" giving the probability of taking each action.  For continuous
    action spaces, the action is an array where each element is chosen independently from a
    normal distribution.  The policy must output two arrays of the same shape: "action_mean"
    gives the mean value for each element, and "action_std" gives the standard deviation for
    each element.  In either case, the policy must also output a scalar called "value" which
    is an estimate of the value function for the current state.
    The algorithm optimizes all outputs at once using a loss that is the sum of three terms:

    1. The policy loss, which seeks to maximize the discounted reward for each action.

    2. The value loss, which tries to make the value estimate match the actual discounted reward
        that was attained at each step.

    3. An entropy term to encourage exploration.

    This class supports Generalized Advantage Estimation as described in Schulman et al., "High-Dimensional
    Continuous Control Using Generalized Advantage Estimation" (https://arxiv.org/abs/1506.02438).
    This is a method of trading off bias and variance in the advantage estimate, which can sometimes
    improve the rate of convergance.  Use the advantage_lambda parameter to adjust the tradeoff.
    This class supports Hindsight Experience Replay as described in Andrychowicz et al., "Hindsight
    Experience Replay" (https://arxiv.org/abs/1707.01495).  This is a method that can enormously
    accelerate learning when rewards are very rare.  It requires that the environment state contains
    information about the goal the agent is trying to achieve.  Each time it generates a rollout, it
    processes that rollout twice: once using the actual goal the agent was pursuing while generating
    it, and again using the final state of that rollout as the goal.  This guarantees that half of
    all rollouts processed will be ones that achieved their goals, and hence received a reward.
    To use this feature, specify use_hindsight=True to the constructor.  The environment must have
    a method defined as follows:

    def apply_hindsight(self, states, actions, goal):
        ...
        return new_states, rewards
    The method receives the list of states generated during the rollout, the action taken for each one,
    and a new goal state.  It should generate a new list of states that are identical to the input ones,
    except specifying the new goal.  It should return that list of states, and the rewards that would
    have been received for taking the specified actions from those states.  The output arrays may be
    shorter than the input ones, if the modified rollout would have terminated sooner.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> import torch
    >>> import torch.nn.functional as F
    >>> from deepchem.rl.torch_rl import A2C
    >>> class RouletteEnvironment(dc.rl.Environment):
    ...     def __init__(self):
    ...         super(RouletteEnvironment, self).__init__([(1,)], 38)
    ...         self._state = [np.array([0])]
    ...     def step(self, action):
    ...         if action == 37:
    ...             self._terminated = True  # Walk away.
    ...             return 0.0
    ...         wheel = np.random.randint(37)
    ...         if wheel == 0:
    ...             if action == 0:
    ...                 return 35.0
    ...             return -1.0
    ...         if action != 0 and wheel % 2 == action % 2:
    ...             return 1.0
    ...         return -1.0
    ...     def reset(self):
    ...         self._terminated = False
    >>> class TestPolicy(dc.rl.Policy):
    ...     def __init__(self, env):
    ...         super(TestPolicy, self).__init__(['action_prob', 'value'])
    ...         self.env = env
    ...     def create_model(self, **kwargs):
    ...         env = self.env
    ...         class TestModel(nn.Module):
    ...             def __init__(self):
    ...                 super(TestModel, self).__init__()
    ...                 self.action = nn.Parameter(torch.ones(env.n_actions, dtype=torch.float32))
    ...                 self.value = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    ...             def forward(self, inputs):
    ...                 prob = F.softmax(torch.reshape(self.action, (-1, env.n_actions)))
    ...                 return prob, self.value
    ...         return TestModel()
    >>> env = RouletteEnvironment()
    >>> policy = TestPolicy(env)
    >>> a2c = A2C(env, policy, max_rollout_length=20, optimizer=Adam(learning_rate=0.001))
    >>> a2c.fit(1000)
    >>> action_prob, value = a2c.predict([[0]])
    """

    def __init__(self,
                 env: Environment,
                 policy: Policy,
                 max_rollout_length: int = 20,
                 discount_factor: float = 0.99,
                 advantage_lambda: float = 0.98,
                 value_weight: float = 1.0,
                 entropy_weight: float = 0.01,
                 optimizer: Optional[Optimizer] = None,
                 model_dir: Optional[str] = None,
                 use_hindsight: bool = False,
                 device: Optional[torch.device] = None) -> None:
        """Create an object for optimizing a policy.

        Parameters
        ----------
        env: Environment
            the Environment to interact with
        policy: Policy
            the Policy to optimize.  It must have outputs with the names 'action_prob'
            and 'value' (for discrete action spaces) or 'action_mean', 'action_std',
            and 'value' (for continuous action spaces)
        max_rollout_length: int
            the maximum length of rollouts to generate
        discount_factor: float
            the discount factor to use when computing rewards
        advantage_lambda: float
            the parameter for trading bias vs. variance in Generalized Advantage Estimation
        value_weight: float
            a scale factor for the value loss term in the loss function
        entropy_weight: float
            a scale factor for the entropy term in the loss function
        optimizer: Optimizer
            the optimizer to use.  If None, a default optimizer is used.
        model_dir: str
            the directory in which the model will be saved.  If None, a temporary directory will be created.
        use_hindsight: bool
            if True, use Hindsight Experience Replay
        device: torch.device, optional (default None)
            the device on which to run computations.  If None, a device is
            chosen automatically.
        """
        self._env: Environment = env
        self._policy: Policy = policy
        self.max_rollout_length: int = max_rollout_length
        self.discount_factor: float = discount_factor
        self.advantage_lambda: float = advantage_lambda
        self.value_weight: float = value_weight
        self.entropy_weight: float = entropy_weight
        self.use_hindsight: bool = use_hindsight
        self._state_is_list: bool = isinstance(env.state_shape[0],
                                               SequenceCollection)
        if optimizer is None:
            self._optimizer: Optimizer = Adam(learning_rate=0.001,
                                              beta1=0.9,
                                              beta2=0.999)
        else:
            self._optimizer = optimizer
        output_names: List[str] = policy.output_names
        self.continuous: bool = ('action_mean' in output_names)
        self._value_index: int = output_names.index('value')
        if self.continuous:
            self._action_mean_index: int = output_names.index('action_mean')
            self._action_std_index: int = output_names.index('action_std')
        else:
            self._action_prob_index: int = output_names.index('action_prob')
        self._rnn_final_state_indices: List[int] = [
            i for i, n in enumerate(output_names) if n == 'rnn_state'
        ]
        self._rnn_states: List[np.ndarray] = policy.rnn_initial_states
        self._model: TorchModel = self._build_model(model_dir)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device: torch.device = device
        self._model.model = self._model.model.to(device)

    def _build_model(self, model_dir: Optional[str]) -> TorchModel:
        """Construct a TorchModel containing the policy and loss calculations."""
        policy_model: nn.Module = self._policy.create_model()
        if self.continuous:
            loss: Union[A2CLossContinuous, A2CLossDiscrete] = A2CLossContinuous(
                self.value_weight, self.entropy_weight, self._action_mean_index,
                self._action_std_index, self._value_index)
        else:
            loss = A2CLossDiscrete(self.value_weight, self.entropy_weight,
                                   self._action_prob_index, self._value_index)
        model: TorchModel = TorchModel(policy_model,
                                       loss,
                                       batch_size=self.max_rollout_length,
                                       model_dir=model_dir,
                                       optimize=self._optimizer)

        model._ensure_built()

        return model

    def fit(self,
            total_steps: int,
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 600,
            restore: bool = False) -> None:
        """Train the policy.

        Parameters
        ----------
        total_steps: int
            the total number of time steps to perform on the environment, across all rollouts
            on all threads
        max_checkpoints_to_keep: int
            the maximum number of checkpoint files to keep.  When this number is reached, older
            files are deleted.
        checkpoint_interval: float
            the time interval at which to save checkpoints, measured in seconds
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        """
        if restore:
            self.restore()
        self._model.model.train()
        checkpoint_time: float = time.time()
        self._env.reset()
        rnn_states: List[np.ndarray] = self._policy.rnn_initial_states

        # Training loop.

        step_count: int = 0
        while step_count < total_steps:
            initial_rnn_states: List[np.ndarray] = rnn_states
            states, actions, rewards, values, rnn_states = self._create_rollout(
                rnn_states)
            self._process_rollout(states, actions, rewards, values,
                                  initial_rnn_states)
            if self.use_hindsight:
                self._process_rollout_with_hindsight(states, actions,
                                                     initial_rnn_states)
            step_count += len(actions)
            self._model._global_step += len(actions)

            # Do checkpointing.

            if step_count >= total_steps or time.time(
            ) >= checkpoint_time + checkpoint_interval:
                self.save_checkpoint(max_checkpoints_to_keep)
                checkpoint_time = time.time()

    def predict(self,
                state: np.ndarray,
                use_saved_states: bool = True,
                save_states: bool = True) -> List[np.ndarray]:
        """Compute the policy's output predictions for a state.
        If the policy involves recurrent layers, this method can preserve their internal
        states between calls.  Use the use_saved_states and save_states arguments to specify
        how it should behave.

        Parameters
        ----------
        state: array or list of arrays
            the state of the environment for which to generate predictions
        use_saved_states: bool
            if True, the states most recently saved by a previous call to predict() or select_action()
            will be used as the initial states.  If False, the internal states of all recurrent layers
            will be set to the initial values defined by the policy before computing the predictions.
        save_states: bool
            if True, the internal states of all recurrent layers at the end of the calculation
            will be saved, and any previously saved states will be discarded.  If False, the
            states at the end of the calculation will be discarded, and any previously saved
            states will be kept.

        Returns
        -------
        the array of action probabilities, and the estimated value function
        """
        results: List[np.ndarray] = self._predict_outputs(
            state, use_saved_states, save_states)
        if self.continuous:
            return [
                results[i] for i in (self._action_mean_index,
                                     self._action_std_index, self._value_index)
            ]
        else:
            return [
                results[i] for i in (self._action_prob_index, self._value_index)
            ]

    def select_action(self,
                      state: List[np.ndarray],
                      deterministic: bool = False,
                      use_saved_states: bool = True,
                      save_states: bool = True) -> int:
        """Select an action to perform based on the environment's state.
        If the policy involves recurrent layers, this method can preserve their internal
        states between calls.  Use the use_saved_states and save_states arguments to specify
        how it should behave.

        Parameters
        ----------
        state: array or list of arrays
            the state of the environment for which to select an action
        deterministic: bool
            if True, always return the best action (that is, the one with highest probability).
            If False, randomly select an action based on the computed probabilities.
        use_saved_states: bool
            if True, the states most recently saved by a previous call to predict() or select_action()
            will be used as the initial states.  If False, the internal states of all recurrent layers
            will be set to the initial values defined by the policy before computing the predictions.
        save_states: bool
            if True, the internal states of all recurrent layers at the end of the calculation
            will be saved, and any previously saved states will be discarded.  If False, the
            states at the end of the calculation will be discarded, and any previously saved
            states will be kept.

        Returns
        -------
        the index of the selected action
        """
        outputs = self._predict_outputs(state, use_saved_states, save_states)
        return self._select_action_from_outputs(outputs, deterministic)

    def restore(self, strict: Optional[bool] = True) -> None:
        """Reload the model parameters from the most recent checkpoint file."""
        last_checkpoint: Union[List[str], str] = sorted(
            self.get_checkpoints(self._model.model_dir))
        if last_checkpoint is None:
            raise ValueError('No checkpoint found')
        last_checkpoint = last_checkpoint[0]
        data: Any = torch.load(last_checkpoint, map_location=self.device)
        self._model.model.load_state_dict(data['model_state_dict'],
                                          strict=strict)
        self._model._pytorch_optimizer.load_state_dict(
            data['optimizer_state_dict'])
        self._model._global_step = data['global_step']

    def _predict_outputs(self, state, use_saved_states: bool,
                         save_states: bool) -> List[np.ndarray]:
        """Compute a set of outputs for a state. """
        if not self._state_is_list:
            state = [state]
        if use_saved_states:
            state = state + list(self._rnn_states)
        else:
            state = state + list(self._policy.rnn_initial_states)
        inputs: List[np.ndarray] = [np.expand_dims(s, axis=0) for s in state]
        results = self._model.model(inputs)
        results = [r.detach().numpy() for r in results]
        if save_states:
            self._rnn_states = [
                np.squeeze(results[i], 0) for i in self._rnn_final_state_indices
            ]
        return results

    def _select_action_from_outputs(self, outputs: List[np.ndarray],
                                    deterministic: bool) -> int:
        """Given the policy outputs, select an action to perform."""
        if self.continuous:
            action_mean: np.ndarray = outputs[self._action_mean_index]
            action_std: np.ndarray = outputs[self._action_std_index]
            if deterministic:
                return action_mean[0]
            else:
                return np.random.normal(action_mean[0], action_std[0])
        else:
            action_prob: np.ndarray = outputs[self._action_prob_index]
            if deterministic:
                return int(action_prob.argmax())
            else:
                action_prob = action_prob.flatten()
                return np.random.choice(np.arange(len(action_prob)),
                                        p=action_prob)

    def _create_rollout(
        self, rnn_states: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[int], np.ndarray, np.ndarray,
               List[np.ndarray]]:
        """Generate a rollout."""
        states: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[np.float32] = []
        values: List[float] = []

        # Generate the rollout.

        for i in range(self.max_rollout_length):
            if self._env.terminated:
                break
            state: np.ndarray = self._env.state
            states.append(state)
            results = self._model.model(
                self._create_model_inputs(state, rnn_states))
            results = [r.detach().numpy() for r in results]
            value: np.ndarray = results[self._value_index]
            rnn_states = [
                np.squeeze(results[i], 0) for i in self._rnn_final_state_indices
            ]
            action: int = self._select_action_from_outputs(results, False)
            actions.append(action)
            values.append(float(value))
            rewards.append(self._env.step(action))

        # Compute an estimate of the reward for the rest of the episode.

        if not self._env.terminated:
            results = self._model.model(
                self._create_model_inputs(self._env.state, rnn_states))
            final_value = self.discount_factor * results[
                self._value_index].detach().numpy()[0]
            final_value = final_value.item()
        else:
            final_value = 0.0
        values.append(final_value)
        if self._env.terminated:
            self._env.reset()
            rnn_states = self._policy.rnn_initial_states
        return states, actions, np.array(rewards, dtype=np.float32), np.array(
            values, dtype=np.float32), rnn_states

    def _process_rollout(self, states: List[np.ndarray], actions: List[int],
                         rewards: np.ndarray, values: np.ndarray,
                         initial_rnn_states: List[np.ndarray]) -> None:
        """Train the network based on a rollout."""

        # Compute the discounted rewards and advantages.

        discounted_rewards: np.ndarray = rewards.copy()
        discounted_rewards[-1] += values[-1]
        advantages: np.ndarray = rewards - values[:
                                                  -1] + self.discount_factor * np.array(
                                                      values[1:])
        for j in range(len(rewards) - 1, 0, -1):
            discounted_rewards[
                j - 1] += self.discount_factor * discounted_rewards[j]
            advantages[
                j -
                1] += self.discount_factor * self.advantage_lambda * advantages[
                    j]

        # Record the actions, converting to one-hot if necessary.

        actions_matrix: List[Union[np.ndarray, int]] = []
        if self.continuous:
            for action in actions:
                actions_matrix.append(action)
        else:
            n_actions = self._env.n_actions
            for action in actions:
                a = np.zeros(n_actions, np.float32)
                a[action] = 1.0
                actions_matrix.append(a)
        actions_matrix_array = np.array(actions_matrix, dtype=np.float32)

        # Rearrange the states into the proper set of arrays.

        if self._state_is_list:
            state_arrays: List[List[np.ndarray]] = [
                [] for i in range(len(self._env.state_shape))
            ]
            for state in states:
                for j in range(len(state)):
                    state_arrays[j].append(state[j])
        else:
            state_arrays = [states]
        state_arrays = [np.stack(s) for s in state_arrays]

        # Build the inputs and apply gradients.

        inputs: List[Union[np.ndarray, List[np.ndarray]]] = state_arrays + [
            np.expand_dims(s, axis=0) for s in initial_rnn_states
        ]
        self._apply_gradients(inputs, actions_matrix_array, discounted_rewards,
                              advantages)

    def _apply_gradients(self, inputs, actions_matrix, discounted_rewards,
                         advantages):
        """Compute the gradient of the loss function for a rollout and update the model."""
        self._model._pytorch_optimizer.zero_grad()
        outputs = self._model.model(inputs)
        loss = self._model._loss_fn(outputs, [actions_matrix],
                                    [discounted_rewards, advantages])
        loss.backward()
        self._model._pytorch_optimizer.step()
        if self._model._lr_schedule is not None:
            self._model._lr_schedule.step()

    def _process_rollout_with_hindsight(self, states, actions,
                                        initial_rnn_states):
        """Create a new rollout by applying hindsight to an existing one, then train the network."""
        hindsight_states, rewards = self._env.apply_hindsight(
            states, actions, states[-1])
        if self._state_is_list:
            state_arrays = [[] for i in range(len(self._env.state_shape))]
            for state in hindsight_states:
                for j in range(len(state)):
                    state_arrays[j].append(state[j])
        else:
            state_arrays = [hindsight_states]
        state_arrays = [np.stack(s) for s in state_arrays]
        inputs = state_arrays + [
            np.expand_dims(s, axis=0) for s in initial_rnn_states
        ]
        outputs = self._model.model(inputs)
        values = outputs[self._value_index].detach().numpy()
        values = np.append(values.flatten(), 0.0)
        self._process_rollout(hindsight_states, actions[:len(rewards)],
                              np.array(rewards, dtype=np.float32),
                              np.array(values, dtype=np.float32),
                              initial_rnn_states)

    def _create_model_inputs(self, state,
                             rnn_states: List[np.ndarray]) -> List[np.ndarray]:
        """Create the inputs to the model for use during a rollout."""
        if not self._state_is_list:
            state = [state]
        state = state + rnn_states
        return [np.expand_dims(s, axis=0) for s in state]

    def save_checkpoint(self,
                        max_checkpoints_to_keep: int = 5,
                        model_dir: Optional[str] = None) -> None:
        """Save a checkpoint to disk.
        Usually you do not need to call this method, since fit() saves checkpoints
        automatically.  If you have disabled automatic checkpointing during fitting,
        this can be called to manually write checkpoints.

        Parameters
        ----------
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        model_dir: str, default None
            Model directory to save checkpoint to. If None, revert to self.model_dir
        """
        if model_dir is None:
            model_dir = self._model.model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the checkpoint to a file.

        data: Dict[str, Any] = {
            'model_state_dict': self._model.model.state_dict(),
            'optimizer_state_dict': self._model._pytorch_optimizer.state_dict(),
            'global_step': self._model._global_step
        }
        temp_file: str = os.path.join(model_dir, 'temp_checkpoint.pt')
        torch.save(data, temp_file)

        # Rename and delete older files.

        paths: List[str] = [
            os.path.join(model_dir, 'checkpoint%d.pt' % (i + 1))
            for i in range(max_checkpoints_to_keep)
        ]
        if os.path.exists(paths[-1]):
            os.remove(paths[-1])
        for i in reversed(range(max_checkpoints_to_keep - 1)):
            if os.path.exists(paths[i]):
                os.rename(paths[i], paths[i + 1])
        os.rename(temp_file, paths[0])

    def get_checkpoints(self, model_dir: Optional[str] = None) -> List[str]:
        """Get a list of all available checkpoint files.

        Parameters
        ----------
        model_dir: str, default None
            Directory to get list of checkpoints from. Reverts to self.model_dir if None
        """
        if model_dir is None:
            model_dir = self._model.model_dir
        files: List[str] = sorted(os.listdir(model_dir))
        files = [
            f for f in files if f.startswith('checkpoint') and f.endswith('.pt')
        ]
        return [os.path.join(model_dir, f) for f in files]
