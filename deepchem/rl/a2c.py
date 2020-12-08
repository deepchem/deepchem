"""Advantage Actor-Critic (A2C) algorithm for reinforcement learning."""
import time
try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection
import numpy as np
import tensorflow as tf

from deepchem.models import KerasModel
from deepchem.models.optimizers import Adam


class A2CLossDiscrete(object):
  """This class computes the loss function for A2C with discrete action spaces."""

  def __init__(self, value_weight, entropy_weight, action_prob_index,
               value_index):
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight
    self.action_prob_index = action_prob_index
    self.value_index = value_index

  def __call__(self, outputs, labels, weights):
    prob = outputs[self.action_prob_index]
    value = outputs[self.value_index]
    reward, advantage = weights
    action = labels[0]
    advantage = tf.expand_dims(advantage, axis=1)
    prob = prob + np.finfo(np.float32).eps
    log_prob = tf.math.log(prob)
    policy_loss = -tf.reduce_mean(
        advantage * tf.reduce_sum(action * log_prob, axis=1))
    value_loss = tf.reduce_mean(tf.square(reward - value))
    entropy = -tf.reduce_mean(tf.reduce_sum(prob * log_prob, axis=1))
    return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy


class A2CLossContinuous(object):
  """This class computes the loss function for A2C with continuous action spaces.

  Note
  ----
  This class requires tensorflow-probability to be installed.
  """

  def __init__(self, value_weight, entropy_weight, mean_index, std_index,
               value_index):
    try:
      import tensorflow_probability as tfp  # noqa: F401
    except ModuleNotFoundError:
      raise ValueError(
          "This class requires tensorflow-probability to be installed.")
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight
    self.mean_index = mean_index
    self.std_index = std_index
    self.value_index = value_index

  def __call__(self, outputs, labels, weights):
    import tensorflow_probability as tfp
    mean = outputs[self.mean_index]
    std = outputs[self.std_index]
    value = outputs[self.value_index]
    reward, advantage = weights
    action = labels[0]
    distrib = tfp.distributions.Normal(mean, std)
    reduce_axes = list(range(1, len(action.shape)))
    log_prob = tf.reduce_sum(distrib.log_prob(action), reduce_axes)
    policy_loss = -tf.reduce_mean(advantage * log_prob)
    value_loss = tf.reduce_mean(tf.square(reward - value))
    entropy = tf.reduce_mean(distrib.entropy())
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


  Note
  ----
  Using this class on continuous action spaces requires that `tensorflow_probability` be installed.
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
               use_hindsight=False):
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
    """
    self._env = env
    self._policy = policy
    self.max_rollout_length = max_rollout_length
    self.discount_factor = discount_factor
    self.advantage_lambda = advantage_lambda
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight
    self.use_hindsight = use_hindsight
    self._state_is_list = isinstance(env.state_shape[0], SequenceCollection)
    if optimizer is None:
      self._optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
    else:
      self._optimizer = optimizer
    output_names = policy.output_names
    self.continuous = ('action_mean' in output_names)
    self._value_index = output_names.index('value')
    if self.continuous:
      self._action_mean_index = output_names.index('action_mean')
      self._action_std_index = output_names.index('action_std')
    else:
      self._action_prob_index = output_names.index('action_prob')
    self._rnn_final_state_indices = [
        i for i, n in enumerate(output_names) if n == 'rnn_state'
    ]
    self._rnn_states = policy.rnn_initial_states
    self._model = self._build_model(model_dir)
    self._checkpoint = tf.train.Checkpoint()
    self._checkpoint.save_counter  # Ensure the variable has been created
    self._checkpoint.listed = self._model.model.trainable_variables

  def _build_model(self, model_dir):
    """Construct a KerasModel containing the policy and loss calculations."""
    policy_model = self._policy.create_model()
    if self.continuous:
      loss = A2CLossContinuous(self.value_weight, self.entropy_weight,
                               self._action_mean_index, self._action_std_index,
                               self._value_index)
    else:
      loss = A2CLossDiscrete(self.value_weight, self.entropy_weight,
                             self._action_prob_index, self._value_index)
    model = KerasModel(
        policy_model,
        loss,
        batch_size=self.max_rollout_length,
        model_dir=model_dir,
        optimize=self._optimizer)
    model._ensure_built()

    return model

  def fit(self,
          total_steps,
          max_checkpoints_to_keep=5,
          checkpoint_interval=600,
          restore=False):
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
    manager = tf.train.CheckpointManager(
        self._checkpoint, self._model.model_dir, max_checkpoints_to_keep)
    checkpoint_time = time.time()
    self._env.reset()
    rnn_states = self._policy.rnn_initial_states

    # Training loop.

    step_count = 0
    while step_count < total_steps:
      initial_rnn_states = rnn_states
      states, actions, rewards, values, rnn_states = self._create_rollout(
          rnn_states)
      self._process_rollout(states, actions, rewards, values,
                            initial_rnn_states)
      if self.use_hindsight:
        self._process_rollout_with_hindsight(states, actions,
                                             initial_rnn_states)
      step_count += len(actions)
      self._model._global_step.assign_add(len(actions))

      # Do checkpointing.

      if step_count >= total_steps or time.time(
      ) >= checkpoint_time + checkpoint_interval:
        manager.save()
        checkpoint_time = time.time()

  def predict(self, state, use_saved_states=True, save_states=True):
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
    results = self._predict_outputs(state, use_saved_states, save_states)
    if self.continuous:
      return [
          results[i] for i in (self._action_mean_index, self._action_std_index,
                               self._value_index)
      ]
    else:
      return [results[i] for i in (self._action_prob_index, self._value_index)]

  def select_action(self,
                    state,
                    deterministic=False,
                    use_saved_states=True,
                    save_states=True):
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

  def restore(self):
    """Reload the model parameters from the most recent checkpoint file."""
    last_checkpoint = tf.train.latest_checkpoint(self._model.model_dir)
    if last_checkpoint is None:
      raise ValueError('No checkpoint found')
    self._checkpoint.restore(last_checkpoint)

  def _predict_outputs(self, state, use_saved_states, save_states):
    """Compute a set of outputs for a state. """
    if not self._state_is_list:
      state = [state]
    if use_saved_states:
      state = state + list(self._rnn_states)
    else:
      state = state + list(self._policy.rnn_initial_states)
    inputs = [np.expand_dims(s, axis=0) for s in state]
    results = self._compute_model(inputs)
    results = [r.numpy() for r in results]
    if save_states:
      self._rnn_states = [
          np.squeeze(results[i], 0) for i in self._rnn_final_state_indices
      ]
    return results

  @tf.function(experimental_relax_shapes=True)
  def _compute_model(self, inputs):
    return self._model.model(inputs)

  def _select_action_from_outputs(self, outputs, deterministic):
    """Given the policy outputs, select an action to perform."""
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
    """Generate a rollout."""
    states = []
    actions = []
    rewards = []
    values = []

    # Generate the rollout.

    for i in range(self.max_rollout_length):
      if self._env.terminated:
        break
      state = self._env.state
      states.append(state)
      results = self._compute_model(
          self._create_model_inputs(state, rnn_states))
      results = [r.numpy() for r in results]
      value = results[self._value_index]
      rnn_states = [
          np.squeeze(results[i], 0) for i in self._rnn_final_state_indices
      ]
      action = self._select_action_from_outputs(results, False)
      actions.append(action)
      values.append(float(value))
      rewards.append(self._env.step(action))

    # Compute an estimate of the reward for the rest of the episode.

    if not self._env.terminated:
      results = self._compute_model(
          self._create_model_inputs(self._env.state, rnn_states))
      final_value = self.discount_factor * results[self._value_index].numpy()[0]
    else:
      final_value = 0.0
    values.append(final_value)
    if self._env.terminated:
      self._env.reset()
      rnn_states = self._policy.rnn_initial_states
    return states, actions, np.array(
        rewards, dtype=np.float32), np.array(
            values, dtype=np.float32), rnn_states

  def _process_rollout(self, states, actions, rewards, values,
                       initial_rnn_states):
    """Train the network based on a rollout."""

    # Compute the discounted rewards and advantages.

    discounted_rewards = rewards.copy()
    discounted_rewards[-1] += values[-1]
    advantages = rewards - values[:-1] + self.discount_factor * np.array(
        values[1:])
    for j in range(len(rewards) - 1, 0, -1):
      discounted_rewards[j - 1] += self.discount_factor * discounted_rewards[j]
      advantages[
          j - 1] += self.discount_factor * self.advantage_lambda * advantages[j]

    # Record the actions, converting to one-hot if necessary.

    actions_matrix = []
    if self.continuous:
      for action in actions:
        actions_matrix.append(action)
    else:
      n_actions = self._env.n_actions
      for action in actions:
        a = np.zeros(n_actions, np.float32)
        a[action] = 1.0
        actions_matrix.append(a)
    actions_matrix = np.array(actions_matrix, dtype=np.float32)

    # Rearrange the states into the proper set of arrays.

    if self._state_is_list:
      state_arrays = [[] for i in range(len(self._env.state_shape))]
      for state in states:
        for j in range(len(state)):
          state_arrays[j].append(state[j])
    else:
      state_arrays = [states]
    state_arrays = [np.stack(s) for s in state_arrays]

    # Build the inputs and apply gradients.

    inputs = state_arrays + [
        np.expand_dims(s, axis=0) for s in initial_rnn_states
    ]
    self._apply_gradients(inputs, actions_matrix, discounted_rewards,
                          advantages)

  @tf.function(experimental_relax_shapes=True)
  def _apply_gradients(self, inputs, actions_matrix, discounted_rewards,
                       advantages):
    """Compute the gradient of the loss function for a rollout and update the model."""
    vars = self._model.model.trainable_variables
    with tf.GradientTape() as tape:
      outputs = self._model.model(inputs)
      loss = self._model._loss_fn(outputs, [actions_matrix],
                                  [discounted_rewards, advantages])
    gradients = tape.gradient(loss, vars)
    self._model._tf_optimizer.apply_gradients(zip(gradients, vars))

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
    outputs = self._compute_model(inputs)
    values = outputs[self._value_index].numpy()
    values = np.append(values.flatten(), 0.0)
    self._process_rollout(hindsight_states, actions[:len(rewards)],
                          np.array(rewards, dtype=np.float32),
                          np.array(values, dtype=np.float32),
                          initial_rnn_states)

  def _create_model_inputs(self, state, rnn_states):
    """Create the inputs to the model for use during a rollout."""
    if not self._state_is_list:
      state = [state]
    state = state + rnn_states
    return [np.expand_dims(s, axis=0) for s in state]
