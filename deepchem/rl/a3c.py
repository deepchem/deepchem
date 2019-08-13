"""Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning."""

from deepchem.models import KerasModel
from deepchem.models.optimizers import Adam
import numpy as np
import tensorflow as tf
import collections
import copy
import multiprocessing
import os
import re
import threading


class A3CLossDiscrete(object):
  """This class computes the loss function for A3C with discrete action spaces."""

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
    log_prob = tf.log(prob)
    policy_loss = -tf.reduce_mean(
        advantage * tf.reduce_sum(action * log_prob, axis=1))
    value_loss = tf.reduce_mean(tf.square(reward - value))
    entropy = -tf.reduce_mean(tf.reduce_sum(prob * log_prob, axis=1))
    return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy


class A3CLossContinuous(object):
  """This class computes the loss function for A3C with continuous action spaces."""

  def __init__(self, value_weight, entropy_weight, mean_index, std_index,
               value_index):
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight
    self.mean_index = mean_index
    self.std_index = std_index
    self.value_index = value_index

  def __call__(self, outputs, labels, weights):
    mean = outputs[self.mean_index]
    std = outputs[self.std_index]
    value = outputs[self.value_index]
    reward, advantage = weights
    action = labels[0]
    distrib = tf.distributions.Normal(mean, std)
    reduce_axes = list(range(1, len(action.shape)))
    log_prob = tf.reduce_sum(distrib.log_prob(action), reduce_axes)
    policy_loss = -tf.reduce_mean(advantage * log_prob)
    value_loss = tf.reduce_mean(tf.square(reward - value))
    entropy = tf.reduce_mean(distrib.entropy())
    return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy


class A3C(object):
  """
  Implements the Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning.

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
  have been received for taking the specified actions from those states.
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
    self._state_is_list = isinstance(env.state_shape[0], collections.Sequence)
    if optimizer is None:
      self._optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
    else:
      self._optimizer = optimizer
    self._model = self._build_model(model_dir)
    output_names = policy.output_names
    output_tensors = self._model._output_tensors
    self._value = output_tensors[output_names.index('value')]
    if self.continuous:
      self._action_mean = output_tensors[output_names.index('action_mean')]
      self._action_std = output_tensors[output_names.index('action_std')]
    else:
      self._action_prob = output_tensors[output_names.index('action_prob')]
    rnn_outputs = [i for i, n in enumerate(output_names) if n == 'rnn_state']
    self._rnn_final_states = [output_tensors[i] for i in rnn_outputs]
    self._session = self._model.session
    self._rnn_states = policy.rnn_initial_states
    self._checkpoint = tf.train.Checkpoint()
    self._checkpoint.save_counter  # Ensure the variable has been created
    self._checkpoint.listed = self._model.model.trainable_variables
    self._session.run(self._checkpoint.save_counter.initializer)

  def _build_model(self, model_dir):
    """Construct a KerasModel containing the policy and loss calculations."""
    state_shape = self._env.state_shape
    state_dtype = self._env.state_dtype
    if not self._state_is_list:
      state_shape = [state_shape]
      state_dtype = [state_dtype]
    features = []
    for s, d in zip(state_shape, state_dtype):
      features.append(
          tf.keras.layers.Input(shape=list(s), dtype=tf.as_dtype(d)))
    policy_model = self._policy.create_model()
    output_names = self._policy.output_names
    if 'action_prob' in output_names:
      self.continuous = False
      loss = A3CLossDiscrete(self.value_weight, self.entropy_weight,
                             output_names.index('action_prob'),
                             output_names.index('value'))
    else:
      self.continuous = True
      loss = A3CLossContinuous(self.value_weight, self.entropy_weight,
                               output_names.index('action_mean'),
                               output_names.index('action_std'),
                               output_names.index('value'))
    model = KerasModel(
        policy_model,
        loss,
        batch_size=self.max_rollout_length,
        model_dir=model_dir,
        optimize=self._optimizer)
    env = self._env
    example_inputs = [
        np.zeros([model.batch_size] + list(shape), dtype)
        for shape, dtype in zip(state_shape, state_dtype)
    ]
    if self.continuous:
      example_labels = [np.zeros([model.batch_size] + list(env.action_shape))]
    else:
      example_labels = [np.zeros((model.batch_size, env.n_actions))]
    example_weights = [np.zeros(model.batch_size)] * 2
    model._create_training_ops((example_inputs, example_labels,
                                example_weights))
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
    step_count = [0]
    workers = []
    threads = []
    for i in range(multiprocessing.cpu_count()):
      workers.append(_Worker(self, i))
    self._session.run(tf.global_variables_initializer())
    if restore:
      self.restore()
    for worker in workers:
      thread = threading.Thread(
          name=worker.scope, target=lambda: worker.run(step_count, total_steps))
      threads.append(thread)
      thread.start()
    manager = tf.train.CheckpointManager(
        self._checkpoint, self._model.model_dir, max_checkpoints_to_keep)
    while True:
      threads = [t for t in threads if t.isAlive()]
      if len(threads) > 0:
        threads[0].join(checkpoint_interval)
      with self._session.as_default():
        manager.save()
      if len(threads) == 0:
        break

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
    if self.continuous:
      outputs = [self._action_mean, self._action_std, self._value]
    else:
      outputs = [self._action_prob, self._value]
    return self._predict_outputs(outputs, state, use_saved_states, save_states)

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
    if self.continuous:
      tensors = [self._action_mean, self._action_std]
    else:
      tensors = [self._action_prob]
    outputs = self._predict_outputs(tensors, state, use_saved_states,
                                    save_states)
    return self._select_action_from_outputs(outputs, deterministic)

  def restore(self):
    """Reload the model parameters from the most recent checkpoint file."""
    last_checkpoint = tf.train.latest_checkpoint(self._model.model_dir)
    if last_checkpoint is None:
      raise ValueError('No checkpoint found')
    self._checkpoint.restore(last_checkpoint).run_restore_ops(self._session)

  def _predict_outputs(self, outputs, state, use_saved_states, save_states):
    """Compute a set of outputs for a state. """
    if not self._state_is_list:
      state = [state]
    if use_saved_states:
      state = state + list(self._rnn_states)
    else:
      state = state + list(self._policy.rnn_initial_states)
    feed_dict = dict((f, np.expand_dims(s, axis=0))
                     for f, s in zip(self._model._input_placeholders, state))
    tensors = outputs
    if save_states:
      tensors = tensors + self._rnn_final_states
    results = self._session.run(tensors, feed_dict=feed_dict)
    if save_states:
      self._rnn_states = [np.squeeze(r, 0) for r in results[len(outputs):]]
    return results[:len(outputs)]

  def _select_action_from_outputs(self, outputs, deterministic):
    """Given the policy outputs, select an action to perform."""
    if self.continuous:
      action_mean, action_std = outputs
      if deterministic:
        return action_mean[0]
      else:
        return np.random.normal(action_mean[0], action_std[0])
    else:
      action_prob = outputs[0]
      if deterministic:
        return action_prob.argmax()
      else:
        action_prob = action_prob.flatten()
        return np.random.choice(np.arange(len(action_prob)), p=action_prob)


class _Worker(object):
  """A Worker object is created for each training thread."""

  def __init__(self, a3c, index):
    self.a3c = a3c
    self.index = index
    self.scope = 'worker%d' % index
    self.env = copy.deepcopy(a3c._env)
    self.env.reset()
    self.model = a3c._build_model(None)
    output_names = a3c._policy.output_names
    output_tensors = self.model._output_tensors
    self.value = output_tensors[output_names.index('value')]
    if a3c.continuous:
      self.action_mean = output_tensors[output_names.index('action_mean')]
      self.action_std = output_tensors[output_names.index('action_std')]
    else:
      self.action_prob = output_tensors[output_names.index('action_prob')]
    rnn_outputs = [i for i, n in enumerate(output_names) if n == 'rnn_state']
    self.rnn_final_states = [output_tensors[i] for i in rnn_outputs]
    self.rnn_states = a3c._policy.rnn_initial_states
    local_vars = self.model.model.trainable_variables
    global_vars = a3c._model.model.trainable_variables
    gradients = tf.gradients(self.model._loss_tensor, local_vars)
    grads_and_vars = list(zip(gradients, global_vars))
    self.train_op = a3c._model._tf_optimizer.apply_gradients(grads_and_vars)
    self.update_local_variables = tf.group(
        *[tf.assign(v1, v2) for v1, v2 in zip(local_vars, global_vars)])
    self.global_step = self.model.get_global_step()

  def run(self, step_count, total_steps):
    while step_count[0] < total_steps:
      self.a3c._session.run(self.update_local_variables)
      initial_rnn_states = self.rnn_states
      states, actions, rewards, values = self.create_rollout()
      self.process_rollout(states, actions, rewards, values, initial_rnn_states,
                           step_count[0])
      if self.a3c.use_hindsight:
        self.process_rollout_with_hindsight(states, actions, initial_rnn_states,
                                            step_count[0])
      step_count[0] += len(actions)

  def create_rollout(self):
    """Generate a rollout."""
    n_actions = self.env.n_actions
    session = self.a3c._session
    states = []
    actions = []
    rewards = []
    values = []

    # Generate the rollout.

    for i in range(self.a3c.max_rollout_length):
      if self.env.terminated:
        break
      state = self.env.state
      states.append(state)
      feed_dict = self.create_feed_dict(state)
      if self.a3c.continuous:
        tensors = [self.action_mean, self.action_std, self.value]
      else:
        tensors = [self.action_prob, self.value]
      results = session.run(
          tensors + self.rnn_final_states, feed_dict=feed_dict)
      value = results[len(tensors) - 1]
      self.rnn_states = [np.squeeze(r, 0) for r in results[len(tensors):]]
      action = self.a3c._select_action_from_outputs(results[:len(tensors) - 1],
                                                    False)
      actions.append(action)
      values.append(float(value))
      rewards.append(self.env.step(action))

    # Compute an estimate of the reward for the rest of the episode.

    if not self.env.terminated:
      feed_dict = self.create_feed_dict(self.env.state)
      final_value = self.a3c.discount_factor * float(
          session.run(self.value, feed_dict))
    else:
      final_value = 0.0
    values.append(final_value)
    if self.env.terminated:
      self.env.reset()
      self.rnn_states = self.a3c._policy.rnn_initial_states
    return states, actions, np.array(
        rewards, dtype=np.float32), np.array(
            values, dtype=np.float32)

  def process_rollout(self, states, actions, rewards, values,
                      initial_rnn_states, step_count):
    """Train the network based on a rollout."""

    # Compute the discounted rewards and advantages.

    discounted_rewards = rewards.copy()
    discounted_rewards[-1] += values[-1]
    advantages = rewards - values[:-1] + self.a3c.discount_factor * np.array(
        values[1:])
    for j in range(len(rewards) - 1, 0, -1):
      discounted_rewards[j -
                         1] += self.a3c.discount_factor * discounted_rewards[j]
      advantages[
          j -
          1] += self.a3c.discount_factor * self.a3c.advantage_lambda * advantages[j]

    # Record the actions, converting to one-hot if necessary.

    actions_matrix = []
    if self.a3c.continuous:
      for action in actions:
        actions_matrix.append(action)
    else:
      n_actions = self.env.n_actions
      for action in actions:
        a = np.zeros(n_actions)
        a[action] = 1.0
        actions_matrix.append(a)

    # Rearrange the states into the proper set of arrays.

    if self.a3c._state_is_list:
      state_arrays = [[] for i in range(len(self.model._input_shapes))]
      for state in states:
        for j in range(len(state)):
          state_arrays[j].append(state[j])
    else:
      state_arrays = [states]

    # Build the feed dict and apply gradients.

    feed_dict = {}
    for f, s in zip(self.model._input_placeholders, state_arrays):
      feed_dict[f] = s
    for f, s in zip(self.model._input_placeholders[len(state_arrays):],
                    initial_rnn_states):
      feed_dict[f] = np.expand_dims(s, axis=0)
    feed_dict[self.model._weights_placeholders[0]] = discounted_rewards
    feed_dict[self.model._label_placeholders[0]] = actions_matrix
    feed_dict[self.model._weights_placeholders[1]] = advantages
    feed_dict[self.model._global_step] = step_count
    self.a3c._session.run(self.train_op, feed_dict=feed_dict)

  def process_rollout_with_hindsight(self, states, actions, initial_rnn_states,
                                     step_count):
    """Create a new rollout by applying hindsight to an existing one, then train the network."""
    hindsight_states, rewards = self.env.apply_hindsight(
        states, actions, states[-1])
    if self.a3c._state_is_list:
      state_arrays = [[] for i in range(len(self.model._input_shapes))]
      for state in hindsight_states:
        for j in range(len(state)):
          state_arrays[j].append(state[j])
    else:
      state_arrays = [hindsight_states]
    state_arrays += initial_rnn_states
    feed_dict = {}
    for f, s in zip(self.model._input_placeholders, state_arrays):
      feed_dict[f] = s
    values = self.a3c._session.run(self.value, feed_dict=feed_dict)
    values = np.append(values.flatten(), 0.0)
    self.process_rollout(hindsight_states, actions, np.array(rewards),
                         np.array(values), initial_rnn_states, step_count)

  def create_feed_dict(self, state):
    """Create a feed dict for use during a rollout."""
    if not self.a3c._state_is_list:
      state = [state]
    state = state + self.rnn_states
    feed_dict = dict((f, np.expand_dims(s, axis=0))
                     for f, s in zip(self.model._input_placeholders, state))
    return feed_dict
