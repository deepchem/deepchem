"""Proximal Policy Optimization (PPO) algorithm for reinforcement learning."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph.optimizers import Adam
from deepchem.models.tensorgraph.layers import Feature, Weights, Label, Layer
import numpy as np
import tensorflow as tf
import collections
import copy
import multiprocessing
from multiprocessing.dummy import Pool
import os
import re
import time


class PPOLoss(Layer):
  """This layer computes the loss function for PPO."""

  def __init__(self, value_weight, entropy_weight, clipping_width, **kwargs):
    super(PPOLoss, self).__init__(**kwargs)
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight
    self.clipping_width = clipping_width

  def create_tensor(self, **kwargs):
    reward, action, prob, value, advantage, old_prob = [
        layer.out_tensor for layer in self.in_layers
    ]
    machine_eps = np.finfo(np.float32).eps
    prob += machine_eps
    old_prob += machine_eps
    ratio = tf.reduce_sum(action * prob, axis=1) / old_prob
    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clipping_width,
                                     1 + self.clipping_width)
    policy_loss = -tf.reduce_mean(
        tf.minimum(ratio * advantage, clipped_ratio * advantage))
    value_loss = tf.reduce_mean(tf.square(reward - value))
    entropy = -tf.reduce_mean(tf.reduce_sum(prob * tf.log(prob), axis=1))
    self.out_tensor = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy
    return self.out_tensor


class PPO(object):
  """
  Implements the Proximal Policy Optimization (PPO) algorithm for reinforcement learning.

  The algorithm is described in Schulman et al, "Proximal Policy Optimization Algorithms"
  (https://openai-public.s3-us-west-2.amazonaws.com/blog/2017-07/ppo/ppo-arxiv.pdf).
  This class requires the policy to output two quantities: a vector giving the probability of
  taking each action, and an estimate of the value function for the current state.  It
  optimizes both outputs at once using a loss that is the sum of three terms:

  1. The policy loss, which seeks to maximize the discounted reward for each action.
  2. The value loss, which tries to make the value estimate match the actual discounted reward
     that was attained at each step.
  3. An entropy term to encourage exploration.

  This class only supports environments with discrete action spaces, not continuous ones.  The
  "action" argument passed to the environment is an integer, giving the index of the action to perform.

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
               optimization_rollouts=8,
               optimization_epochs=4,
               batch_size=64,
               clipping_width=0.2,
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
      the Policy to optimize.  Its create_layers() method must return a map containing the
      keys 'action_prob' and 'value', corresponding to the action probabilities and value estimate
    max_rollout_length: int
      the maximum length of rollouts to generate
    optimization_rollouts: int
      the number of rollouts to generate for each iteration of optimization
    optimization_epochs: int
      the number of epochs of optimization to perform within each iteration
    batch_size: int
      the batch size to use during optimization.  If this is 0, each rollout will be used as a
      separate batch.
    clipping_width: float
      in computing the PPO loss function, the probability ratio is clipped to the range
      (1-clipping_width, 1+clipping_width)
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
    self.optimization_rollouts = optimization_rollouts
    self.optimization_epochs = optimization_epochs
    self.batch_size = batch_size
    self.clipping_width = clipping_width
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
    (self._graph, self._features, self._rewards, self._actions,
     self._action_prob, self._value, self._advantages,
     self._old_action_prob) = self._build_graph(None, 'global', model_dir)
    with self._graph._get_tf("Graph").as_default():
      self._session = tf.Session()
      self._train_op = self._graph._get_tf('Optimizer').minimize(
          self._graph.loss.out_tensor)
    self._rnn_states = self._graph.rnn_zero_states
    if len(self._rnn_states) > 0 and batch_size != 0:
      raise ValueError(
          'Cannot batch rollouts when the policy contains a recurrent layer.  Set batch_size to 0.'
      )

  def _build_graph(self, tf_graph, scope, model_dir):
    """Construct a TensorGraph containing the policy and loss calculations."""
    state_shape = self._env.state_shape
    state_dtype = self._env.state_dtype
    if not self._state_is_list:
      state_shape = [state_shape]
      state_dtype = [state_dtype]
    features = []
    for s, d in zip(state_shape, state_dtype):
      features.append(Feature(shape=[None] + list(s), dtype=tf.as_dtype(d)))
    policy_layers = self._policy.create_layers(features)
    action_prob = policy_layers['action_prob']
    value = policy_layers['value']
    rewards = Weights(shape=(None,))
    advantages = Weights(shape=(None,))
    old_action_prob = Weights(shape=(None,))
    actions = Label(shape=(None, self._env.n_actions))
    loss = PPOLoss(
        self.value_weight,
        self.entropy_weight,
        self.clipping_width,
        in_layers=[
            rewards, actions, action_prob, value, advantages, old_action_prob
        ])
    graph = TensorGraph(
        batch_size=self.max_rollout_length,
        use_queue=False,
        graph=tf_graph,
        model_dir=model_dir)
    for f in features:
      graph._add_layer(f)
    graph.add_output(action_prob)
    graph.add_output(value)
    graph.set_loss(loss)
    graph.set_optimizer(self._optimizer)
    with graph._get_tf("Graph").as_default():
      with tf.variable_scope(scope):
        graph.build()
    return graph, features, rewards, actions, action_prob, value, advantages, old_action_prob

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
    with self._graph._get_tf("Graph").as_default():
      step_count = 0
      workers = []
      threads = []
      for i in range(self.optimization_rollouts):
        workers.append(_Worker(self, i))
      self._session.run(tf.global_variables_initializer())
      if restore:
        self.restore()
      pool = Pool()
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
      saver = tf.train.Saver(variables, max_to_keep=max_checkpoints_to_keep)
      checkpoint_index = 0
      checkpoint_time = time.time()
      while step_count < total_steps:
        # Have the worker threads generate the rollouts for this iteration.

        rollouts = []
        pool.map(lambda x: rollouts.extend(x.run()), workers)

        # Perform optimization.

        for epoch in range(self.optimization_epochs):
          if self.batch_size == 0:
            batches = rollouts
          else:
            batches = self._iter_batches(rollouts)
          for batch in batches:
            initial_rnn_states, state_arrays, discounted_rewards, actions_matrix, action_prob, advantages = batch

            # Build the feed dict and run the optimizer.

            feed_dict = {}
            for placeholder, value in zip(self._graph.rnn_initial_states,
                                          initial_rnn_states):
              feed_dict[placeholder] = value
            for f, s in zip(self._features, state_arrays):
              feed_dict[f.out_tensor] = s
            feed_dict[self._rewards.out_tensor] = discounted_rewards
            feed_dict[self._actions.out_tensor] = actions_matrix
            feed_dict[self._advantages.out_tensor] = advantages
            feed_dict[self._old_action_prob.out_tensor] = action_prob
            feed_dict[self._graph.get_global_step()] = step_count
            self._session.run(self._train_op, feed_dict=feed_dict)

        # Update the number of steps taken so far and perform checkpointing.

        new_steps = sum(len(r[3]) for r in rollouts)
        if self.use_hindsight:
          new_steps /= 2
        step_count += new_steps
        if step_count >= total_steps or time.time(
        ) >= checkpoint_time + checkpoint_interval:
          saver.save(
              self._session,
              self._graph.save_file,
              global_step=checkpoint_index)
          checkpoint_index += 1
          checkpoint_time = time.time()

  def _iter_batches(self, rollouts):
    """Given a set of rollouts, merge them into batches for optimization."""

    # Merge all the rollouts into a single set of arrays.

    state_arrays = []
    for i in range(len(rollouts[0][1])):
      state_arrays.append(np.concatenate([r[1][i] for r in rollouts]))
    discounted_rewards = np.concatenate([r[2] for r in rollouts])
    actions_matrix = np.concatenate([r[3] for r in rollouts])
    action_prob = np.concatenate([r[4] for r in rollouts])
    advantages = np.concatenate([r[5] for r in rollouts])
    total_length = len(discounted_rewards)

    # Iterate slices.

    start = 0
    while start < total_length:
      end = min(start + self.batch_size, total_length)
      batch = [[]]
      batch.append([s[start:end] for s in state_arrays])
      batch.append(discounted_rewards[start:end])
      batch.append(actions_matrix[start:end])
      batch.append(action_prob[start:end])
      batch.append(advantages[start:end])
      start = end
      yield batch

  def predict(self, state, use_saved_states=True, save_states=True):
    """Compute the policy's output predictions for a state.

    If the policy involves recurrent layers, this method can preserve their internal
    states between calls.  Use the use_saved_states and save_states arguments to specify
    how it should behave.

    Parameters
    ----------
    state: array
      the state of the environment for which to generate predictions
    use_saved_states: bool
      if True, the states most recently saved by a previous call to predict() or select_action()
      will be used as the initial states.  If False, the internal states of all recurrent layers
      will be set to all zeros before computing the predictions.
    save_states: bool
      if True, the internal states of all recurrent layers at the end of the calculation
      will be saved, and any previously saved states will be discarded.  If False, the
      states at the end of the calculation will be discarded, and any previously saved
      states will be kept.

    Returns
    -------
    the array of action probabilities, and the estimated value function
    """
    if not self._state_is_list:
      state = [state]
    with self._graph._get_tf("Graph").as_default():
      feed_dict = self._create_feed_dict(state, use_saved_states)
      tensors = [self._action_prob.out_tensor, self._value.out_tensor]
      if save_states:
        tensors += self._graph.rnn_final_states
      results = self._session.run(tensors, feed_dict=feed_dict)
      if save_states:
        self._rnn_states = results[2:]
      return results[:2]

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
    state: array
      the state of the environment for which to select an action
    deterministic: bool
      if True, always return the best action (that is, the one with highest probability).
      If False, randomly select an action based on the computed probabilities.
    use_saved_states: bool
      if True, the states most recently saved by a previous call to predict() or select_action()
      will be used as the initial states.  If False, the internal states of all recurrent layers
      will be set to all zeros before computing the predictions.
    save_states: bool
      if True, the internal states of all recurrent layers at the end of the calculation
      will be saved, and any previously saved states will be discarded.  If False, the
      states at the end of the calculation will be discarded, and any previously saved
      states will be kept.

    Returns
    -------
    the index of the selected action
    """
    if not self._state_is_list:
      state = [state]
    with self._graph._get_tf("Graph").as_default():
      feed_dict = self._create_feed_dict(state, use_saved_states)
      tensors = [self._action_prob.out_tensor]
      if save_states:
        tensors += self._graph.rnn_final_states
      results = self._session.run(tensors, feed_dict=feed_dict)
      probabilities = results[0]
      if save_states:
        self._rnn_states = results[1:]
      if deterministic:
        return probabilities.argmax()
      else:
        return np.random.choice(
            np.arange(self._env.n_actions), p=probabilities[0])

  def restore(self):
    """Reload the model parameters from the most recent checkpoint file."""
    last_checkpoint = tf.train.latest_checkpoint(self._graph.model_dir)
    if last_checkpoint is None:
      raise ValueError('No checkpoint found')
    with self._graph._get_tf("Graph").as_default():
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
      saver = tf.train.Saver(variables)
      saver.restore(self._session, last_checkpoint)

  def _create_feed_dict(self, state, use_saved_states):
    """Create a feed dict for use by predict() or select_action()."""
    feed_dict = dict((f.out_tensor, np.expand_dims(s, axis=0))
                     for f, s in zip(self._features, state))
    if use_saved_states:
      rnn_states = self._rnn_states
    else:
      rnn_states = self._graph.rnn_zero_states
    for (placeholder, value) in zip(self._graph.rnn_initial_states, rnn_states):
      feed_dict[placeholder] = value
    return feed_dict


class _Worker(object):
  """A Worker object is created for each training thread."""

  def __init__(self, ppo, index):
    self.ppo = ppo
    self.index = index
    self.scope = 'worker%d' % index
    self.env = copy.deepcopy(ppo._env)
    self.env.reset()
    self.graph, self.features, self.rewards, self.actions, self.action_prob, self.value, self.advantages, self.old_action_prob = ppo._build_graph(
        ppo._graph._get_tf('Graph'), self.scope, None)
    self.rnn_states = self.graph.rnn_zero_states
    with ppo._graph._get_tf("Graph").as_default():
      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.scope)
      global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      'global')
      self.update_local_variables = tf.group(
          * [tf.assign(v1, v2) for v1, v2 in zip(local_vars, global_vars)])

  def run(self):
    rollouts = []
    with self.graph._get_tf("Graph").as_default():
      self.ppo._session.run(self.update_local_variables)
      initial_rnn_states = self.rnn_states
      states, actions, action_prob, rewards, values = self.create_rollout()
      rollouts.append(
          self.process_rollout(states, actions, action_prob, rewards, values,
                               initial_rnn_states))
      if self.ppo.use_hindsight:
        rollouts.append(
            self.process_rollout_with_hindsight(states, actions,
                                                initial_rnn_states))
    return rollouts

  def create_rollout(self):
    """Generate a rollout."""
    n_actions = self.env.n_actions
    session = self.ppo._session
    states = []
    action_prob = []
    actions = []
    rewards = []
    values = []

    # Generate the rollout.

    for i in range(self.ppo.max_rollout_length):
      if self.env.terminated:
        break
      state = self.env.state
      states.append(state)
      feed_dict = self.create_feed_dict(state)
      results = session.run(
          [self.action_prob.out_tensor, self.value.out_tensor] +
          self.graph.rnn_final_states,
          feed_dict=feed_dict)
      probabilities, value = results[:2]

      self.rnn_states = results[2:]
      action = np.random.choice(np.arange(n_actions), p=probabilities[0])
      actions.append(action)
      action_prob.append(probabilities[0][action])
      values.append(float(value))
      rewards.append(self.env.step(action))

    # Compute an estimate of the reward for the rest of the episode.

    if not self.env.terminated:
      feed_dict = self.create_feed_dict(self.env.state)
      final_value = self.ppo.discount_factor * float(
          session.run(self.value.out_tensor, feed_dict))
    else:
      final_value = 0.0
    values.append(final_value)
    if self.env.terminated:
      self.env.reset()
      self.rnn_states = self.graph.rnn_zero_states
    return states, np.array(
        actions, dtype=np.int32), np.array(action_prob), np.array(
            rewards), np.array(values)

  def process_rollout(self, states, actions, action_prob, rewards, values,
                      initial_rnn_states):
    """Construct the arrays needed for training."""

    # Compute the discounted rewards and advantages.

    discounted_rewards = rewards.copy()
    discounted_rewards[-1] += values[-1]
    advantages = rewards - values[:-1] + self.ppo.discount_factor * np.array(
        values[1:])
    for j in range(len(rewards) - 1, 0, -1):
      discounted_rewards[j -
                         1] += self.ppo.discount_factor * discounted_rewards[j]
      advantages[
          j -
          1] += self.ppo.discount_factor * self.ppo.advantage_lambda * advantages[
              j]

    # Convert the actions to one-hot.

    n_actions = self.env.n_actions
    actions_matrix = []
    for action in actions:
      a = np.zeros(n_actions)
      a[action] = 1.0
      actions_matrix.append(a)

    # Rearrange the states into the proper set of arrays.

    if self.ppo._state_is_list:
      state_arrays = [[] for i in range(len(self.features))]
      for state in states:
        for j in range(len(state)):
          state_arrays[j].append(state[j])
    else:
      state_arrays = [states]

    # Return the processed arrays.

    return (initial_rnn_states, state_arrays, discounted_rewards,
            actions_matrix, action_prob, advantages)

  def process_rollout_with_hindsight(self, states, actions, initial_rnn_states):
    """Create a new rollout by applying hindsight to an existing one, then process it."""
    hindsight_states, rewards = self.env.apply_hindsight(
        states, actions, states[-1])
    if self.ppo._state_is_list:
      state_arrays = [[] for i in range(len(self.features))]
      for state in hindsight_states:
        for j in range(len(state)):
          state_arrays[j].append(state[j])
    else:
      state_arrays = [hindsight_states]
    feed_dict = {}
    for placeholder, value in zip(self.graph.rnn_initial_states,
                                  initial_rnn_states):
      feed_dict[placeholder] = value
    for f, s in zip(self.features, state_arrays):
      feed_dict[f.out_tensor] = s
    values, probabilities = self.ppo._session.run(
        [self.value.out_tensor, self.action_prob.out_tensor],
        feed_dict=feed_dict)
    values = np.append(values.flatten(), 0.0)
    action_prob = probabilities[np.arange(len(actions)), actions]
    return self.process_rollout(hindsight_states, actions, action_prob,
                                np.array(rewards),
                                np.array(values), initial_rnn_states)

  def create_feed_dict(self, state):
    """Create a feed dict for use during a rollout."""
    if not self.ppo._state_is_list:
      state = [state]
    feed_dict = dict((f.out_tensor, np.expand_dims(s, axis=0))
                     for f, s in zip(self.features, state))
    for (placeholder, value) in zip(self.graph.rnn_initial_states,
                                    self.rnn_states):
      feed_dict[placeholder] = value
    return feed_dict
