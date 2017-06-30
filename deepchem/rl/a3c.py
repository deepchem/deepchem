"""Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph import TFWrapper
from deepchem.models.tensorgraph.layers import Feature, Weights, Label, Layer
import numpy as np
import tensorflow as tf
import copy
import multiprocessing
import os
import re
import threading


class A3CLoss(Layer):
  """This layer computes the loss function for A3C."""

  def __init__(self, value_weight, entropy_weight, **kwargs):
    super(A3CLoss, self).__init__(**kwargs)
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight

  def create_tensor(self, **kwargs):
    reward, action, prob, value, advantage = [
        layer.out_tensor for layer in self.in_layers
    ]
    prob = prob + np.finfo(np.float32).eps
    log_prob = tf.log(prob)
    policy_loss = -tf.reduce_sum(advantage * tf.reduce_sum(action * log_prob))
    value_loss = tf.reduce_sum(tf.square(reward - value))
    entropy = -tf.reduce_sum(prob * log_prob)
    self.out_tensor = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy
    return self.out_tensor


def _create_feed_dict(features, state):
  return dict((f.out_tensor, np.expand_dims(s, axis=0))
              for f, s in zip(features, state))


class A3C(object):
  """
  Implements the Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning.

  This algorithm requires the policy to output two quantities: a vector giving the probability of
  taking each action, and an estimate of the value function for the current state.  It optimizes
  both outputs at once using a loss that is the sum of three terms:

  1. The policy loss, which seeks to maximize the discounted reward for each action.
  2. The value loss, which tries to make the value estimate match the actual discounted reward
     that was attained at each step.
  3. An entropy term to encourage exploration.

  This class only supports environments with discrete action spaces, not continuous ones.  The
  "action" argument passed to the environment is an integer, giving the index of the action to perform.
  """

  def __init__(self,
               env,
               policy,
               max_rollout_length=20,
               discount_factor=0.99,
               value_weight=1.0,
               entropy_weight=0.01,
               optimizer=None,
               model_dir=None):
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
    discount_factor: float
      the discount factor to use when computing rewards
    value_weight: float
      a scale factor for the value loss term in the loss function
    entropy_weight: float
      a scale factor for the entropy term in the loss function
    optimizer: TFWrapper
      a callable object that creates the optimizer to use.  If None, a default optimizer is used.
    model_dir: str
      the directory in which the model will be saved.  If None, a temporary directory will be created.
    """
    self._env = env
    self._policy = policy
    self.max_rollout_length = max_rollout_length
    self.discount_factor = discount_factor
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight
    if optimizer is None:
      self._optimizer = TFWrapper(
          tf.train.AdamOptimizer, learning_rate=0.001, beta1=0.9, beta2=0.999)
    else:
      self._optimizer = optimizer
    (self._graph, self._features, self._rewards, self._actions,
     self._action_prob, self._value, self._advantages) = self._build_graph(
         None, 'global', model_dir)
    with self._graph._get_tf("Graph").as_default():
      self._session = tf.Session()

  def _build_graph(self, tf_graph, scope, model_dir):
    """Construct a TensorGraph containing the policy and loss calculations."""
    features = [Feature(shape=[None] + list(s)) for s in self._env.state_shape]
    policy_layers = self._policy.create_layers(features)
    action_prob = policy_layers['action_prob']
    value = policy_layers['value']
    rewards = Weights(shape=(None,))
    advantages = Weights(shape=(None,))
    actions = Label(shape=(None, self._env.n_actions))
    loss = A3CLoss(
        self.value_weight,
        self.entropy_weight,
        in_layers=[rewards, actions, action_prob, value, advantages])
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
    return graph, features, rewards, actions, action_prob, value, advantages

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
            name=worker.scope,
            target=lambda: worker.run(step_count, total_steps))
        threads.append(thread)
        thread.start()
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
      saver = tf.train.Saver(variables, max_to_keep=max_checkpoints_to_keep)
      checkpoint_index = 0
      while True:
        threads = [t for t in threads if t.isAlive()]
        if len(threads) > 0:
          threads[0].join(checkpoint_interval)
        checkpoint_index += 1
        saver.save(
            self._session, self._graph.save_file, global_step=checkpoint_index)
        if len(threads) == 0:
          break

  def predict(self, state):
    """Compute the policy's output predictions for a state.

    Parameters
    ----------
    state: array
      the state of the environment for which to generate predictions

    Returns
    -------
    the array of action probabilities, and the estimated value function
    """
    with self._graph._get_tf("Graph").as_default():
      feed_dict = _create_feed_dict(self._features, state)
      return self._session.run(
          [self._action_prob.out_tensor, self._value.out_tensor],
          feed_dict=feed_dict)

  def select_action(self, state, deterministic=False):
    """Select an action to perform based on the environment's state.

    Parameters
    ----------
    state: array
      the state of the environment for which to select an action
    deterministic: bool
      if True, always return the best action (that is, the one with highest probability).
      If False, randomly select an action based on the computed probabilities.

    Returns
    -------
    the index of the selected action
    """
    with self._graph._get_tf("Graph").as_default():
      feed_dict = _create_feed_dict(self._features, state)
      probabilities = self._session.run(
          self._action_prob.out_tensor, feed_dict=feed_dict)
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


class _Worker(object):
  """A Worker object is created for each training thread."""

  def __init__(self, a3c, index):
    self.a3c = a3c
    self.index = index
    self.scope = 'worker%d' % index
    self.env = copy.deepcopy(a3c._env)
    self.env.reset()
    self.graph, self.features, self.rewards, self.actions, self.action_prob, self.value, self.advantages = a3c._build_graph(
        a3c._graph._get_tf('Graph'), self.scope, None)
    with a3c._graph._get_tf("Graph").as_default():
      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.scope)
      global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      'global')
      gradients = tf.gradients(self.graph.loss.out_tensor, local_vars)
      grads_and_vars = list(zip(gradients, global_vars))
      self.train_op = a3c._graph._get_tf('Optimizer').apply_gradients(
          grads_and_vars)
      self.update_local_variables = tf.group(
          * [tf.assign(v1, v2) for v1, v2 in zip(local_vars, global_vars)])

  def run(self, step_count, total_steps):
    with self.graph._get_tf("Graph").as_default():
      session = self.a3c._session
      while step_count[0] < total_steps:
        session.run(self.update_local_variables)
        episode_states, episode_actions, episode_rewards, episode_advantages = self.create_rollout(
        )
        feed_dict = {}
        for f, s in zip(self.features, episode_states):
          feed_dict[f.out_tensor] = s
        feed_dict[self.rewards.out_tensor] = episode_rewards
        feed_dict[self.actions.out_tensor] = episode_actions
        feed_dict[self.advantages.out_tensor] = episode_advantages
        session.run(self.train_op, feed_dict=feed_dict)
        step_count[0] += len(episode_actions)

  def create_rollout(self):
    """Generate a rollout."""
    n_actions = self.env.n_actions
    session = self.a3c._session
    states = [[] for i in range(len(self.features))]
    actions = []
    rewards = []
    values = []
    for i in range(self.a3c.max_rollout_length):
      if self.env.terminated:
        break
      state = self.env.state
      for j in range(len(state)):
        states[j].append(state[j])
      feed_dict = _create_feed_dict(self.features, state)
      probabilities, value = session.run(
          [self.action_prob.out_tensor, self.value.out_tensor],
          feed_dict=feed_dict)
      action = np.random.choice(np.arange(n_actions), p=probabilities[0])
      actions.append(np.zeros(n_actions))
      actions[i][action] = 1.0
      values.append(float(value))
      rewards.append(self.env.step(action))
    if not self.env.terminated:
      # Add an estimate of the reward for the rest of the episode.
      feed_dict = _create_feed_dict(self.features, self.env.state)
      rewards[-1] += self.a3c.discount_factor * float(
          session.run(self.value.out_tensor, feed_dict))
    for j in range(len(rewards) - 1, 0, -1):
      rewards[j - 1] += self.a3c.discount_factor * rewards[j]
    rewards_array = np.array(rewards)
    advantages = rewards_array - np.array(values)
    if self.env.terminated:
      self.env.reset()
    return np.array(states), np.array(actions), rewards_array, advantages
