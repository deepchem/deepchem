"""Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph import TFWrapper
from deepchem.models.tensorgraph.layers import Feature, Weights, Label, Layer
import numpy as np
import tensorflow as tf
import copy
import multiprocessing
import threading

class A3CLoss(Layer):
  """This layer computes the loss function for A3C."""
  def __init__(self, value_weight, entropy_weight, **kwargs):
    super(A3CLoss, self).__init__(**kwargs)
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight

  def create_tensor(self, **kwargs):
    reward, action, prob, value = [layer.out_tensor for layer in self.in_layers]
    log_prob = tf.log(prob)
    policy_loss = -tf.reduce_sum((reward-value)*tf.reduce_sum(action*log_prob))
    value_loss = tf.reduce_sum(tf.square(reward-value))
    entropy = -tf.reduce_sum(prob*log_prob)
    self.out_tensor = policy_loss + self.value_weight*value_loss - self.entropy_weight*entropy
    return self.out_tensor

class A3C(object):
  """
  Implements the Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning.

  This algorithm requires the policy to output two quantities: a vector giving the probably of
  taking each action, and an estimate of the value function for the current state.  It optimizes
  both outputs at once using a loss that is the sum of three terms:

  1. The policy loss, which seeks to maximize the discounted reward for each action.
  2. The value loss, which tries to make the value estimate match the actual discounted reward
     that was attained at each step.
  3. An entropy term to encourage exploration.
  """

  def __init__(self, env, policy, max_rollout_length=20, discount_factor=0.99, value_weight=0.25, entropy_weight=0.01):
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
    entropy:weight: float
      a scale factor for the entropy term in the loss function
    """
    self._env = env
    self._policy = policy
    self.max_rollout_length = max_rollout_length
    self.discount_factor = discount_factor
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight
    self.optimizer = TFWrapper(tf.train.AdamOptimizer, learning_rate=0.001, beta1=0.9, beta2=0.999)

  def _build_graph(self, tf_graph, scope):
    """Construct a TensorGraph containing the policy and loss calculations."""
    features = Feature(shape=[None]+list(self._env.state_shape))
    policy_layers = self._policy.create_layers(features)
    action_prob = policy_layers['action_prob']
    value = policy_layers['value']
    rewards = Weights(shape=(None, 1))
    actions = Label(shape=(None, self._env.n_actions))
    loss = A3CLoss(self.value_weight, self.entropy_weight, in_layers=[rewards, actions, action_prob, value])
    graph = TensorGraph(batch_size=self.max_rollout_length, use_queue=False, graph=tf_graph)
    graph.add_output(action_prob)
    graph.add_output(value)
    graph.set_loss(loss)
    graph.set_optimizer(self.optimizer)
    with graph._get_tf("Graph").as_default():
      with tf.variable_scope(scope):
        graph.build()
    return graph, features, rewards, actions, action_prob, value

  def fit(self, total_steps):
    (graph, features, rewards, actions, action_prob, value) = self._build_graph(None, 'global')
    with graph._get_tf("Graph").as_default():
      train_op = graph._get_tf('train_op')
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step_count = [0]
        workers = []
        threads = []
        for i in range(multiprocessing.cpu_count()):
          workers.append(Worker(self, graph, i))
        for worker in workers:
          thread = threading.Thread(name=worker.scope, target=lambda: worker.run(sess, step_count, total_steps))
          threads.append(thread)
          thread.start()
        for thread in threads:
          thread.join()


class Worker(object):
  def __init__(self, a3c, global_graph, index):
    self.a3c = a3c
    self.index = index
    self.scope = 'worker%d' % index
    self.env = copy.deepcopy(a3c._env)
    self.env.reset()
    self.graph, self.features, self.rewards, self.actions, self.action_prob, self.value = a3c._build_graph(global_graph._get_tf('Graph'), self.scope)
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    gradients = tf.gradients(self.graph.loss.out_tensor, local_vars)
    grads_and_vars = list(zip(gradients, global_vars))
    self.train_op = global_graph._get_tf('Optimizer').apply_gradients(grads_and_vars)
    self.update_local_variables = tf.group(*[tf.assign(v1, v2) for v1, v2 in zip(local_vars, global_vars)])

  def run(self, sess, step_count, total_steps):
    with self.graph._get_tf("Graph").as_default():
      while step_count[0] < total_steps:
        sess.run(self.update_local_variables)
        episode_states, episode_actions, episode_rewards = self.create_rollout(sess)
        feed_dict = {}
        feed_dict[self.features.out_tensor] = episode_states
        feed_dict[self.rewards.out_tensor] = episode_rewards
        feed_dict[self.actions.out_tensor] = episode_actions
        sess.run(self.train_op, feed_dict=feed_dict)
        step_count[0] += len(episode_states)

  def create_rollout(self, sess):
    """Generate a rollout."""
    n_actions = self.env.n_actions
    states = []
    actions = []
    scores = []
    for i in range(self.a3c.max_rollout_length):
      if self.env.terminated:
        break
      states.append(self.env.state)
      feed_dict = {self.features.out_tensor: np.expand_dims(self.env.state, axis=0)}
      probabilities = sess.run(self.action_prob.out_tensor, feed_dict=feed_dict)
      action = np.random.choice([0,1], p=probabilities[0])
      actions.append(np.zeros(n_actions))
      actions[i][action] = 1.0
      scores.append(self.env.step(action))
    if not self.env.terminated:
      # Add an estimate of the reward for the rest of the episode.
      feed_dict = {self.features.out_tensor: np.expand_dims(self.env.state, axis=0)}
      scores[-1] += sess.run(self.value.out_tensor, feed_dict)
    for j in range(len(scores)-1, 0, -1):
      scores[j-1] += self.a3c.discount_factor*scores[j]
    if self.env.terminated:
      self.env.reset()
    return np.array(states), np.array(actions), np.array(scores).reshape((len(scores), 1))
