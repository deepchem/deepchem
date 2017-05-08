"""Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph import TFWrapper
from deepchem.models.tensorgraph.layers import Feature, Weights, Label, Layer
import numpy as np
import tensorflow as tf

class A3CLoss(Layer):
  def __init__(self, value_weight, entropy_weight, **kwargs):
    super(A3CLoss, self).__init__(**kwargs)
    self.value_weight = value_weight
    self.entropy_weight = entropy_weight

  def create_tensor(self, **kwargs):
    reward, action, prob, value = [layer.out_tensor for layer in self.in_layers]
    log_prob = tf.log(prob)
    policy_loss = -tf.reduce_sum((reward-value)*tf.reduce_sum(action*log_prob))
    value_loss = tf.reduce_sum(tf.square(reward-value))
    entropy = tf.reduce_sum(prob*log_prob)
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

  def __init__(self, env, policy, max_rollout_length, discount_factor=0.99, value_weight=0.25, entropy_weight=0.01):
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

  def _build_graph(self):
    """Construct a TensorGraph containing the policy and loss calculations."""
    features = Feature(shape=[None]+list(self._env.state_shape))
    policy_layers = self._policy.create_layers(features)
    action_prob = policy_layers['action_prob']
    value = policy_layers['value']
    rewards = Weights(shape=(None, 1))
    actions = Label(shape=(None, self._env.n_actions))
    loss = A3CLoss(self.value_weight, self.entropy_weight, in_layers=[rewards, actions, action_prob, value])
    graph = TensorGraph(batch_size=self.max_rollout_length, use_queue=False)
    graph.add_output(action_prob)
    graph.add_output(value)
    graph.set_loss(loss)
    graph.set_optimizer(self.optimizer)
    graph.build()
    return graph, features, rewards, actions, action_prob, value

  def _create_rollout(self, sess, features, action_prob):
    """Generate a rollout."""
    n_actions = self._env.n_actions
    self._env.reset()
    states = []
    actions = []
    scores = []
    for i in range(self.max_rollout_length):
      if self._env.terminated:
        break
      states.append(self._env.state)
      feed_dict = {features.out_tensor: np.expand_dims(self._env.state, axis=0)}
      probabilities = sess.run(action_prob.out_tensor, feed_dict=feed_dict)
      action = np.random.choice([0,1], p=probabilities[0])
      actions.append(np.zeros(n_actions))
      actions[i][action] = 1.0
      scores.append(self._env.step(action))
    for j in range(len(scores)-1, 0, -1):
      scores[j-1] += self.discount_factor*scores[j]
    return np.array(states), np.array(actions), np.array(scores).reshape((len(scores), 1))

  def fit(self, n_episodes):
    (graph, features, rewards, actions, action_prob, value) = self._build_graph()
    with graph._get_tf("Graph").as_default():
      train_op = graph._get_tf('train_op')
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_episodes):
          episode_states, episode_actions, episode_rewards = self._create_rollout(sess, features, action_prob)
          feed_dict = {}
          feed_dict[features.out_tensor] = episode_states
          feed_dict[rewards.out_tensor] = episode_rewards
          feed_dict[actions.out_tensor] = episode_actions
          sess.run(train_op, feed_dict=feed_dict)

