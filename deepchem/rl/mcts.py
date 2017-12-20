"""Monte Carlo tree search algorithm for reinforcement learning."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph.optimizers import Adam
from deepchem.models.tensorgraph.layers import Feature, Weights, Label, Layer
import numpy as np
import tensorflow as tf
import collections
import copy
import time


class MCTSLoss(Layer):
  """This layer computes the loss function for MCTS."""

  def __init__(self, value_weight, **kwargs):
    super(MCTSLoss, self).__init__(**kwargs)
    self.value_weight = value_weight

  def create_tensor(self, **kwargs):
    pred_prob, pred_value, search_prob, search_value = [
        layer.out_tensor for layer in self.in_layers
    ]
    log_prob = tf.log(pred_prob + np.finfo(np.float32).eps)
    probability_loss = -tf.reduce_mean(search_prob * log_prob)
    value_loss = tf.reduce_mean(tf.square(pred_value - search_value))
    self.out_tensor = probability_loss + self.value_weight * value_loss
    self.probability_loss = probability_loss
    self.value_loss = value_loss
    return self.out_tensor


class MCTS(object):
  """
  Implements a Monte Carlo tree search algorithm for reinforcement learning.

  This is adapted from Silver et al, "Mastering the game of Go without human
  knowledge" (https://www.nature.com/articles/nature24270).  The methods
  described in that paper rely on features of Go that are not generally true of
  all reinforcement learning problems.  To transform it into a more generally
  useful RL algorithm, it has been necessary to change some aspects of the
  method.  The overall approach used in this implementation is still the same,
  although some of the details differ.

  This class requires the policy to output two quantities: a vector giving the
  probability of taking each action, and an estimate of the value function for
  the current state.  At every step of simulating an episode, it performs an
  expensive tree search to explore the consequences of many possible actions.
  Based on that search, it computes much better estimates for the value function
  of the current state and the desired action probabilities.  In then tries to
  optimize the policy to make its outputs match the result of the tree search.

  Optimization proceeds through a series of iterations.  Each iteration consists
  of two stages:

  1. Simulate many episodes.  At every step perform a tree search to determine
     targets for the probabilities and value function, and store them into a
     buffer.
  2. Optimize the policy using batches drawn from the buffer generated in step 1.

  The tree search involves repeatedly selecting actions starting from the
  current state.  This is done by using deepcopy() to clone the environment.  It
  is essential that this produce a deterministic sequence of states: performing
  an action on the cloned environment must always lead to the same state as
  performing that action on the original environment.  For environments whose
  state transitions are deterministic, this is not a problem.  For ones whose
  state transitions are stochastic, it is essential that the random number
  generator used to select new states be stored as part of the environment and
  be properly cloned by deepcopy().

  This class does not support policies that include recurrent layers.
  """

  def __init__(self,
               env,
               policy,
               max_search_depth=100,
               n_search_episodes=1000,
               discount_factor=0.99,
               value_weight=1.0,
               optimizer=Adam(),
               model_dir=None):
    """Create an object for optimizing a policy.

    Parameters
    ----------
    env: Environment
      the Environment to interact with
    policy: Policy
      the Policy to optimize.  Its create_layers() method must return a dict containing the
      keys 'action_prob' and 'value', corresponding to the action probabilities and value estimate
    max_search_depth: int
      the maximum depth of the tree search, measured in steps
    n_search_episodes: int
      the number of episodes to simulate (up to max_search_depth, if they do not
      terminate first) for each tree search
    discount_factor: float
      the discount factor to use when computing rewards
    value_weight: float
      a scale factor for the value loss term in the loss function
    optimizer: Optimizer
      the optimizer to use
    model_dir: str
      the directory in which the model will be saved.  If None, a temporary directory will be created.
    """
    self._env = copy.deepcopy(env)
    self._policy = policy
    self.max_search_depth = max_search_depth
    self.n_search_episodes = n_search_episodes
    self.discount_factor = discount_factor
    self.value_weight = value_weight
    self._state_is_list = isinstance(env.state_shape[0], collections.Sequence)
    if optimizer is None:
      self._optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
    else:
      self._optimizer = optimizer
    (self._graph, self._features, self._pred_prob, self._pred_value,
     self._search_prob, self._search_value) = self._build_graph(
         None, 'global', model_dir)

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
    search_prob = Label(shape=(None, self._env.n_actions))
    search_value = Label(shape=(None,))
    loss = MCTSLoss(
        self.value_weight,
        in_layers=[action_prob, value, search_prob, search_value])
    graph = TensorGraph(
        batch_size=self.max_search_depth,
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
    if len(graph.rnn_initial_states) > 0:
      raise ValueError('MCTS does not support policies with recurrent layers')
    return graph, features, action_prob, value, search_prob, search_value

  def fit(self,
          iterations,
          steps_per_iteration=10000,
          epochs_per_iteration=10,
          temperature=0.5,
          puct_scale=None,
          max_checkpoints_to_keep=5,
          checkpoint_interval=600,
          restore=False):
    """Train the policy.

    Parameters
    ----------
    iterations: int
      the total number of iterations (simulation followed by optimization) to perform
    steps_per_iteration: int
      the total number of steps to simulate in each iteration.  Every step consists
      of a tree search, followed by selecting an action based on the results of
      the search.
    epochs_per_iteration: int
      the number of epochs of optimization to perform for each iteration.  Each
      epoch involves randomly ordering all the steps that were just simulated in
      the current iteration, splitting them into batches, and looping over the
      batches.
    temperature: float
      the temperature factor to use when selecting a move for each step of
      simulation.  Larger values produce a broader probability distribution and
      hence more exploration.  Smaller values produce a stronger preference for
      whatever action did best in the tree search.
    puct_scale: float
      the scale of the PUCT term in the expression for selecting actions during
      tree search.  This should be roughly similar in magnitude to the rewards
      given by the environment, since the PUCT term is added to the mean
      discounted reward.  This may be None, in which case a value is adaptively
      selected that tries to match the mean absolute value of the discounted
      reward.
    max_checkpoints_to_keep: int
      the maximum number of checkpoint files to keep.  When this number is reached, older
      files are deleted.
    checkpoint_interval: float
      the time interval at which to save checkpoints, measured in seconds
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    """
    if puct_scale is None:
      self._puct_scale = 1.0
      adapt_puct = True
    else:
      self._puct_scale = puct_scale
      adapt_puct = False
    with self._graph._get_tf("Graph").as_default():
      self._graph.session.run(tf.global_variables_initializer())
      if restore:
        self.restore()
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
      saver = tf.train.Saver(variables, max_to_keep=max_checkpoints_to_keep)
      self._checkpoint_index = 0
      self._checkpoint_time = time.time() + checkpoint_interval

      # Run the algorithm.

      for iteration in range(iterations):
        buffer = self._run_episodes(steps_per_iteration, temperature, saver,
                                    adapt_puct)
        self._optimize_policy(buffer, epochs_per_iteration)

      # Save a file checkpoint.

      self._checkpoint_index += 1
      saver.save(
          self._graph.session,
          self._graph.save_file,
          global_step=self._checkpoint_index)

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
    if not self._state_is_list:
      state = [state]
    with self._graph._get_tf("Graph").as_default():
      feed_dict = self._create_feed_dict(state)
      tensors = [self._pred_prob, self._pred_value]
      results = self._graph.session.run(tensors, feed_dict=feed_dict)
      return results[:2]

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
    if not self._state_is_list:
      state = [state]
    with self._graph._get_tf("Graph").as_default():
      feed_dict = self._create_feed_dict(state)
      probabilities = self._graph.session.run(
          self._pred_prob, feed_dict=feed_dict)
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
      saver.restore(self._graph.session, last_checkpoint)

  def _create_feed_dict(self, state):
    """Create a feed dict for use by predict() or select_action()."""
    feed_dict = dict((f.out_tensor, np.expand_dims(s, axis=0))
                     for f, s in zip(self._features, state))
    return feed_dict

  def _run_episodes(self, steps, temperature, saver, adapt_puct):
    """Simulate the episodes for one iteration."""
    buffer = []
    self._env.reset()
    root = TreeSearchNode(0.0)
    for step in range(steps):
      prob, reward = self._do_tree_search(root, temperature, adapt_puct)
      state = self._env.state
      if not self._state_is_list:
        state = [state]
      buffer.append((state, prob, reward))
      action = np.random.choice(np.arange(self._env.n_actions), p=prob)
      self._env.step(action)
      if self._env.terminated:
        self._env.reset()
        root = TreeSearchNode(0.0)
      else:
        root = root.children[action]
      if time.time() > self._checkpoint_time:
        self._checkpoint_index += 1
        saver.save(
            self._graph.session,
            self._graph.save_file,
            global_step=self._checkpoint_index)
        self._checkpoint_time = time.time()
    return buffer

  def _optimize_policy(self, buffer, epochs):
    """Optimize the policy based on the replay buffer from the current iteration."""
    batch_size = self._graph.batch_size
    n_batches = len(buffer) // batch_size
    for epoch in range(epochs):
      np.random.shuffle(buffer)

      def generate_batches():
        for batch in range(n_batches):
          indices = list(range(batch * batch_size, (batch + 1) * batch_size))
          feed_dict = {}
          for i, f in enumerate(self._features):
            feed_dict[f] = np.stack(buffer[j][0][i] for j in indices)
          feed_dict[self._search_prob] = np.stack(buffer[j][1] for j in indices)
          feed_dict[self._search_value] = np.array(
              [buffer[j][2] for j in indices])
          yield feed_dict

      loss = self._graph.fit_generator(
          generate_batches(), checkpoint_interval=0)

  def _do_tree_search(self, root, temperature, adapt_puct):
    """Perform the tree search for a state."""
    # Build the tree.

    for i in range(self.n_search_episodes):
      env = copy.deepcopy(self._env)
      self._create_trace(env, root, 1)

    # Compute the final probabilities and expected reward.

    prob = np.array([c.count**(1.0 / temperature) for c in root.children])
    prob /= np.sum(prob)
    reward = np.sum(p * c.mean_reward for p, c in zip(prob, root.children))
    if adapt_puct:
      scale = np.sum(
          [p * np.abs(c.mean_reward) for p, c in zip(prob, root.children)])
      self._puct_scale = 0.99 * self._puct_scale + 0.01 * scale
    return prob, reward

  def _create_trace(self, env, node, depth):
    """Create one trace as part of the tree search."""
    node.count += 1
    if env.terminated:
      # Mark this node as terminal
      node.children = None
      node.value = 0.0
      return 0.0
    if node.children is not None and len(node.children) == 0:
      # Expand this node.
      prob_pred, value = self.predict(env.state)
      node.value = float(value)
      node.children = [TreeSearchNode(p) for p in prob_pred[0]]
    if depth == self.max_search_depth:
      reward = 0.0
      future_rewards = node.value
    else:
      # Select the next action to perform.

      total_counts = sum(c.count for c in node.children)
      if total_counts == 0:
        score = [c.prior_prob for c in node.children]
      else:
        scale = self._puct_scale * np.sqrt(total_counts)
        score = [
            c.mean_reward + scale * c.prior_prob / (1 + c.count)
            for c in node.children
        ]
      action = np.argmax(score)
      next_node = node.children[action]
      reward = env.step(action)

      # Recursively build the tree.

      future_rewards = self._create_trace(env, next_node, depth + 1)

    # Update statistics for this node.

    future_rewards = reward + self.discount_factor * future_rewards
    node.total_reward += future_rewards
    node.mean_reward = node.total_reward / node.count
    return future_rewards


class TreeSearchNode(object):
  """Represents a node in the Monte Carlo tree search."""

  def __init__(self, prior_prob):
    self.count = 0
    self.reward = 0.0
    self.total_reward = 0.0
    self.mean_reward = 0.0
    self.prior_prob = prior_prob
    self.children = []
