"""Monte Carlo tree search algorithm for reinforcement learning."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph.optimizers import Adam
from deepchem.models.tensorgraph.layers import Feature, Weights, Label, Layer
import numpy as np
import tensorflow as tf
import collections
import copy


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
  """

  def __init__(self,
               env,
               policy,
               max_search_depth=100,
               n_search_episodes=1000,
               discount_factor=0.99,
               value_weight=1.0,
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
    max_search_depth: int
      the maximum length of rollouts to generate
    discount_factor: float
      the discount factor to use when computing rewards
    value_weight: float
      a scale factor for the value loss term in the loss function
    optimizer: Optimizer
      the optimizer to use.  If None, a default optimizer is used.
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
    self._rnn_states = self._graph.rnn_zero_states

    self.c_puct = 1.0
    self.temperature = 1.0

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
    return graph, features, action_prob, value, search_prob, search_value

  def fit(self,
          iterations,
          steps_per_iteration = 10000,
          epochs_per_iteration = 1,
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
      self._graph.session.run(tf.global_variables_initializer())
      if restore:
        self.restore()
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope='global')
      saver = tf.train.Saver(variables, max_to_keep=max_checkpoints_to_keep)
      checkpoint_index = 0
      for iteration in range(iterations):
        print(iteration)
        buffer = self._run_episodes(steps_per_iteration)
        self._optimize_policy(buffer, epochs_per_iteration)
        checkpoint_index += 1
        saver.save(
            self._graph.session, self._graph.save_file, global_step=checkpoint_index)

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
      tensors = [self._pred_prob.out_tensor, self._pred_value.out_tensor]
      if save_states:
        tensors += self._graph.rnn_final_states
      results = self._graph.session.run(tensors, feed_dict=feed_dict)
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
      tensors = [self._pred_prob.out_tensor]
      if save_states:
        tensors += self._graph.rnn_final_states
      results = self._graph.session.run(tensors, feed_dict=feed_dict)
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
      saver.restore(self._graph.session, last_checkpoint)

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

  def _run_episodes(self, steps):
    buffer = []
    self._env.reset()
    root = TreeSearchNode(0.0)
    for step in range(steps):
      prob, reward = self._do_tree_search(root)
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
#        root = root.children[action]
        root = TreeSearchNode(0.0)
    return buffer

  def _optimize_policy(self, buffer, epochs):
    batch_size = self._graph.batch_size
    n_batches = len(buffer) // batch_size
    for epoch in range(epochs):
      np.random.shuffle(buffer)

      def generate_batches():
        for batch in range(n_batches):
          indices = list(range(batch*batch_size, (batch+1)*batch_size))
          feed_dict = {}
          for i, f in enumerate(self._features):
            feed_dict[f] = np.stack(buffer[j][0][i] for j in indices)
          feed_dict[self._search_prob] = np.stack(buffer[j][1] for j in indices)
          feed_dict[self._search_value] = np.array([buffer[j][2] for j in indices])
          yield feed_dict

      loss = self._graph.fit_generator(generate_batches(), checkpoint_interval=0)

  def _do_tree_search(self, root):
    # Build the tree.

    traces = []
    for i in range(self.n_search_episodes):
      env = copy.deepcopy(self._env)
      trace = Trace()
      traces.append(trace)
      self._create_trace(env, root, trace)
      self._record_trace_rewards(trace)

    # Compute the final probabilities and expected reward.

    prob = np.array([c.count**self.temperature for c in root.children])
    prob /= np.sum(prob)
    reward = np.sum(p*c.mean_reward for p, c in zip(prob, root.children))
    return prob, reward

  def _create_trace(self, env, node, trace):
    trace.nodes.append(node)
    node.count += 1
    if env.terminated:
      # Mark this node as terminal
      node.children = None
      node.value = 0.0
      return
    if node.children is not None and len(node.children) == 0:
      # Expand this node.
      prob_pred, value = self.predict(env.state)
      node.value = float(value)
      node.children = [TreeSearchNode(p) for p in prob_pred[0]]
    if len(trace.nodes) == self.max_search_depth:
      return

    # Select the next action to perform.

    total_counts = sum(c.count for c in node.children)
    if total_counts == 0:
      score = [c.prior_prob for c in node.children]
    else:
      scale = self.c_puct*np.sqrt(total_counts)
      score = [c.mean_reward + scale*c.prior_prob/(1+c.count) for c in node.children]
    action = np.argmax(score)
    next_node = node.children[action]
    next_node.reward = env.step(action)

    # Recursively build the tree.

    self._create_trace(env, next_node, trace)

  def _record_trace_rewards(self, trace):
    value = trace.nodes[-1].value
    for node in reversed(trace.nodes):
      value = node.reward + self.discount_factor*value
      node.total_reward += value
      node.mean_reward = node.total_reward/node.count


class TreeSearchNode(object):
  """Represents a node in the Monte Carlo tree search."""

  def __init__(self, prior_prob):
    self.count = 0
    self.reward = 0.0
    self.total_reward = 0.0
    self.mean_reward = 0.0
    self.prior_prob = prior_prob
    self.children = []


class Trace(object):

  def __init__(self):
    self.nodes = []
