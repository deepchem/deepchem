"""Interface for reinforcement learning."""

from deepchem.rl.a3c import A3C
from deepchem.rl.ppo import PPO


class Environment(object):
  """An environment in which an actor performs actions to accomplish a task.

  An environment has a current state, which is represented as either a single NumPy
  array, or optionally a list of NumPy arrays.  When an action is taken, that causes
  the state to be updated.  Exactly what is meant by an "action" is defined by each
  subclass.  As far as this interface is concerned, it is simply an arbitrary object.
  The environment also computes a reward for each action, and reports when the task
  has been terminated (meaning that no more actions may be taken).

  Environment objects should be written to support pickle and deepcopy operations.
  Many algorithms involve creating multiple copies of the Environment, possibly
  running in different processes or even on different computers.
  """

  def __init__(self, state_shape, n_actions, state_dtype=None):
    """Subclasses should call the superclass constructor in addition to doing their own initialization."""
    self._state_shape = state_shape
    self._n_actions = n_actions
    self._state = None
    self._terminated = None
    if state_dtype is None:
      # Assume all arrays are float32.
      import numpy
      import collections
      if isinstance(state_shape[0], collections.Sequence):
        self._state_dtype = [numpy.float32] * len(state_shape)
      else:
        self._state_dtype = numpy.float32
    else:
      self._state_dtype = state_dtype

  @property
  def state(self):
    """The current state of the environment, represented as either a NumPy array or list of arrays.

    If reset() has not yet been called at least once, this is undefined.
    """
    return self._state

  @property
  def terminated(self):
    """Whether the task has reached its end.

    If reset() has not yet been called at least once, this is undefined.
    """
    return self._terminated

  @property
  def state_shape(self):
    """The shape of the arrays that describe a state.

    If the state is a single array, this returns a tuple giving the shape of that array.
    If the state is a list of arrays, this returns a list of tuples where each tuple is
    the shape of one array.
    """
    return self._state_shape

  @property
  def state_dtype(self):
    """The dtypes of the arrays that describe a state.

    If the state is a single array, this returns the dtype of that array.  If the state
    is a list of arrays, this returns a list containing the dtypes of the arrays.
    """
    return self._state_dtype

  @property
  def n_actions(self):
    """The number of possible actions that can be performed in this Environment."""
    return self._n_actions

  def reset(self):
    """Initialize the environment in preparation for doing calculations with it.

    This must be called before calling step() or querying the state.  You can call it
    again later to reset the environment back to its original state.
    """
    raise NotImplemented("Subclasses must implement this")

  def step(self, action):
    """Take a time step by performing an action.

    This causes the "state" and "terminated" properties to be updated.

    Parameters
    ----------
    action: object
      an object describing the action to take

    Returns
    -------
    the reward earned by taking the action, represented as a floating point number
    (higher values are better)
    """
    raise NotImplemented("Subclasses must implement this")


class GymEnvironment(Environment):
  """This is a convenience class for working with environments from OpenAI Gym."""

  def __init__(self, name):
    """Create an Environment wrapping the OpenAI Gym environment with a specified name."""
    import gym
    self.env = gym.make(name)
    self.name = name
    super(GymEnvironment, self).__init__(self.env.observation_space.shape,
                                         self.env.action_space.n)

  def reset(self):
    self._state = self.env.reset()
    self._terminated = False

  def step(self, action):
    self._state, reward, self._terminated, info = self.env.step(action)
    return reward

  def __deepcopy__(self, memo):
    return GymEnvironment(self.name)


class Policy(object):
  """A policy for taking actions within an environment.

  A policy is defined by a set of TensorGraph Layer objects that perform the
  necessary calculations.  There are many algorithms for reinforcement learning,
  and they differ in what values they require a policy to compute.  That makes
  it impossible to define a single interface allowing any policy to be optimized
  with any algorithm.  Instead, this interface just tries to be as flexible and
  generic as possible.  Each algorithm must document what values it expects
  create_layers() to return.

  Policy objects should be written to support pickling.  Many algorithms involve
  creating multiple copies of the Policy, possibly running in different processes
  or even on different computers.
  """

  def create_layers(self, state, **kwargs):
    """Create the TensorGraph Layers that define the policy.

    The arguments always include a list of Feature layers representing the current
    state of the environment (one layer for each array in the state).  Depending on
    the algorithm being used, other arguments might get passed as well.  It is up
    to each algorithm to document that.

    This method should construct and return a dict that maps strings to Layer
    objects.  Each algorithm must document what Layers it expects the policy to
    create.  If this method is called multiple times, it should create a new set
    of Layers every time.
    """
    raise NotImplemented("Subclasses must implement this")
