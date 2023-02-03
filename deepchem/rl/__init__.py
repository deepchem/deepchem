"""Interface for reinforcement learning."""
try:
    from deepchem.rl.a2c import A2C  # noqa: F401
    from deepchem.rl.ppo import PPO  # noqa: F401
except ModuleNotFoundError:
    pass


class Environment(object):
    """An environment in which an actor performs actions to accomplish a task.

    An environment has a current state, which is represented as either a single NumPy
    array, or optionally a list of NumPy arrays.  When an action is taken, that causes
    the state to be updated.  The environment also computes a reward for each action,
    and reports when the task has been terminated (meaning that no more actions may
    be taken).

    Two types of actions are supported.  For environments with discrete action spaces,
    the action is an integer specifying the index of the action to perform (out of a
    fixed list of possible actions).  For environments with continuous action spaces,
    the action is a NumPy array.

    Environment objects should be written to support pickle and deepcopy operations.
    Many algorithms involve creating multiple copies of the Environment, possibly
    running in different processes or even on different computers.
    """

    def __init__(self,
                 state_shape,
                 n_actions=None,
                 state_dtype=None,
                 action_shape=None):
        """Subclasses should call the superclass constructor in addition to doing their own initialization.

        A value should be provided for either n_actions (for discrete action spaces)
        or action_shape (for continuous action spaces), but not both.

        Parameters
        ----------
        state_shape: tuple or list of tuples
            the shape(s) of the array(s) making up the state
        n_actions: int
            the number of discrete actions that can be performed.  If the action space
            is continuous, this should be None.
        state_dtype: dtype or list of dtypes
            the type(s) of the array(s) making up the state.  If this is None, all
            arrays are assumed to be float32.
        action_shape: tuple
            the shape of the array describing an action.  If the action space
            is discrete, this should be none.
        """
        self._state_shape = state_shape
        self._n_actions = n_actions
        self._action_shape = action_shape
        self._state = None
        self._terminated = None
        if state_dtype is None:
            # Assume all arrays are float32.
            import numpy
            try:
                from collections.abc import Sequence as SequenceCollection
            except:
                from collections import Sequence as SequenceCollection
            if isinstance(state_shape[0], SequenceCollection):
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
        """The number of possible actions that can be performed in this Environment.

        If the environment uses a continuous action space, this returns None.
        """
        return self._n_actions

    @property
    def action_shape(self):
        """The expected shape of NumPy arrays representing actions.

        If the environment uses a discrete action space, this returns None.
        """
        return self._action_shape

    def reset(self):
        """Initialize the environment in preparation for doing calculations with it.

        This must be called before calling step() or querying the state.  You can call it
        again later to reset the environment back to its original state.
        """
        raise NotImplementedError("Subclasses must implement this")

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
        raise NotImplementedError("Subclasses must implement this")


class GymEnvironment(Environment):
    """This is a convenience class for working with environments from OpenAI Gym."""

    def __init__(self, name):
        """Create an Environment wrapping the OpenAI Gym environment with a specified name."""
        import gym
        self.env = gym.make(name)
        self.name = name
        space = self.env.action_space
        if 'n' in dir(space):
            super(GymEnvironment,
                  self).__init__(self.env.observation_space.shape, space.n)
        else:
            super(GymEnvironment,
                  self).__init__(self.env.observation_space.shape,
                                 action_shape=space.shape)

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

    A policy is defined by a tf.keras.Model that takes the current state as input
    and performs the necessary calculations.  There are many algorithms for
    reinforcement learning, and they differ in what values they require a policy to
    compute.  That makes it impossible to define a single interface allowing any
    policy to be optimized with any algorithm.  Instead, this interface just tries
    to be as flexible and generic as possible.  Each algorithm must document what
    values it expects the model to output.

    Special handling is needed for models that include recurrent layers.  In that
    case, the model has its own internal state which the learning algorithm must
    be able to specify and query.  To support this, the Policy must do three things:

    1. The Model must take additional inputs that specify the initial states of
        all its recurrent layers.  These will be appended to the list of arrays
        specifying the environment state.

    2. The Model must also return the final states of all its recurrent layers as
        outputs.

    3. The constructor argument rnn_initial_states must be specified to define
        the states to use for the Model's recurrent layers at the start of a new
        rollout.

    Policy objects should be written to support pickling.  Many algorithms involve
    creating multiple copies of the Policy, possibly running in different processes
    or even on different computers.
    """

    def __init__(self, output_names, rnn_initial_states=[]):
        """Subclasses should call the superclass constructor in addition to doing
        their own initialization.

        Parameters
        ----------
        output_names: list of strings
            the names of the Model's outputs, in order.  It is up to each reinforcement
            learning algorithm to document what outputs it expects policies to compute.
            Outputs that return the final states of recurrent layers should have the
            name 'rnn_state'.
        rnn_initial_states: list of NumPy arrays
            the initial states of the Model's recurrent layers at the start of a new
            rollout
        """
        self.output_names = output_names
        self.rnn_initial_states = rnn_initial_states

    def create_model(self, **kwargs):
        """Construct and return a tf.keras.Model that computes the policy.

        The inputs to the model consist of the arrays representing the current state
        of the environment, followed by the initial states for all recurrent layers.
        Depending on the algorithm being used, other inputs might get passed as
        well.  It is up to each algorithm to document that.
        """
        raise NotImplementedError("Subclasses must implement this")
