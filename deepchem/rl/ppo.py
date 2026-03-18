"""Proximal Policy Optimization (PPO) algorithm for reinforcement learning."""
import copy
import time

from collections.abc import Sequence as SequenceCollection
from multiprocessing.dummy import Pool

import numpy as np
import tensorflow as tf

from deepchem.models import KerasModel
from deepchem.models.optimizers import Adam


class PPOLoss(object):
    """This class computes the loss function for PPO."""

    def __init__(self, value_weight, entropy_weight, clipping_width,
                 action_prob_index, value_index):
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.clipping_width = clipping_width
        self.action_prob_index = action_prob_index
        self.value_index = value_index

    def __call__(self, outputs, labels, weights):
        prob = outputs[self.action_prob_index]
        value = outputs[self.value_index]
        reward, advantage, old_prob = weights
        action = labels[0]
        advantage = tf.expand_dims(advantage, axis=1)
        machine_eps = np.finfo(np.float32).eps
        prob += machine_eps
        old_prob += machine_eps
        ratio = tf.reduce_sum(action * prob, axis=1) / old_prob
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clipping_width,
                                         1 + self.clipping_width)
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage, clipped_ratio * advantage))
        value_loss = tf.reduce_mean(tf.square(reward - value))
        entropy = -tf.reduce_mean(
            tf.reduce_sum(prob * tf.math.log(prob), axis=1))
        return policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy


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
    have been received for taking the specified actions from those states.  The output arrays may be
    shorter than the input ones, if the modified rollout would have terminated sooner.
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
            the Policy to optimize.  It must have outputs with the names 'action_prob'
            and 'value', corresponding to the action probabilities and value estimate
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
        self._state_is_list = isinstance(env.state_shape[0], SequenceCollection)
        if optimizer is None:
            self._optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        else:
            self._optimizer = optimizer
        output_names = policy.output_names
        self._value_index = output_names.index('value')
        self._action_prob_index = output_names.index('action_prob')
        self._rnn_final_state_indices = [
            i for i, n in enumerate(output_names) if n == 'rnn_state'
        ]
        self._rnn_states = policy.rnn_initial_states
        if len(self._rnn_states) > 0 and batch_size != 0:
            raise ValueError(
                'Cannot batch rollouts when the policy contains a recurrent layer.  Set batch_size to 0.'
            )
        self._model = self._build_model(model_dir)
        self._checkpoint = tf.train.Checkpoint()
        self._checkpoint.save_counter  # Ensure the variable has been created
        self._checkpoint.listed = self._model.model.trainable_variables

    def _build_model(self, model_dir):
        """Construct a KerasModel containing the policy and loss calculations."""
        policy_model = self._policy.create_model()
        loss = PPOLoss(self.value_weight, self.entropy_weight,
                       self.clipping_width, self._action_prob_index,
                       self._value_index)
        model = KerasModel(policy_model,
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
        step_count = 0
        workers = []
        for i in range(self.optimization_rollouts):
            workers.append(_Worker(self, i))
        if restore:
            self.restore()
        pool = Pool()
        manager = tf.train.CheckpointManager(self._checkpoint,
                                             self._model.model_dir,
                                             max_checkpoints_to_keep)
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

                    # Build the inputs and run the optimizer.

                    state_arrays = [np.stack(s) for s in state_arrays]
                    inputs = state_arrays + [
                        np.expand_dims(s, axis=0) for s in initial_rnn_states
                    ]
                    self._apply_gradients(inputs, actions_matrix,
                                          discounted_rewards, advantages,
                                          action_prob)

            # Update the number of steps taken so far and perform checkpointing.

            new_steps = sum(len(r[3]) for r in rollouts)
            if self.use_hindsight:
                new_steps /= 2
            step_count += new_steps
            if step_count >= total_steps or time.time(
            ) >= checkpoint_time + checkpoint_interval:
                manager.save()
                checkpoint_time = time.time()

    @tf.function(experimental_relax_shapes=True)
    def _apply_gradients(self, inputs, actions_matrix, discounted_rewards,
                         advantages, action_prob):
        """Compute the gradient of the loss function for a batch and update the model."""
        vars = self._model.model.trainable_variables
        with tf.GradientTape() as tape:
            outputs = self._model.model(inputs)
            loss = self._model._loss_fn(
                outputs, [actions_matrix],
                [discounted_rewards, advantages, action_prob])
        gradients = tape.gradient(loss, vars)
        self._model._tf_optimizer.apply_gradients(zip(gradients, vars))

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
        return [
            results[i] for i in (self._action_prob_index, self._value_index)
        ]

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
        action_prob = outputs[self._action_prob_index]
        if deterministic:
            return action_prob.argmax()
        else:
            action_prob = action_prob.flatten()
            return np.random.choice(np.arange(len(action_prob)), p=action_prob)

    def _create_feed_dict(self, state, use_saved_states):
        """Create a feed dict for use by predict() or select_action()."""
        if use_saved_states:
            state = state + list(self._rnn_states)
        else:
            state = state + list(self._policy.rnn_initial_states)
        return dict((f, np.expand_dims(s, axis=0))
                    for f, s in zip(self._model._input_placeholders, state))


class _Worker(object):
    """A Worker object is created for each training thread."""

    def __init__(self, ppo, index):
        self.ppo = ppo
        self.index = index
        self.scope = 'worker%d' % index
        self.env = copy.deepcopy(ppo._env)
        self.env.reset()
        self.model = ppo._build_model(None)
        self.rnn_states = ppo._policy.rnn_initial_states

    def run(self):
        rollouts = []
        local_vars = self.model.model.trainable_variables
        global_vars = self.ppo._model.model.trainable_variables
        for v1, v2 in zip(local_vars, global_vars):
            v1.assign(v2)
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
            results = self._compute_model(
                self._create_model_inputs(state, self.rnn_states))
            results = [r.numpy() for r in results]
            value = results[self.ppo._value_index]
            probabilities = np.squeeze(results[self.ppo._action_prob_index])
            self.rnn_states = [
                np.squeeze(results[i], 0)
                for i in self.ppo._rnn_final_state_indices
            ]
            action = self.ppo._select_action_from_outputs(results, False)
            actions.append(action)
            action_prob.append(probabilities[action])
            values.append(float(value))
            rewards.append(self.env.step(action))

        # Compute an estimate of the reward for the rest of the episode.

        if not self.env.terminated:
            results = self._compute_model(
                self._create_model_inputs(self.env.state, self.rnn_states))
            final_value = self.ppo.discount_factor * results[
                self.ppo._value_index].numpy()[0]
            try:
                final_value = final_value.tolist()[0]
            except:
                pass
        else:
            final_value = 0.0
        values.append(final_value)
        if self.env.terminated:
            self.env.reset()
            self.rnn_states = self.ppo._policy.rnn_initial_states
        return states, np.array(actions, dtype=np.int32), np.array(
            action_prob, dtype=np.float32), np.array(
                rewards, dtype=np.float32), np.array(values, dtype=np.float32)

    def process_rollout(self, states, actions, action_prob, rewards, values,
                        initial_rnn_states):
        """Construct the arrays needed for training."""

        # Compute the discounted rewards and advantages.

        discounted_rewards = rewards.copy()
        discounted_rewards[-1] += values[-1]
        advantages = rewards - values[:-1] + self.ppo.discount_factor * np.array(
            values[1:])
        for j in range(len(rewards) - 1, 0, -1):
            discounted_rewards[
                j - 1] += self.ppo.discount_factor * discounted_rewards[j]
            advantages[
                j -
                1] += self.ppo.discount_factor * self.ppo.advantage_lambda * advantages[
                    j]

        # Convert the actions to one-hot.

        n_actions = self.env.n_actions
        actions_matrix = []
        for action in actions:
            a = np.zeros(n_actions, np.float32)
            a[action] = 1.0
            actions_matrix.append(a)
        actions_matrix = np.array(actions_matrix, dtype=np.float32)

        # Rearrange the states into the proper set of arrays.

        if self.ppo._state_is_list:
            state_arrays = [[] for i in range(len(self.env.state_shape))]
            for state in states:
                for j in range(len(state)):
                    state_arrays[j].append(state[j])
        else:
            state_arrays = [states]

        # Return the processed arrays.

        return (initial_rnn_states, state_arrays, discounted_rewards,
                actions_matrix, action_prob, advantages)

    def process_rollout_with_hindsight(self, states, actions,
                                       initial_rnn_states):
        """Create a new rollout by applying hindsight to an existing one, then process it."""
        hindsight_states, rewards = self.env.apply_hindsight(
            states, actions, states[-1])
        if self.ppo._state_is_list:
            state_arrays = [[] for i in range(len(self.env.state_shape))]
            for state in hindsight_states:
                for j in range(len(state)):
                    state_arrays[j].append(state[j])
        else:
            state_arrays = [hindsight_states]
        state_arrays += initial_rnn_states
        state_arrays = [np.stack(s) for s in state_arrays]
        inputs = state_arrays + [
            np.expand_dims(s, axis=0) for s in initial_rnn_states
        ]
        outputs = self._compute_model(inputs)
        values = outputs[self.ppo._value_index].numpy()
        values = np.append(values.flatten(), 0.0)
        probabilities = outputs[self.ppo._action_prob_index].numpy()
        actions = actions[:len(rewards)]
        action_prob = probabilities[np.arange(len(actions)), actions]
        return self.process_rollout(hindsight_states, actions, action_prob,
                                    np.array(rewards, dtype=np.float32),
                                    np.array(values, dtype=np.float32),
                                    initial_rnn_states)

    def _create_model_inputs(self, state, rnn_states):
        """Create the inputs to the model for use during a rollout."""
        if not self.ppo._state_is_list:
            state = [state]
        state = state + rnn_states
        return [np.expand_dims(s, axis=0) for s in state]

    @tf.function(experimental_relax_shapes=True)
    def _compute_model(self, inputs):
        return self.model.model(inputs)
