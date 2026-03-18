# flake8: noqa
import logging

logger = logging.getLogger(__name__)
try:
    from deepchem.metalearning.maml import MAML
    from deepchem.metalearning import MAML
except ModuleNotFoundError as e:
    logger.warning(
        f'Skipped loading some PyTorch models, missing a dependency. {e}')


class MetaLearner(object):
    """Model and data to which the MAML algorithm can be applied.

    To use MAML, create a subclass of this defining the learning problem to solve.
    It consists of a model that can be trained to perform many different tasks, and
    data for training it on a large (possibly infinite) set of different tasks.
    """

    def compute_model(self, inputs, variables, training):
        """Compute the model for a set of inputs and variables.

        Parameters
        ----------
        inputs: list of tensors
            the inputs to the model
        variables: list of tensors
            the values to use for the model's variables.  This might be the actual
            variables (as returned by the MetaLearner's variables property), or
            alternatively it might be the values of those variables after one or more
            steps of gradient descent for the current task.
        training: bool
            indicates whether the model is being invoked for training or prediction

        Returns
        -------
        (loss, outputs) where loss is the value of the model's loss function, and
        outputs is a list of the model's outputs
        """
        raise NotImplementedError("Subclasses must implement this")

    @property
    def variables(self):
        """Get the list of variables to train."""
        raise NotImplementedError("Subclasses must implement this")

    def select_task(self):
        """Select a new task to train on.

        If there is a fixed set of training tasks, this will typically cycle through them.
        If there are infinitely many training tasks, this can simply select a new one each
        time it is called.
        """
        raise NotImplementedError("Subclasses must implement this")

    def get_batch(self):
        """Get a batch of data for training.

        This should return the data as a list of arrays, one for each of the model's
        inputs.  This will usually be called twice for each task, and should
        return a different batch on each call.
        """
        raise NotImplementedError("Subclasses must implement this")

    def parameters(self):
        """Returns an iterator over the MetaLearner parameters."""
        raise NotImplementedError("Subclasses must implement this")
