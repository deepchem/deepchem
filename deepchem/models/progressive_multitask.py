import numpy as np
import tensorflow as tf
from collections.abc import Sequence as SequenceCollection

import logging
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.keras_model import _StandardLoss
from tensorflow.keras.layers import Input, Dense, Dropout, ReLU, Concatenate, Add, Multiply, Softmax

logger = logging.getLogger(__name__)


class ProgressiveMultitaskRegressor(KerasModel):
    """Implements a progressive multitask neural network for regression.

    Progressive networks allow for multitask learning where each task
    gets a new column of weights. As a result, there is no exponential
    forgetting where previous tasks are ignored.

    References
    ----------
    See [1]_ for a full description of the progressive architecture

    .. [1] Rusu, Andrei A., et al. "Progressive neural networks." arXiv preprint
        arXiv:1606.04671 (2016).
    """

    def __init__(self,
                 n_tasks,
                 n_features,
                 alpha_init_stddevs=0.02,
                 layer_sizes=[1000],
                 weight_init_stddevs=0.02,
                 bias_init_consts=1.0,
                 weight_decay_penalty=0.0,
                 weight_decay_penalty_type="l2",
                 dropouts=0.5,
                 activation_fns=tf.nn.relu,
                 n_outputs=1,
                 **kwargs):
        """Creates a progressive network.

        Only listing parameters specific to progressive networks here.

        Parameters
        ----------
        n_tasks: int
            Number of tasks
        n_features: int
            Number of input features
        alpha_init_stddevs: list
            List of standard-deviations for alpha in adapter layers.
        layer_sizes: list
            the size of each dense layer in the network.  The length of this list determines the number of layers.
        weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight initialization of each layer.  The length
            of this list should equal len(layer_sizes)+1.  The final element corresponds to the output layer.
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        bias_init_consts: list or float
            the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes)+1.
            The final element corresponds to the output layer.  Alternatively this may be a single value instead of a list,
            in which case the same value is used for every layer.
        weight_decay_penalty: float
            the magnitude of the weight decay penalty to use
        weight_decay_penalty_type: str
            the type of penalty to use for weight decay, either 'l1' or 'l2'
        dropouts: list or float
            the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        activation_fns: list or object
            the Tensorflow activation function to apply to each layer.  The length of this list should equal
            len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
            same value is used for every layer.
        """

        if weight_decay_penalty != 0.0:
            raise ValueError('Weight decay is not currently supported')
        self.n_tasks = n_tasks
        self.n_features = n_features
        self.layer_sizes = layer_sizes
        self.alpha_init_stddevs = alpha_init_stddevs
        self.weight_init_stddevs = weight_init_stddevs
        self.bias_init_consts = bias_init_consts
        self.dropouts = dropouts
        self.activation_fns = activation_fns
        self.n_outputs = n_outputs

        n_layers = len(layer_sizes)
        if not isinstance(weight_init_stddevs, SequenceCollection):
            self.weight_init_stddevs = [weight_init_stddevs] * n_layers
        if not isinstance(alpha_init_stddevs, SequenceCollection):
            self.alpha_init_stddevs = [alpha_init_stddevs] * n_layers
        if not isinstance(bias_init_consts, SequenceCollection):
            self.bias_init_consts = [bias_init_consts] * n_layers
        if not isinstance(dropouts, SequenceCollection):
            self.dropouts = [dropouts] * n_layers
        if not isinstance(activation_fns, SequenceCollection):
            self.activation_fns = [activation_fns] * n_layers

        # Add the input features.
        mol_features = Input(shape=(n_features,))

        all_layers = {}
        outputs = []
        self._task_layers = []
        for task in range(self.n_tasks):
            task_layers = []
            for i in range(n_layers):
                if i == 0:
                    prev_layer = mol_features
                else:
                    prev_layer = all_layers[(i - 1, task)]
                    if task > 0:
                        lateral_contrib, trainables = self.add_adapter(
                            all_layers, task, i)
                        task_layers.extend(trainables)

                dense = Dense(
                    layer_sizes[i],
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        stddev=self.weight_init_stddevs[i]),
                    bias_initializer=tf.constant_initializer(
                        value=self.bias_init_consts[i]))
                layer = dense(prev_layer)
                task_layers.append(dense)

                if i > 0 and task > 0:
                    layer = Add()([layer, lateral_contrib])
                assert self.activation_fns[
                    i] is tf.nn.relu, "Only ReLU is supported"
                layer = ReLU()(layer)
                if self.dropouts[i] > 0.0:
                    layer = Dropout(self.dropouts[i])(layer)
                all_layers[(i, task)] = layer

            prev_layer = all_layers[(n_layers - 1, task)]
            dense = Dense(
                n_outputs,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=self.weight_init_stddevs[-1]),
                bias_initializer=tf.constant_initializer(
                    value=self.bias_init_consts[-1]))
            layer = dense(prev_layer)
            task_layers.append(dense)

            if task > 0:
                lateral_contrib, trainables = self.add_adapter(
                    all_layers, task, n_layers)
                task_layers.extend(trainables)
                layer = Add()([layer, lateral_contrib])
            output_layer = self.create_output(layer)
            outputs.append(output_layer)
            self._task_layers.append(task_layers)

        outputs = layers.Stack(axis=1)(outputs)
        model = tf.keras.Model(inputs=mol_features, outputs=outputs)
        super(ProgressiveMultitaskRegressor,
              self).__init__(model, self.create_loss(), **kwargs)

    def create_loss(self):
        return L2Loss()

    def create_output(self, layer):
        return layer

    def add_adapter(self, all_layers, task, layer_num):
        """Add an adapter connection for given task/layer combo"""
        i = layer_num
        prev_layers = []
        trainable_layers = []
        # Handle output layer
        if i < len(self.layer_sizes):
            layer_sizes = self.layer_sizes
            alpha_init_stddev = self.alpha_init_stddevs[i]
            weight_init_stddev = self.weight_init_stddevs[i]
            bias_init_const = self.bias_init_consts[i]
        elif i == len(self.layer_sizes):
            layer_sizes = self.layer_sizes + [self.n_outputs]
            alpha_init_stddev = self.alpha_init_stddevs[-1]
            weight_init_stddev = self.weight_init_stddevs[-1]
            bias_init_const = self.bias_init_consts[-1]
        else:
            raise ValueError("layer_num too large for add_adapter.")
        # Iterate over all previous tasks.
        for prev_task in range(task):
            prev_layers.append(all_layers[(i - 1, prev_task)])
        # prev_layers is a list with elements of size
        # (batch_size, layer_sizes[i-1])
        if len(prev_layers) == 1:
            prev_layer = prev_layers[0]
        else:
            prev_layer = Concatenate(axis=1)(prev_layers)
        alpha = layers.Variable(
            tf.random.truncated_normal((1,), stddev=alpha_init_stddev))
        trainable_layers.append(alpha)

        prev_layer = Multiply()([prev_layer, alpha([prev_layer])])
        dense1 = Dense(
            layer_sizes[i - 1],
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=weight_init_stddev),
            bias_initializer=tf.constant_initializer(value=bias_init_const))
        prev_layer = dense1(prev_layer)
        trainable_layers.append(dense1)

        dense2 = Dense(layer_sizes[i],
                       kernel_initializer=tf.keras.initializers.TruncatedNormal(
                           stddev=weight_init_stddev),
                       use_bias=False)
        prev_layer = dense2(prev_layer)
        trainable_layers.append(dense2)

        return prev_layer, trainable_layers

    def fit(self,
            dataset,
            nb_epoch=10,
            max_checkpoints_to_keep=5,
            checkpoint_interval=1000,
            deterministic=False,
            restore=False,
            **kwargs):
        for task in range(self.n_tasks):
            self.fit_task(dataset,
                          task,
                          nb_epoch=nb_epoch,
                          max_checkpoints_to_keep=max_checkpoints_to_keep,
                          checkpoint_interval=checkpoint_interval,
                          deterministic=deterministic,
                          restore=restore,
                          **kwargs)

    def fit_task(self,
                 dataset,
                 task,
                 nb_epoch=10,
                 max_checkpoints_to_keep=5,
                 checkpoint_interval=1000,
                 deterministic=False,
                 restore=False,
                 **kwargs):
        """Fit one task."""
        shape = dataset.get_shape()
        batch = [[np.zeros((self.batch_size,) + s[1:])] for s in shape]
        self._create_training_ops(batch)
        generator = self.default_generator(dataset,
                                           epochs=nb_epoch,
                                           deterministic=deterministic)
        variables = []
        for layer in self._task_layers[task]:
            variables += layer.trainable_variables
        loss = TaskLoss(self.model, self.create_loss(), task)
        self.fit_generator(generator,
                           max_checkpoints_to_keep,
                           checkpoint_interval,
                           restore,
                           variables=variables,
                           loss=loss)


class ProgressiveMultitaskClassifier(ProgressiveMultitaskRegressor):
    """Implements a progressive multitask neural network for classification.

    Progressive Networks: https://arxiv.org/pdf/1606.04671v3.pdf

    Progressive networks allow for multitask learning where each task
    gets a new column of weights. As a result, there is no exponential
    forgetting where previous tasks are ignored.

    """

    def __init__(self,
                 n_tasks,
                 n_features,
                 alpha_init_stddevs=0.02,
                 layer_sizes=[1000],
                 weight_init_stddevs=0.02,
                 bias_init_consts=1.0,
                 weight_decay_penalty=0.0,
                 weight_decay_penalty_type="l2",
                 dropouts=0.5,
                 activation_fns=tf.nn.relu,
                 **kwargs):
        n_outputs = 2
        super(ProgressiveMultitaskClassifier, self).__init__(
            n_tasks,
            n_features,
            alpha_init_stddevs=alpha_init_stddevs,
            layer_sizes=layer_sizes,
            weight_init_stddevs=weight_init_stddevs,
            bias_init_consts=bias_init_consts,
            weight_decay_penalty=weight_decay_penalty,
            weight_decay_penalty_type=weight_decay_penalty_type,
            dropouts=dropouts,
            activation_fns=activation_fns,
            n_outputs=n_outputs,
            **kwargs)

    def create_loss(self):
        return SparseSoftmaxCrossEntropy()

    def create_output(self, layer):
        return Softmax()(layer)


class TaskLoss(_StandardLoss):

    def __init__(self, model, loss, task):
        super(TaskLoss, self).__init__(model, loss)
        self.task = task

    def __call__(self, outputs, labels, weights):
        outputs = [t[:, self.task] for t in outputs]
        labels = [t[:, self.task] for t in labels]
        weights = [t[:, self.task] for t in weights]
        return super(TaskLoss, self).__call__(outputs, labels, weights)
