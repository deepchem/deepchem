"""Optimizers and related classes for use with TensorGraph."""

import math
from functools import partial
try:
    from deepchem.utils.optimizer_utils import LambOptimizer
except:
    pass
from typing import Dict, Union, Optional


class Optimizer(object):
    """An algorithm for optimizing a model.

    This is an abstract class.  Subclasses represent specific optimization algorithms.
    """

    def __init__(self, learning_rate: "Union[float, LearningRateSchedule]"):
        """This constructor should only be called by subclasses.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        """
        self.learning_rate = learning_rate

    def _create_tf_optimizer(self, global_step):
        """Construct a TensorFlow optimizer.

        Parameters
        ----------
        global_step: tensor
            a tensor containing the global step index during optimization, used for learning rate decay

        Returns
        -------
        a new TensorFlow optimizer implementing the algorithm
        """
        raise NotImplementedError("Subclasses must implement this")

    def _create_pytorch_optimizer(self, params):
        """Construct a PyTorch optimizer.

        Parameters
        ----------
        params: Iterable
            the model parameters to optimize

        Returns
        -------
        a new PyTorch optimizer implementing the algorithm
        """
        raise NotImplementedError("Subclasses must implement this")

    def _create_jax_optimizer(self):
        """Construct a Jax optimizer.

        Returns
        -------
        a new Optax optimizer optax.GradientTransformation implementing the algorithm
        """
        raise NotImplementedError("Subclasses must implement this")


class LearningRateSchedule(object):
    """A schedule for changing the learning rate over the course of optimization.

    This is an abstract class.  Subclasses represent specific schedules.
    """

    def _create_tf_tensor(self, global_step):
        """Construct a tensor that equals the learning rate.

        Parameters
        ----------
        global_step: tensor
            a tensor containing the global step index during optimization

        Returns
        -------
        a tensor that equals the learning rate
        """
        raise NotImplementedError("Subclasses must implement this")

    def _create_pytorch_schedule(self, optimizer):
        """Construct a PyTorch learning rate scheduler.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            the Optimizer whose learning rate will be modified

        Returns
        -------
        a PyTorch scheduler implementing the schedule
        """
        raise NotImplementedError("Subclasses must implement this")

    def _create_jax_schedule(self, learning_rate):
        """Construct a Jax learning rate scheduler using optax.

        Parameters
        ----------
        learning_rate: float
            the initial learning rate that will be modified

        Returns
        -------
        a optax scheduler implementing the schedule
        """
        raise NotImplementedError("Subclasses must implement this")


class AdaGrad(Optimizer):
    """The AdaGrad optimization algorithm.

    Adagrad is an optimizer with parameter-specific learning rates, which are
    adapted relative to how frequently a parameter gets updated during training.
    The more updates a parameter receives, the smaller the updates. See [1]_ for
    a full reference for the algorithm.

    References
    ----------
    .. [1] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient
        methods for online learning and stochastic optimization." Journal of machine
        learning research 12.7 (2011).
    """

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 initial_accumulator_value: float = 0.1,
                 epsilon: float = 1e-07):
        """Construct an AdaGrad optimizer.
        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        initial_accumulator_value: float
            a parameter of the AdaGrad algorithm
        epsilon: float
            a parameter of the AdaGrad algorithm

        """
        super(AdaGrad, self).__init__(learning_rate)
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

    def _create_tf_optimizer(self, global_step):
        import tensorflow as tf
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tf.keras.optimizers.legacy.Adagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=self.initial_accumulator_value,
            epsilon=self.epsilon)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.Adagrad(
            params,
            lr,
            initial_accumulator_value=self.initial_accumulator_value,
            eps=self.epsilon)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)

        process.append(
            optax.scale_by_rss(
                initial_accumulator_value=self.initial_accumulator_value,
                eps=self.epsilon))
        process.append(last_process)
        return optax.chain(*process)


class Adam(Optimizer):
    """The Adam optimization algorithm."""

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08,
                 weight_decay: float = 0):
        """Construct an Adam optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        beta1: float
            a parameter of the Adam algorithm
        beta2: float
            a parameter of the Adam algorithm
        epsilon: float
            a parameter of the Adam algorithm
        weight_decay: float
            L2 penalty - a parameter of the Adam algorithm
        """
        super(Adam, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def _create_tf_optimizer(self, global_step):
        import tensorflow as tf
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate,
                                               beta_1=self.beta1,
                                               beta_2=self.beta2,
                                               epsilon=self.epsilon)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.Adam(params,
                                lr=lr,
                                betas=(self.beta1, self.beta2),
                                eps=self.epsilon,
                                weight_decay=self.weight_decay)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            scheduler = self.learning_rate._create_jax_schedule()
            process.append(optax.scale_by_schedule(scheduler))
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)

        process.append(
            optax.scale_by_adam(b1=self.beta1, b2=self.beta2, eps=self.epsilon))
        process.append(last_process)
        return optax.chain(*process)


class SparseAdam(Optimizer):
    """The Sparse Adam optimization algorithm, also known as Lazy Adam.
    Sparse Adam is suitable for sparse tensors. It handles sparse updates more efficiently.
    It only updates moving-average accumulators for sparse variable indices that appear in the current batch, rather than updating the accumulators for all indices.
    """

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08):
        """Construct an Adam optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        beta1: float
            a parameter of the SparseAdam algorithm
        beta2: float
            a parameter of the SparseAdam algorithm
        epsilon: float
            a parameter of the SparseAdam algorithm
        """
        super(SparseAdam, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_tf_optimizer(self, global_step):
        import tensorflow_addons as tfa
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tfa.optimizers.LazyAdam(learning_rate=learning_rate,
                                       beta_1=self.beta1,
                                       beta_2=self.beta2,
                                       epsilon=self.epsilon)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.SparseAdam(params, lr, (self.beta1, self.beta2),
                                      self.epsilon)


class AdamW(Optimizer):
    """The AdamW optimization algorithm.
    AdamW is a variant of Adam, with improved weight decay.
    In Adam, weight decay is implemented as: weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
    In AdamW, weight decay is implemented as: weight_decay (float, optional) – weight decay coefficient (default: 1e-2)
    """

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 weight_decay: Union[float, LearningRateSchedule] = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08,
                 amsgrad: bool = False):
        """Construct an AdamW optimizer.
        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        weight_decay: float or LearningRateSchedule
            weight decay coefficient for AdamW
        beta1: float
            a parameter of the Adam algorithm
        beta2: float
            a parameter of the Adam algorithm
        epsilon: float
            a parameter of the Adam algorithm
        amsgrad: bool
            If True, will use the AMSGrad variant of AdamW (from "On the Convergence of Adam and Beyond"), else will use the original algorithm.
        """
        super(AdamW, self).__init__(learning_rate)
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def _create_tf_optimizer(self, global_step):
        import tensorflow_addons as tfa
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tfa.optimizers.AdamW(weight_decay=self.weight_decay,
                                    learning_rate=learning_rate,
                                    beta_1=self.beta1,
                                    beta_2=self.beta2,
                                    epsilon=self.epsilon,
                                    amsgrad=self.amsgrad)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.AdamW(params, lr, (self.beta1, self.beta2),
                                 self.epsilon, self.weight_decay, self.amsgrad)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            scheduler = self.learning_rate._create_jax_schedule()
            process.append(optax.scale_by_schedule(scheduler))
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)

        process.append(
            optax.scale_by_adam(b1=self.beta1,
                                b2=self.beta2,
                                eps=self.epsilon,
                                eps_root=0.0))
        process.append(optax.add_decayed_weights(self.weight_decay, None))
        process.append(last_process)
        return optax.chain(*process)


class RMSProp(Optimizer):
    """RMSProp Optimization algorithm."""

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 momentum: float = 0.0,
                 decay: float = 0.9,
                 epsilon: float = 1e-10):
        """Construct an RMSProp Optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning_rate used for optimization
        momentum: float, default 0.0
            a parameter of the RMSProp algorithm
        decay: float, default 0.9
            a parameter of the RMSProp algorithm
        epsilon: float, default 1e-10
            a parameter of the RMSProp algorithm
        """
        super(RMSProp, self).__init__(learning_rate)
        self.momentum = momentum
        self.decay = decay
        self.epsilon = epsilon

    def _create_tf_optimizer(self, global_step):
        import tensorflow as tf
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate,
                                                  momentum=self.momentum,
                                                  rho=self.decay,
                                                  epsilon=self.epsilon)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.RMSprop(params,
                                   lr,
                                   alpha=self.decay,
                                   eps=self.epsilon,
                                   momentum=self.momentum)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            scheduler = self.learning_rate._create_jax_schedule()
            process.append(optax.scale_by_schedule(scheduler))
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)

        process.append(
            optax.scale_by_rms(decay=self.decay,
                               eps=self.epsilon,
                               initial_scale=0.0))
        if self.momentum is not None or self.momentum != 0.0:
            process.append(optax.trace(decay=self.momentum, nesterov=False))
        process.append(last_process)
        return optax.chain(*process)


class GradientDescent(Optimizer):
    """The gradient descent optimization algorithm."""

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001):
        """Construct a gradient descent optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        """
        super(GradientDescent, self).__init__(learning_rate)

    def _create_tf_optimizer(self, global_step):
        import tensorflow as tf
        if isinstance(self.learning_rate, LearningRateSchedule):
            learning_rate = self.learning_rate._create_tf_tensor(global_step)
        else:
            learning_rate = self.learning_rate
        return tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

    def _create_pytorch_optimizer(self, params):
        import torch
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return torch.optim.SGD(params, lr)

    def _create_jax_optimizer(self):
        import optax
        process = []
        if isinstance(self.learning_rate, LearningRateSchedule):
            scheduler = self.learning_rate._create_jax_schedule()
            process.append(optax.scale_by_schedule(scheduler))
            last_process = optax.scale(-1.0)
        else:
            lr = self.learning_rate
            last_process = optax.scale(-1.0 * lr)
        process.append(last_process)
        return optax.chain(*process)


class ExponentialDecay(LearningRateSchedule):
    """A learning rate that decreases exponentially with the number of training steps."""

    def __init__(self,
                 initial_rate: float,
                 decay_rate: float,
                 decay_steps: int,
                 staircase: bool = True):
        """Create an exponentially decaying learning rate.

        The learning rate starts as initial_rate.  Every decay_steps training steps, it is multiplied by decay_rate.

        Parameters
        ----------
        initial_rate: float
            the initial learning rate
        decay_rate: float
            the base of the exponential
        decay_steps: int
            the number of training steps over which the rate decreases by decay_rate
        staircase: bool
            if True, the learning rate decreases by discrete jumps every decay_steps.
            if False, the learning rate decreases smoothly every step
        """
        self.initial_rate = initial_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase

    def _create_tf_tensor(self, global_step):
        import tensorflow as tf
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_rate,
            decay_rate=self.decay_rate,
            decay_steps=self.decay_steps,
            staircase=self.staircase)(global_step)

    def _create_pytorch_schedule(self, optimizer):
        import torch
        if self.staircase:
            return torch.optim.lr_scheduler.StepLR(optimizer, self.decay_steps,
                                                   self.decay_rate)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, math.pow(self.decay_rate, 1 / self.decay_steps))

    def _create_jax_schedule(self):
        import optax
        return optax.exponential_decay(init_value=self.initial_rate,
                                       transition_steps=self.decay_steps,
                                       decay_rate=self.decay_rate,
                                       staircase=self.staircase)


class LambdaLRWithWarmup(LearningRateSchedule):
    """A learning rate scheduler supporting warmup followed by cool down.

    Example
    -------
    >>> import torch
    >>> opt = Adam(learning_rate=5e-5)
    >>> lr_schedule = LambdaLRWithWarmup(initial_rate=5e-5,
    ...     num_training_steps=100, num_warmup_steps=10)
    >>> params = [torch.nn.Parameter(torch.Tensor([1.0]))]
    >>> optimizer = opt._create_pytorch_optimizer(params)
    >>> scheduler = lr_schedule._create_pytorch_schedule(optimizer)
    """

    def __init__(self,
                 initial_rate: float,
                 num_warmup_steps: int,
                 num_training_steps: Optional[int] = None,
                 warmup_type: str = 'linear'):
        """
        Parameters
        ----------
        initial_rate: float
            Initial learning rate
        num_warmup_steps: int
            Number of warmup steps
        num_training_steps: int
            Number of training steps - required for linear schedule.
        warmup_type: str, optional. default: linear
            When `linear`, creates a learning rate schedule that decreases linearly from
                the initial lr in the optimizer to 0.
            When `constant`, creates a constant learning rate preceded by a warmup period
                during which the learning rate increases linearly between 0 and the initial
                lr set in the optimizer.
        """
        assert warmup_type == 'linear' or 'constant', f'Warmup type {warmup_type} is not supported.'
        self.initial_rate = initial_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.warmup_type = warmup_type

    def _create_pytorch_schedule(self, optimizer):
        """Creates a PyTorch learning rate scheduler for the given optimizer.

        When the warmup type is linear, the method _linear_schedule_with_warmup
        is used to create a learning rate schedule such that the learning
        rate increases linearly from 0 to initial_rate and
        then cools down to 0 linearly.

        When the warmup type is constant, the method _constant_schedule_with_warmup
        is used to create a learning rate schedule such that the learning
        rate linearly increases form 0 to initial_rate and then stays constant.
        """

        def _linear_schedule_with_warmup(current_step: int, *,
                                         num_warmup_steps: int,
                                         num_training_steps: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps)))

        def _constant_schedule_with_warmup(current_step: int, *,
                                           num_warmup_steps: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

        if self.warmup_type == 'linear':
            f = partial(_linear_schedule_with_warmup,
                        num_warmup_steps=self.num_warmup_steps,
                        num_training_steps=self.num_training_steps)
        elif self.warmup_type == 'constant':
            f = partial(_constant_schedule_with_warmup,
                        num_warmup_steps=self.num_warmup_steps)

        import torch
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class PolynomialDecay(LearningRateSchedule):
    """A learning rate that decreases from an initial value to a final value over a fixed number of training steps."""

    def __init__(self,
                 initial_rate: float,
                 final_rate: float,
                 decay_steps: int,
                 power: float = 1.0):
        """Create a smoothly decaying learning rate.

        The learning rate starts as initial_rate.  It smoothly decreases to final_rate over decay_steps training steps.
        It decays as a function of (1-step/decay_steps)**power.  Once the final rate is reached, it remains there for
        the rest of optimization.

        Parameters
        ----------
        initial_rate: float
            the initial learning rate
        final_rate: float
            the final learning rate
        decay_steps: int
            the number of training steps over which the rate decreases from initial_rate to final_rate
        power: float
            the exponent controlling the shape of the decay
        """
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.decay_steps = decay_steps
        self.power = power

    def _create_tf_tensor(self, global_step):
        import tensorflow as tf
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.initial_rate,
            end_learning_rate=self.final_rate,
            decay_steps=self.decay_steps,
            power=self.power)(global_step)

    def _create_pytorch_schedule(self, optimizer):

        def f(step):
            t = min(step, self.decay_steps) / self.decay_steps
            return ((self.initial_rate - self.final_rate) *
                    (1 - t)**self.power) + self.final_rate

        import torch
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    def _create_jax_schedule(self):
        import optax
        return optax.polynomial_schedule(init_value=self.initial_rate,
                                         end_value=self.final_rate,
                                         power=self.power,
                                         transition_steps=self.decay_steps)


class LinearCosineDecay(LearningRateSchedule):
    """Applies linear cosine decay to the learning rate"""

    def __init__(self,
                 initial_rate: float,
                 decay_steps: int,
                 alpha: float = 0.0,
                 beta: float = 0.001,
                 num_periods: float = 0.5):
        """
        Parameters
        ----------
        learning_rate : float
        initial learning rate
        decay_steps : int
        number of steps to decay over
        num_periods : number of periods in the cosine part of the decay
        """

        self.initial_rate = initial_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.beta = beta
        self.num_periods = num_periods

    def _create_tf_tensor(self, global_step):
        import tensorflow as tf
        return tf.compat.v1.train.linear_cosine_decay(
            learning_rate=self.initial_rate,
            global_step=global_step,
            decay_steps=self.decay_steps,
            alpha=self.alpha,
            beta=self.beta,
            num_periods=self.num_periods)

    def _create_pytorch_schedule(self, optimizer):

        def f(step):
            t = min(step, self.decay_steps) / self.decay_steps
            linear_decay = 1 - t
            cosine_decay = 0.5 * (1 +
                                  math.cos(math.pi * 2 * self.num_periods * t))
            decayed = (self.alpha + linear_decay) * cosine_decay + self.beta
            return self.initial_rate * decayed

        import torch
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    def _create_jax_schedule(self):
        import optax
        return optax.cosine_decay_schedule(init_value=self.initial_rate,
                                           decay_steps=self.decay_steps,
                                           alpha=self.alpha)


class PiecewiseConstantSchedule(LearningRateSchedule):
    """Applies scheduler which multiplies by a constant factor on the boundaries"""

    def __init__(self,
                 initial_rate: float,
                 boundaries_and_scales: Optional[Dict[int, float]] = None):
        """
        Parameters
        ----------
        init_value : float
            initial learning rate
        boundaries_and_scales:
            A map from boundaries b_i to non-negative scaling factors f_i. For any step
            count s, the schedule returns init_v scaled by the product of all factors f_i
            such that b_i < s.
        """
        self.initial_rate = initial_rate
        self.boundaries_and_scales = boundaries_and_scales

    def _create_jax_schedule(self):
        import optax
        return optax.piecewise_constant_schedule(
            init_value=self.initial_rate,
            boundaries_and_scales=self.boundaries_and_scales)


class KFAC(Optimizer):
    """The Second order gradient optimiation algorithm which uses an approximation to calculate the inverse of the Fischer matrrix"""

    def __init__(self, **kwargs):
        """
        Parameters:
        -----------
        model: torch.nn.Module
            The model to be optimized.
        lr: float (default: 0.001)
            Learning rate for the optimizer.
        momentum: float (default: 0.9)
            Momentum for the optimizer.
        stat_decay: float (default: 0.95)
            Decay rate for the update of covariance matrix with mean.
        damping: float (default: 0.001)
            damping factor for the update of covariance matrix.
        kl_clip: float (default: 0.001)
            Clipping value for the update of covariance matrix.
        weight_decay: float (default: 0)
            weight decay for the optimizer.
        Tcov: int (default: 10)
            The number of steps to update the covariance matrix.
        Tinv: int (default: 100)
            The number of steps to calculate the inverse of covariance matrix.
        batch_averaged: bool (default: True)
            States whether to use batch averaged covariance matrix.
        mean: bool (default: False)
            States whether to use mean centered covariance matrix.
        """
        self.kwargs = kwargs

    def _create_pytorch_optimizer(self):
        from deepchem.models.torch_models.kfac_optimizer import KFACOptimizer
        if isinstance(self.learning_rate, LearningRateSchedule):
            self.kwargs['lr'] = self.learning_rate.initial_rate
        else:
            self.kwargs['lr'] = self.learning_rate
        return KFACOptimizer([self.kwargs])


class Lamb(Optimizer):
    """The Lamb optimization algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`__.
    """

    def __init__(self,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-08,
                 weight_decay: float = 0):
        """Construct an Adam optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
            the learning rate to use for optimization
        beta1: float
            a parameter of the Lamb algorithm
        beta2: float
            a parameter of the Lamb algorithm
        epsilon: float
            a parameter of the Lamb algorithm
        weight_decay: float
            L2 penalty - a parameter of the Lamb algorithm

        Examples
        --------
        >>> import deepchem as dc
        >>> from deepchem.models import GCNModel
        >>> # preparing dataset
        >>> smiles = ["C1CCC1", "CCC"]
        >>> labels = [0., 1.]
        >>> featurizer = dc.feat.MolGraphConvFeaturizer()
        >>> X = featurizer.featurize(smiles)
        >>> dataset = dc.data.NumpyDataset(X=X, y=labels)
        >>> # training model
        >>> model = GCNModel(mode='classification', n_tasks=1,
        ...                  batch_size=16, learning_rate=0.001,
        ...                  optimizers=dc.models.optimizers.Lamb(learning_rate=0.01))
        >>> loss = model.fit(dataset, nb_epoch=5)

        References
        ----------
        Yang You and Jing Li and Sashank Reddi, et, al., "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes", 2020,
        https://arxiv.org/abs/1904.00962
        """
        super(Lamb, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def _create_pytorch_optimizer(self, params):
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return LambOptimizer(params,
                             lr=lr,
                             betas=(self.beta1, self.beta2),
                             eps=self.epsilon,
                             weight_decay=self.weight_decay)
