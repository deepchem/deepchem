import deepchem as dc
import deepchem.models.optimizers as optimizers
import unittest
import pytest

try:
    import tensorflow as tf
    has_tensorflow = True
except:
    has_tensorflow = False

try:
    import tensorflow_addons as tfa
    has_tensorflow_addons = True
except:
    has_tensorflow_addons = False

try:
    import torch
    from deepchem.utils.optimizer_utils import LambOptimizer
    has_pytorch = True
except:
    has_pytorch = False

try:
    import jax  # noqa: F401
    import optax
    has_jax = True
except:
    has_jax = False


class TestOptimizers(unittest.TestCase):
    """Test optimizers and related classes."""

    @pytest.mark.tensorflow
    def test_adam_tf(self):
        """Test creating an Adam optimizer."""
        opt = optimizers.Adam(learning_rate=0.01)
        global_step = tf.Variable(0)
        tfopt = opt._create_tf_optimizer(global_step)
        assert isinstance(tfopt, tf.keras.optimizers.legacy.Adam)

    @pytest.mark.torch
    def test_adam_pytorch(self):
        """Test creating an Adam optimizer."""
        opt = optimizers.Adam(learning_rate=0.01)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        assert isinstance(torchopt, torch.optim.Adam)

    @pytest.mark.jax
    def test_adam_jax(self):
        """Test creating an Adam optimizer."""
        opt = optimizers.Adam(learning_rate=0.01)
        jaxopt = opt._create_jax_optimizer()
        assert isinstance(jaxopt, optax.GradientTransformation)

    @pytest.mark.tensorflow
    def test_adamw_tf(self):
        """Test creating an AdamW optimizer."""
        opt = optimizers.AdamW(learning_rate=0.01)
        global_step = tf.Variable(0)
        tfopt = opt._create_tf_optimizer(global_step)
        assert isinstance(tfopt, tfa.optimizers.AdamW)

    @pytest.mark.torch
    def test_adamw_pytorch(self):
        """Test creating an AdamW optimizer."""
        opt = optimizers.AdamW(learning_rate=0.01)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        assert isinstance(torchopt, torch.optim.AdamW)

    @pytest.mark.jax
    def test_adamw_jax(self):
        """Test creating an AdamW optimizer."""
        opt = optimizers.AdamW(learning_rate=0.01)
        jaxopt = opt._create_jax_optimizer()
        assert isinstance(jaxopt, optax.GradientTransformation)

    @pytest.mark.tensorflow
    def test_sparseadam_tf(self):
        """Test creating a SparseAdam optimizer."""
        opt = optimizers.SparseAdam(learning_rate=0.01)
        global_step = tf.Variable(0)
        tfopt = opt._create_tf_optimizer(global_step)
        assert isinstance(tfopt, tfa.optimizers.LazyAdam)

    @pytest.mark.torch
    def test_sparseadam_pytorch(self):
        """Test creating a SparseAdam optimizer."""
        opt = optimizers.SparseAdam(learning_rate=0.01)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        assert isinstance(torchopt, torch.optim.SparseAdam)

    @pytest.mark.tensorflow
    def test_adagrad_tf(self):
        """Test creating an AdaGrad optimizer."""
        opt = optimizers.AdaGrad(learning_rate=0.01)
        global_step = tf.Variable(0)
        tfopt = opt._create_tf_optimizer(global_step)
        assert isinstance(tfopt, tf.keras.optimizers.legacy.Adagrad)

    @pytest.mark.torch
    def test_adagrad_pytorch(self):
        """Test creating an AdaGrad optimizer."""
        opt = optimizers.AdaGrad(learning_rate=0.01)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        assert isinstance(torchopt, torch.optim.Adagrad)

    @pytest.mark.jax
    def test_adagrad_jax(self):
        """Test creating an AdaGrad optimizer."""
        opt = optimizers.AdaGrad(learning_rate=0.01)
        jaxopt = opt._create_jax_optimizer()
        assert isinstance(jaxopt, optax.GradientTransformation)

    @pytest.mark.tensorflow
    def test_rmsprop_tf(self):
        """Test creating an RMSProp Optimizer."""
        opt = optimizers.RMSProp(learning_rate=0.01)
        global_step = tf.Variable(0)
        tfopt = opt._create_tf_optimizer(global_step)
        assert isinstance(tfopt, tf.keras.optimizers.legacy.RMSprop)

    @pytest.mark.torch
    def test_rmsprop_pytorch(self):
        """Test creating an RMSProp Optimizer."""
        opt = optimizers.RMSProp(learning_rate=0.01)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        assert isinstance(torchopt, torch.optim.RMSprop)

    @pytest.mark.jax
    def test_rmsprop_jax(self):
        """Test creating an RMSProp Optimizer."""
        opt = optimizers.RMSProp(learning_rate=0.01)
        jaxopt = opt._create_jax_optimizer()
        assert isinstance(jaxopt, optax.GradientTransformation)

    @pytest.mark.tensorflow
    def test_gradient_descent_tf(self):
        """Test creating a Gradient Descent optimizer."""
        opt = optimizers.GradientDescent(learning_rate=0.01)
        global_step = tf.Variable(0)
        tfopt = opt._create_tf_optimizer(global_step)
        assert isinstance(tfopt, tf.keras.optimizers.legacy.SGD)

    @pytest.mark.torch
    def test_gradient_descent_pytorch(self):
        """Test creating a Gradient Descent optimizer."""
        opt = optimizers.GradientDescent(learning_rate=0.01)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        assert isinstance(torchopt, torch.optim.SGD)

    @pytest.mark.jax
    def test_gradient_descent_jax(self):
        """Test creating an Gradient Descent Optimizer."""
        opt = optimizers.GradientDescent(learning_rate=0.01)
        jaxopt = opt._create_jax_optimizer()
        assert isinstance(jaxopt, optax.GradientTransformation)

    @pytest.mark.tensorflow
    def test_exponential_decay_tf(self):
        """Test creating an optimizer with an exponentially decaying learning rate."""
        rate = optimizers.ExponentialDecay(initial_rate=0.001,
                                           decay_rate=0.99,
                                           decay_steps=10000)
        opt = optimizers.Adam(learning_rate=rate)
        global_step = tf.Variable(0)
        _ = opt._create_tf_optimizer(global_step)

    @pytest.mark.torch
    def test_exponential_decay_pytorch(self):
        """Test creating an optimizer with an exponentially decaying learning rate."""
        rate = optimizers.ExponentialDecay(initial_rate=0.001,
                                           decay_rate=0.99,
                                           decay_steps=10000)
        opt = optimizers.Adam(learning_rate=rate)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        _ = rate._create_pytorch_schedule(torchopt)

    @pytest.mark.torch
    def test_lambda_lr_with_warmup(self):
        opt = optimizers.Adam(learning_rate=5e-5)
        lr_schedule = optimizers.LambdaLRWithWarmup(initial_rate=5e-5,
                                                    num_training_steps=100_000 *
                                                    10,
                                                    num_warmup_steps=10_000)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        _ = lr_schedule._create_pytorch_schedule(torchopt)

    @pytest.mark.jax
    def test_exponential_decay_jax(self):
        """Test creating an optimizer with an exponentially decaying learning rate."""
        rate = optimizers.ExponentialDecay(initial_rate=0.001,
                                           decay_rate=0.99,
                                           decay_steps=10000)
        opt = optimizers.Adam(learning_rate=rate)
        jaxopt = opt._create_jax_optimizer()
        assert isinstance(jaxopt, optax.GradientTransformation)

    @pytest.mark.tensorflow
    def test_polynomial_decay_tf(self):
        """Test creating an optimizer with a polynomially decaying learning rate."""
        rate = optimizers.PolynomialDecay(initial_rate=0.001,
                                          final_rate=0.0001,
                                          decay_steps=10000)
        opt = optimizers.Adam(learning_rate=rate)
        global_step = tf.Variable(0)
        _ = opt._create_tf_optimizer(global_step)

    @pytest.mark.torch
    def test_polynomial_decay_pytorch(self):
        """Test creating an optimizer with a polynomially decaying learning rate."""
        rate = optimizers.PolynomialDecay(initial_rate=0.001,
                                          final_rate=0.0001,
                                          decay_steps=10000)
        opt = optimizers.Adam(learning_rate=rate)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        _ = rate._create_pytorch_schedule(torchopt)

    @pytest.mark.jax
    def test_polynomial_decay_jax(self):
        """Test creating an optimizer with a polynomially decaying learning rate."""
        rate = optimizers.PolynomialDecay(initial_rate=0.001,
                                          final_rate=0.0001,
                                          decay_steps=10000)
        opt = optimizers.Adam(learning_rate=rate)
        jaxopt = opt._create_jax_optimizer()
        assert isinstance(jaxopt, optax.GradientTransformation)

    @pytest.mark.tensorflow
    def test_linearCosine_decay_tf(self):
        """test creating an optimizer with a linear cosine decay to the learning rate"""
        rate = optimizers.LinearCosineDecay(initial_rate=0.1, decay_steps=10000)
        opt = optimizers.Adam(learning_rate=rate)
        global_step = tf.Variable(0)
        _ = opt._create_tf_optimizer(global_step)

    @pytest.mark.torch
    def test_linearCosine_decay_pytorch(self):
        """test creating an optimizer with a linear cosine decay to the learning rate"""
        rate = optimizers.LinearCosineDecay(initial_rate=0.1, decay_steps=10000)
        opt = optimizers.Adam(learning_rate=rate)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        _ = rate._create_pytorch_schedule(torchopt)

    @pytest.mark.jax
    def test_linearCosine_decay_jax(self):
        """test creating an optimizer with a linear cosine decay to the learning rate"""
        rate = optimizers.LinearCosineDecay(initial_rate=0.1, decay_steps=10000)
        opt = optimizers.Adam(learning_rate=rate)
        jaxopt = opt._create_jax_optimizer()
        assert isinstance(jaxopt, optax.GradientTransformation)

    @pytest.mark.jax
    def test_PieceWise_decay_jax(self):
        """test creating an optimizer with a PeiceWise constant decay to the learning rate"""
        rate = optimizers.PiecewiseConstantSchedule(initial_rate=0.1,
                                                    boundaries_and_scales={
                                                        5000: 0.1,
                                                        10000: 0.1,
                                                        15000: 0.1
                                                    })
        opt = optimizers.Adam(learning_rate=rate)
        jaxopt = opt._create_jax_optimizer()
        assert isinstance(jaxopt, optax.GradientTransformation)

    @pytest.mark.torch
    def test_KFAC(self):
        """test creating a KFAC optimizer"""
        import numpy as np

        np.random.seed(123)
        # Conv2d and Linear layers test(CNN classification)
        n_samples = 10
        n_features = 10
        n_tasks = 1

        X = np.random.rand(n_samples, 1, n_features, n_features)
        y = np.random.randint(2, size=(n_samples, n_tasks)).astype(np.float32)
        dataset = dc.data.NumpyDataset(X, y)

        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.Conv2d(32, 64, kernel_size=3,
                            padding=1), torch.nn.Flatten(),
            torch.nn.Linear(64 * n_features * n_features, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, n_tasks))
        model = dc.models.TorchModel(model,
                                     dc.models.losses.L2Loss(),
                                     optimizers=optimizers.KFAC(
                                         model=model,
                                         learning_rate=0.003,
                                         Tinv=10))
        # Fit trained model
        model.fit(
            dataset,
            nb_epoch=100,
        )

        # Eval model on train
        scores = model.evaluate(dataset, [metric])
        assert scores[metric.name] > 0.9

    @pytest.mark.torch
    def test_lamb_pytorch(self):
        """Test creating an Lamb optimizer."""
        opt = optimizers.Lamb(learning_rate=0.01)
        params = [torch.nn.Parameter(torch.Tensor([1.0]))]
        torchopt = opt._create_pytorch_optimizer(params)
        assert isinstance(torchopt, LambOptimizer)
