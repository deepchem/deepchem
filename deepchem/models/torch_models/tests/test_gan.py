import deepchem as dc
import numpy as np
import pytest
import tempfile
from flaky import flaky

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from deepchem.models.torch_models import GAN, GANModel

    # helper classes that depend on torch, they need to be in the try/catch block
    class Generator(nn.Module):
        """A simple generator for testing."""

        def __init__(self, noise_input_shape, conditional_input_shape):
            super(Generator, self).__init__()
            self.noise_input_shape = noise_input_shape
            self.conditional_input_shape = conditional_input_shape

            self.noise_dim = noise_input_shape[1:]
            self.conditional_dim = conditional_input_shape[1:]

            input_dim = sum(self.noise_dim) + sum(self.conditional_dim)
            self.output = nn.Linear(input_dim, 1)

        def forward(self, input):
            noise_input, conditional_input = input

            inputs = torch.cat((noise_input, conditional_input), dim=1)
            output = self.output(inputs)
            return output

    class Discriminator(nn.Module):
        """A simple discriminator for testing."""

        def __init__(self, data_input_shape, conditional_input_shape):
            super(Discriminator, self).__init__()
            self.data_input_shape = data_input_shape
            self.conditional_input_shape = conditional_input_shape

            data_dim = data_input_shape[
                1:]  # Extracting the actual data dimension
            conditional_dim = conditional_input_shape[
                1:]  # Extracting the actual conditional dimension
            input_dim = sum(data_dim) + sum(conditional_dim)

            # Define the dense layers
            self.dense1 = nn.Linear(input_dim, 10)
            self.dense2 = nn.Linear(10, 1)

        def forward(self, input):
            data_input, conditional_input = input
            discrim_in = torch.cat((data_input, conditional_input), dim=1)
            x = F.relu(self.dense1(discrim_in))
            a = self.dense2(x)
            output = torch.sigmoid(a)
            return output

    class Discriminator_WGAN(nn.Module):
        """A simple discriminator for testing."""

        def __init__(self, data_input_shape, conditional_input_shape):
            super(Discriminator_WGAN, self).__init__()
            self.data_input_shape = data_input_shape
            self.conditional_input_shape = conditional_input_shape

            data_dim = data_input_shape[
                1:]  # Extracting the actual data dimension
            conditional_dim = conditional_input_shape[
                1:]  # Extracting the actual conditional dimension
            input_dim = sum(data_dim) + sum(conditional_dim)

            # Define the dense layers
            self.dense1 = nn.Linear(input_dim, 10)
            self.dense2 = nn.Linear(10, 1)

        def forward(self, input):
            data_input, conditional_input = input
            discrim_in = torch.cat((data_input, conditional_input), dim=1)
            output = F.relu(self.dense1(discrim_in))
            output = self.dense2(output)
            return output

    class ExampleGAN(GAN):
        """A simple GAN for testing."""

        def get_noise_input_shape(self):
            return (
                16,
                2,
            )

        def get_data_input_shapes(self):
            return [(
                16,
                1,
            )]

        def get_conditional_input_shapes(self):
            return [(
                16,
                1,
            )]

        def create_generator(self):
            noise_dim = self.get_noise_input_shape()
            conditional_dim = self.get_conditional_input_shapes()[0]

            return nn.Sequential(Generator(noise_dim, conditional_dim))

        def create_discriminator(self):
            data_input_shape = self.get_data_input_shapes()[0]
            conditional_input_shape = self.get_conditional_input_shapes()[0]

            return nn.Sequential(
                Discriminator(data_input_shape, conditional_input_shape))

    class ExampleGANModel(GANModel):
        """A simple GAN for testing."""

        def get_noise_input_shape(self):
            return (
                100,
                2,
            )

        def get_data_input_shapes(self):
            return [(
                100,
                1,
            )]

        def get_conditional_input_shapes(self):
            return [(
                100,
                1,
            )]

        def create_generator(self):
            noise_dim = self.get_noise_input_shape()
            conditional_dim = self.get_conditional_input_shapes()[0]

            return nn.Sequential(Generator(noise_dim, conditional_dim))

        def create_discriminator(self):
            data_input_shape = self.get_data_input_shapes()[0]
            conditional_input_shape = self.get_conditional_input_shapes()[0]

            return nn.Sequential(
                Discriminator(data_input_shape, conditional_input_shape))

    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
def create_generator(noise_dim, conditional_dim):
    noise_dim = noise_dim
    conditional_dim = conditional_dim[0]

    return nn.Sequential(Generator(noise_dim, conditional_dim))


@pytest.mark.torch
def create_discriminator(data_input_shape, conditional_input_shape):
    data_input_shape = data_input_shape[0]
    conditional_input_shape = conditional_input_shape[0]

    return nn.Sequential(
        Discriminator(data_input_shape, conditional_input_shape))


@pytest.mark.torch
def test_forward_pass():
    batch_size = 16
    noise_shape = (
        batch_size,
        2,
    )
    data_shape = [(
        batch_size,
        1,
    )]
    conditional_shape = [(
        batch_size,
        1,
    )]

    gan = ExampleGAN(noise_shape,
                     data_shape,
                     conditional_shape,
                     create_generator(noise_shape, conditional_shape),
                     create_discriminator(data_shape, conditional_shape),
                     device='cpu')

    noise = torch.rand(*gan.noise_input_shape)
    real_data = torch.rand(*gan.data_input_shape[0])
    conditional = torch.rand(*gan.conditional_input_shape[0])
    gen_loss, disc_loss = gan([noise, real_data, conditional])

    assert isinstance(gen_loss, torch.Tensor)
    assert gen_loss > 0

    assert isinstance(disc_loss, torch.Tensor)
    assert disc_loss > 0


@pytest.mark.torch
def test_get_noise_batch():
    batch_size = 16
    noise_shape = (
        batch_size,
        2,
    )
    data_shape = [(
        batch_size,
        1,
    )]
    conditional_shape = [(
        batch_size,
        1,
    )]

    gan = ExampleGAN(noise_shape,
                     data_shape,
                     conditional_shape,
                     create_generator(noise_shape, conditional_shape),
                     create_discriminator(data_shape, conditional_shape),
                     device='cpu')
    noise = gan.get_noise_batch(batch_size)
    assert noise.shape == (gan.noise_input_shape)


@pytest.mark.torch
def generate_batch(batch_size):
    """Draw training data from a Gaussian distribution, where the mean  is a conditional input."""
    means = 10 * np.random.random([batch_size, 1])
    values = np.random.normal(means, scale=2.0)
    return means, values


@pytest.mark.torch
def generate_data(gan, batches, batch_size):
    for _ in range(batches):
        means, values = generate_batch(batch_size)
        batch = {gan.data_inputs[0]: values, gan.conditional_inputs[0]: means}
        yield batch


@flaky
@pytest.mark.torch
def test_cgan():
    """Test fitting a conditional GAN."""

    gan = ExampleGANModel(learning_rate=0.01)
    data = generate_data(gan, 500, 100)
    gan.fit_gan(data, generator_steps=0.5, checkpoint_interval=0)

    # See if it has done a plausible job of learning the distribution.

    means = 10 * np.random.random([1000, 1])
    values = gan.predict_gan_generator(conditional_inputs=[means])
    deltas = values - means
    assert abs(np.mean(deltas)) < 1.0
    assert np.std(deltas) > 1.0
    assert gan.get_global_step() == 500


@pytest.mark.torch
def test_cgan_reload():
    """Test reloading a conditional GAN."""

    model_dir = tempfile.mkdtemp()
    gan = ExampleGANModel(learning_rate=0.01, model_dir=model_dir)
    gan.fit_gan(generate_data(gan, 500, 100), generator_steps=0.5)

    # See if it has done a plausible job of learning the distribution.
    means = 10 * np.random.random([1000, 1])
    batch_size = len(means)
    noise_input = gan.get_noise_batch(batch_size=batch_size)
    values = gan.predict_gan_generator(noise_input=noise_input,
                                       conditional_inputs=[means])
    deltas = values - means
    assert np.std(deltas) > 1.0
    assert gan.get_global_step() == 500

    reloaded_gan = ExampleGANModel(learning_rate=0.01, model_dir=model_dir)
    reloaded_gan.restore(strict=False)
    reloaded_values = reloaded_gan.predict_gan_generator(
        noise_input=noise_input, conditional_inputs=[means])

    assert np.all(values == reloaded_values)


@flaky
@pytest.mark.torch
def test_mix_gan():
    """Test a GAN with multiple generators and discriminators."""

    gan = ExampleGANModel(n_generators=2,
                          n_discriminators=2,
                          learning_rate=0.01)
    data = generate_data(gan, 1000, 100)
    gan.fit_gan(data, generator_steps=0.5, checkpoint_interval=0)

    # See if it has done a plausible job of learning the distribution.

    means = 10 * np.random.random([1000, 1])
    for i in range(2):
        values = gan.predict_gan_generator(conditional_inputs=[means],
                                           generator_index=i)
        deltas = values - means
        assert abs(np.mean(deltas)) < 1.0
        assert np.std(deltas) > 1.0
    assert gan.get_global_step() == 1000


@flaky
@pytest.mark.torch
def test_mix_gan_reload():
    """Test reloading a GAN with multiple generators and discriminators."""

    model_dir = tempfile.mkdtemp()
    gan = ExampleGANModel(n_generators=2,
                          n_discriminators=2,
                          learning_rate=0.01,
                          model_dir=model_dir)
    gan.fit_gan(generate_data(gan, 1000, 100), generator_steps=0.5)

    reloaded_gan = ExampleGANModel(n_generators=2,
                                   n_discriminators=2,
                                   learning_rate=0.01,
                                   model_dir=model_dir)
    reloaded_gan.restore(strict=False)
    # See if it has done a plausible job of learning the distribution.

    means = 10 * np.random.random([1000, 1])
    batch_size = len(means)
    noise_input = gan.get_noise_batch(batch_size=batch_size)
    for i in range(2):
        values = gan.predict_gan_generator(noise_input=noise_input,
                                           conditional_inputs=[means],
                                           generator_index=i)
        reloaded_values = reloaded_gan.predict_gan_generator(
            noise_input=noise_input,
            conditional_inputs=[means],
            generator_index=i)
        assert np.all(values == reloaded_values)
    assert gan.get_global_step() == 1000
    # No training has been done after reload
    assert reloaded_gan.get_global_step() == 1000


@flaky
@pytest.mark.torch
def test_wgan():
    """Test fitting a conditional WGAN."""

    class ExampleWGAN(dc.models.torch_models.WGANModel):

        def get_noise_input_shape(self):
            return (
                100,
                2,
            )

        def get_data_input_shapes(self):
            return [(
                100,
                1,
            )]

        def get_conditional_input_shapes(self):
            return [(
                100,
                1,
            )]

        def create_generator(self):
            noise_dim = self.get_noise_input_shape()
            conditional_dim = self.get_conditional_input_shapes()[0]

            return nn.Sequential(Generator(noise_dim, conditional_dim))

        def create_discriminator(self):
            data_input_shape = self.get_data_input_shapes()[0]
            conditional_input_shape = self.get_conditional_input_shapes()[0]

            return nn.Sequential(
                Discriminator_WGAN(data_input_shape, conditional_input_shape))

    # We have to set the gradient penalty very small because the generator's
    # output is only a single number, so the default penalty would constrain
    # it far too much.

    gan = ExampleWGAN(learning_rate=0.01, gradient_penalty=0.1)
    gan.fit_gan(generate_data(gan, 1000, 100), generator_steps=0.1)

    # See if it has done a plausible job of learning the distribution.

    means = 10 * np.random.random([1000, 1])
    values = gan.predict_gan_generator(conditional_inputs=[means])
    deltas = values - means

    assert abs(np.mean(deltas)) < 1.0
    assert np.std(deltas) > 1.0


@flaky
@pytest.mark.torch
def test_wgan_reload():
    """Test fitting a conditional WGAN."""

    class ExampleWGAN(dc.models.torch_models.WGANModel):

        def get_noise_input_shape(self):
            return (
                100,
                2,
            )

        def get_data_input_shapes(self):
            return [(
                100,
                1,
            )]

        def get_conditional_input_shapes(self):
            return [(
                100,
                1,
            )]

        def create_generator(self):
            noise_dim = self.get_noise_input_shape()
            conditional_dim = self.get_conditional_input_shapes()[0]

            return nn.Sequential(Generator(noise_dim, conditional_dim))

        def create_discriminator(self):
            data_input_shape = self.get_data_input_shapes()[0]
            conditional_input_shape = self.get_conditional_input_shapes()[0]

            return nn.Sequential(
                Discriminator_WGAN(data_input_shape, conditional_input_shape))

    # We have to set the gradient penalty very small because the generator's
    # output is only a single number, so the default penalty would constrain
    # it far too much.

    model_dir = tempfile.mkdtemp()
    gan = ExampleWGAN(
        learning_rate=0.01,
        gradient_penalty=0.1,
        model_dir=model_dir,
    )
    gan.fit_gan(generate_data(gan, 1000, 100), generator_steps=0.1)

    reloaded_gan = ExampleWGAN(
        learning_rate=0.01,
        gradient_penalty=0.1,
        model_dir=model_dir,
    )
    reloaded_gan.restore()

    # See if it has done a plausible job of learning the distribution.
    means = 10 * np.random.random([1000, 1])
    batch_size = len(means)
    noise_input = gan.get_noise_batch(batch_size=batch_size)
    values = gan.predict_gan_generator(noise_input=noise_input,
                                       conditional_inputs=[means])
    reloaded_values = reloaded_gan.predict_gan_generator(
        noise_input=noise_input, conditional_inputs=[means])
    assert np.all(values == reloaded_values)


@pytest.mark.torch
def gradient_penalty_layer():
    """A gradient penalty layer for testing."""
    from deepchem.models.torch_models import GradientPenaltyLayer

    class ExampleWGAN(dc.models.torch_models.WGANModel):

        def get_noise_input_shape(self):
            return (
                100,
                2,
            )

        def get_data_input_shapes(self):
            return [(
                100,
                1,
            )]

        def get_conditional_input_shapes(self):
            return [(
                100,
                1,
            )]

        def create_generator(self):
            noise_dim = self.get_noise_input_shape()
            conditional_dim = self.get_conditional_input_shapes()[0]

            return nn.Sequential(Generator(noise_dim, conditional_dim))

        def create_discriminator(self):
            data_input_shape = self.get_data_input_shapes()[0]
            conditional_input_shape = self.get_conditional_input_shapes()[0]

            return nn.Sequential(
                Discriminator_WGAN(data_input_shape, conditional_input_shape))

    gan = ExampleWGAN()
    discriminator = gan.discriminators[0]
    return GradientPenaltyLayer(gan, discriminator)


@pytest.mark.torch
def test_gpl_forward():
    # Create dummy data
    gpl = gradient_penalty_layer()
    inputs = [torch.randn(4, 1)]
    conditional_inputs = [torch.randn(4, 1)]

    # Call forward
    output, penalty = gpl(inputs, conditional_inputs)

    # Asserts
    assert isinstance(output, torch.Tensor), "Output must be a torch.Tensor"
    assert isinstance(penalty, torch.Tensor), "Penalty must be a torch.Tensor"
    assert output.shape[
        0] == 4, "Output tensor must have the same batch size as inputs"
    assert penalty.ndim == 0 or penalty.shape == (
        4,), "Penalty should be a scalar or the same size as the batch"


@pytest.mark.torch
def test_gpl_penalty_calculation():
    gpl = gradient_penalty_layer()
    # Create dummy data
    inputs = [torch.randn(4, 1)]
    conditional_inputs = [torch.randn(4, 1)]

    # Call forward
    _, penalty = gpl(inputs, conditional_inputs)

    # Since the penalty is a squared norm of the gradients minus 1, multiplied by a constant,
    # it should be non-negative
    assert penalty.item() >= 0, "Penalty should be non-negative"
