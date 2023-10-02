import deepchem as dc
# import numpy as np
import pytest
# import tempfile
from flaky import flaky

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@flaky
@pytest.mark.torch
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


@flaky
@pytest.mark.torch
class Discriminator(nn.Module):
    """A simple discriminator for testing."""

    def __init__(self, data_input_shape, conditional_input_shape):
        super(Discriminator, self).__init__()
        self.data_input_shape = data_input_shape
        self.conditional_input_shape = conditional_input_shape

        data_dim = data_input_shape[1:]  # Extracting the actual data dimension
        conditional_dim = conditional_input_shape[
            1:]  # Extracting the actual conditional dimension
        input_dim = sum(data_dim) + sum(conditional_dim)

        # Define the dense layers
        self.dense1 = nn.Linear(input_dim, 10)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, input):
        data_input, conditional_input = input

        # Concatenate data_input and conditional_input along the second dimension
        discrim_in = torch.cat((data_input, conditional_input), dim=1)

        # Pass the concatenated input through the dense layers
        x = F.relu(self.dense1(discrim_in))
        output = torch.sigmoid(self.dense2(x))

        return output


@flaky
@pytest.mark.torch
class ExampleGAN(dc.models.torch_models.GAN):
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


@flaky
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

    gan = ExampleGAN(noise_shape, data_shape, conditional_shape,
                     create_generator(noise_shape, conditional_shape),
                     create_discriminator(data_shape, conditional_shape))

    noise = torch.rand(*gan.noise_input_shape)
    real_data = torch.rand(*gan.data_input_shape[0])
    conditional = torch.rand(*gan.conditional_input_shape[0])
    gen_loss, disc_loss = gan([noise, real_data, conditional])

    assert isinstance(gen_loss, torch.Tensor)
    assert gen_loss > 0

    assert isinstance(disc_loss, torch.Tensor)
    assert disc_loss > 0


@flaky
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

    gan = ExampleGAN(noise_shape, data_shape, conditional_shape,
                     create_generator(noise_shape, conditional_shape),
                     create_discriminator(data_shape, conditional_shape))
    noise = gan.get_noise_batch(batch_size)
    assert noise.shape == (gan.noise_input_shape)
