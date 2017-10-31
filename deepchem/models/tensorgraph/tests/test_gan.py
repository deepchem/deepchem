import deepchem as dc
import numpy as np
import tensorflow as tf
import unittest
from deepchem.models.tensorgraph import layers


def generate_batch(batch_size):
  """Draw training data from a Gaussian distribution, where the mean  is a conditional input."""
  means = 10 * np.random.random([batch_size, 1])
  values = np.random.normal(means, scale=2.0)
  return means, values


def generate_data(gan, batches, batch_size):
  for i in range(batches):
    means, values = generate_batch(batch_size)
    batch = {gan.data_inputs[0]: values, gan.conditional_inputs[0]: means}
    yield batch


class TestGAN(unittest.TestCase):

  def test_cgan(self):
    """Test fitting a conditional GAN."""

    class ExampleGAN(dc.models.GAN):

      def get_noise_input_shape(self):
        return (None, 2)

      def get_data_input_shapes(self):
        return [(None, 1)]

      def get_conditional_input_shapes(self):
        return [(None, 1)]

      def create_generator(self, noise_input, conditional_inputs):
        gen_in = layers.Concat([noise_input] + conditional_inputs)
        return [layers.Dense(1, in_layers=gen_in)]

      def create_discriminator(self, data_inputs, conditional_inputs):
        discrim_in = layers.Concat(data_inputs + conditional_inputs)
        dense = layers.Dense(10, in_layers=discrim_in, activation_fn=tf.nn.relu)
        return layers.Dense(1, in_layers=dense, activation_fn=tf.sigmoid)

    gan = ExampleGAN(learning_rate=0.003)
    gan.fit_gan(
        generate_data(gan, 5000, 100),
        generator_steps=0.5,
        checkpoint_interval=0)

    # See if it has done a plausible job of learning the distribution.

    means = 10 * np.random.random([1000, 1])
    values = gan.predict_gan_generator(conditional_inputs=[means])
    deltas = values - means
    assert abs(np.mean(deltas)) < 1.0
    assert np.std(deltas) > 1.0

  def test_wgan(self):
    """Test fitting a conditional WGAN."""

    class ExampleWGAN(dc.models.WGAN):

      def get_noise_input_shape(self):
        return (None, 2)

      def get_data_input_shapes(self):
        return [(None, 1)]

      def get_conditional_input_shapes(self):
        return [(None, 1)]

      def create_generator(self, noise_input, conditional_inputs):
        gen_in = layers.Concat([noise_input] + conditional_inputs)
        return [layers.Dense(1, in_layers=gen_in)]

      def create_discriminator(self, data_inputs, conditional_inputs):
        discrim_in = layers.Concat(data_inputs + conditional_inputs)
        dense = layers.Dense(10, in_layers=discrim_in, activation_fn=tf.nn.relu)
        return layers.Dense(1, in_layers=dense)

    # We have to set the gradient penalty very small because the generator's
    # output is only a single number, so the default penalty would constrain
    # it far too much.

    gan = ExampleWGAN(learning_rate=0.003, gradient_penalty=0.1)
    gan.fit_gan(
        generate_data(gan, 10000, 100),
        generator_steps=0.1,
        checkpoint_interval=0)

    # See if it has done a plausible job of learning the distribution.

    means = 10 * np.random.random([1000, 1])
    values = gan.predict_gan_generator(conditional_inputs=[means])
    deltas = values - means
    assert abs(np.mean(deltas)) < 1.0
    assert np.std(deltas) > 1.0
