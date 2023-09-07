"""Generative Adversarial Networks."""

from deepchem.models.torch_models import layers
from deepchem.models.torch_models.torch_model import TorchModel
# from tensorflow.keras.layers import Input, Lambda, Layer, Softmax, Reshape, Multiply

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class GAN(TorchModel):
    """Implements Generative Adversarial Networks.

    A Generative Adversarial Network (GAN) is a type of generative model.  It
    consists of two parts called the "generator" and the "discriminator".  The
    generator takes random noise as input and transforms it into an output that
    (hopefully) resembles the training data.  The discriminator takes a set of
    samples as input and tries to distinguish the real training samples from the
    ones created by the generator.  Both of them are trained together.  The
    discriminator tries to get better and better at telling real from false data,
    while the generator tries to get better and better at fooling the discriminator.

    In many cases there also are additional inputs to the generator and
    discriminator.  In that case it is known as a Conditional GAN (CGAN), since it
    learns a distribution that is conditional on the values of those inputs.  They
    are referred to as "conditional inputs".

    Many variations on this idea have been proposed, and new varieties of GANs are
    constantly being proposed.  This class tries to make it very easy to implement
    straightforward GANs of the most conventional types.  At the same time, it
    tries to be flexible enough that it can be used to implement many (but
    certainly not all) variations on the concept.

    To define a GAN, you must create a subclass that provides implementations of
    the following methods:

    get_noise_input_shape()
    get_data_input_shapes()
    create_generator()
    create_discriminator()

    If you want your GAN to have any conditional inputs you must also implement:

    get_conditional_input_shapes()

    The following methods have default implementations that are suitable for most
    conventional GANs.  You can override them if you want to customize their
    behavior:

    create_generator_loss()
    create_discriminator_loss()
    get_noise_batch()

    This class allows a GAN to have multiple generators and discriminators, a model
    known as MIX+GAN.  It is described in Arora et al., "Generalization and
    Equilibrium in Generative Adversarial Nets (GANs)" (https://arxiv.org/abs/1703.00573).
    This can lead to better models, and is especially useful for reducing mode
    collapse, since different generators can learn different parts of the
    distribution.  To use this technique, simply specify the number of generators
    and discriminators when calling the constructor.  You can then tell
    predict_gan_generator() which generator to use for predicting samples.
    """

    def __init__(self, n_generators=1, n_discriminators=1):
        """Construct a GAN.

        In addition to the parameters listed below, this class accepts all the
        keyword arguments from KerasModel.

        Parameters
        ----------
        n_generators: int
            the number of generators to include
        n_discriminators: int
            the number of discriminators to include
        """
        
        self.n_generators = n_generators
        self.n_discriminators = n_discriminators

        # Create the inputs.

        # self.noise_input = Input(shape=self.get_noise_input_shape())
        self.noise_input = torch.randn(1, self.get_noise_input_shape())
        self.data_input_layers = []
        for shape in self.get_data_input_shapes():
            self.data_input_layers.append(torch.randn(1, shape))
        self.data_inputs = [i.ref() for i in self.data_input_layers]
        self.conditional_input_layers = []
        for shape in self.get_conditional_input_shapes():
            self.conditional_input_layers.append(torch.randn(1, shape))
        self.conditional_inputs = [
            i.ref() for i in self.conditional_input_layers
        ]

        # Create the generators.

        self.generators = []
        self.gen_variables = []
        generator_outputs = []
        for i in range(n_generators):
            generator = self.create_generator()
            self.generators.append(generator)
            generator_outputs.append(
                generator(
                    _list_or_tensor([self.noise_input] +
                                    self.conditional_input_layers)))
            self.gen_variables += generator.trainable_variables

        # Create the discriminators.

        self.discriminators = []
        self.discrim_variables = []
        discrim_train_outputs = []
        discrim_gen_outputs = []
        for i in range(n_discriminators):
            discriminator = self.create_discriminator()
            self.discriminators.append(discriminator)
            discrim_train_outputs.append(
                self._call_discriminator(discriminator, self.data_input_layers,
                                         True))
            for gen_output in generator_outputs:
                if torch.is_tensor(gen_output):
                    gen_output = [gen_output]
                discrim_gen_outputs.append(
                    self._call_discriminator(discriminator, gen_output, False))
            self.discrim_variables += discriminator.trainable_variables

        # Compute the loss functions.

        gen_losses = [
            self.create_generator_loss(d) for d in discrim_gen_outputs
        ]
        discrim_losses = []
        for i in range(n_discriminators):
            for j in range(n_generators):
                discrim_losses.append(
                    self.create_discriminator_loss(
                        discrim_train_outputs[i],
                        discrim_gen_outputs[i * n_generators + j]))
        if n_generators == 1 and n_discriminators == 1:
            total_gen_loss = gen_losses[0]
            total_discrim_loss = discrim_losses[0]
        else:
            # Create learnable weights for the generators and discriminators.

            gen_alpha = layers.Variable(np.ones((1, n_generators)),
                                        dtype=torch.float32)
            # We pass an input to the Variable layer to work around a bug in TF 1.14.
            gen_weights = torch.softmax()(gen_alpha([self.noise_input]))
            discrim_alpha = layers.Variable(np.ones((1, n_discriminators)),
                                            dtype=torch.float32)
            discrim_weights = torch.softmax()(discrim_alpha([self.noise_input]))

            # Compute the weighted errors

            weight_products = Reshape(
                (n_generators * n_discriminators,))(Multiply()([
                    Reshape((n_discriminators, 1))(discrim_weights),
                    Reshape((1, n_generators))(gen_weights)
                ]))
            stacked_gen_loss = layers.Stack(axis=0)(gen_losses)
            stacked_discrim_loss = layers.Stack(axis=0)(discrim_losses)
            total_gen_loss = torch.sum(stacked_gen_loss * weight_products)
            total_discrim_loss = torch.sum(stacked_discrim_loss *
                                           weight_products)
            self.gen_variables += gen_alpha.trainable_variables
            self.discrim_variables += gen_alpha.trainable_variables
            self.discrim_variables += discrim_alpha.trainable_variables

            # Add an entropy term to the loss.

            entropy = -(
                torch.sum(torch.log(gen_weights)) / n_generators +
                torch.sum(torch.log(discrim_weights)) / n_discriminators)
            total_discrim_loss = total_discrim_loss + entropy

        # Create the Keras model.

        inputs = [self.noise_input
                 ] + self.data_input_layers + self.conditional_input_layers
        outputs = [total_gen_loss, total_discrim_loss]
        self.gen_loss_fn = lambda outputs, labels, weights: outputs[0]
        self.discrim_loss_fn = lambda outputs, labels, weights: outputs[1]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        super(GAN, self).__init__(model, self.gen_loss_fn)

    def _call_discriminator(self, discriminator, inputs, train):
        """Invoke the discriminator on a set of inputs.

        This is a separate method so WGAN can override it and also return the
        gradient penalty.
        """
        return discriminator(
            _list_or_tensor(inputs + self.conditional_input_layers))

    def get_noise_input_shape(self):
        """Get the shape of the generator's noise input layer.

        Subclasses must override this to return a tuple giving the shape of the
        noise input.  The actual Input layer will be created automatically.  The
        dimension corresponding to the batch size should be omitted.
        """
        raise NotImplementedError("Subclasses must implement this.")

    def get_data_input_shapes(self):
        """Get the shapes of the inputs for training data.

        Subclasses must override this to return a list of tuples, each giving the
        shape of one of the inputs.  The actual Input layers will be created
        automatically.  This list of shapes must also match the shapes of the
        generator's outputs.  The dimension corresponding to the batch size should
        be omitted.
        """
        raise NotImplementedError("Subclasses must implement this.")

    def get_conditional_input_shapes(self):
        """Get the shapes of any conditional inputs.

        Subclasses may override this to return a list of tuples, each giving the
        shape of one of the conditional inputs.  The actual Input layers will be
        created automatically.  The dimension corresponding to the batch size should
        be omitted.

        The default implementation returns an empty list, meaning there are no
        conditional inputs.
        """
        return []

    def get_noise_batch(self, batch_size):
        """Get a batch of random noise to pass to the generator.

        This should return a NumPy array whose shape matches the one returned by
        get_noise_input_shape().  The default implementation returns normally
        distributed values.  Subclasses can override this to implement a different
        distribution.
        """
        size = list(self.get_noise_input_shape())
        size = [batch_size] + size
        return np.random.normal(size=size)

    def create_generator(self):
        """Create and return a generator.

        Subclasses must override this to construct the generator.  The returned
        value should be a tf.keras.Model whose inputs are a batch of noise, followed
        by any conditional inputs.  The number and shapes of its outputs must match
        the return value from get_data_input_shapes(), since generated data must
        have the same form as training data.
        """
        raise NotImplementedError("Subclasses must implement this.")

    def create_discriminator(self):
        """Create and return a discriminator.

        Subclasses must override this to construct the discriminator.  The returned
        value should be a tf.keras.Model whose inputs are all data inputs, followed
        by any conditional inputs.  Its output should be a one dimensional tensor
        containing the probability of each sample being a training sample.
        """
        raise NotImplementedError("Subclasses must implement this.")

    def create_generator_loss(self, discrim_output):
        """Create the loss function for the generator.

        The default implementation is appropriate for most cases.  Subclasses can
        override this if the need to customize it.

        Parameters
        ----------
        discrim_output: Tensor
            the output from the discriminator on a batch of generated data.  This is
            its estimate of the probability that each sample is training data.

        Returns
        -------
        A Tensor equal to the loss function to use for optimizing the generator.
        """
        return -torch.mean(torch.log(discrim_output + 1e-10))

    def create_discriminator_loss(self, discrim_output_train,
                                  discrim_output_gen):
        """Create the loss function for the discriminator.

        The default implementation is appropriate for most cases.  Subclasses can
        override this if the need to customize it.

        Parameters
        ----------
        discrim_output_train: Tensor
            the output from the discriminator on a batch of training data.  This is
            its estimate of the probability that each sample is training data.
        discrim_output_gen: Tensor
            the output from the discriminator on a batch of generated data.  This is
            its estimate of the probability that each sample is training data.

        Returns
        -------
        A Tensor equal to the loss function to use for optimizing the discriminator.
        """
        return -torch.mean(
            torch.log(discrim_output_train + 1e-10) +
            torch.log(1 - discrim_output_gen + 1e-10))


def _list_or_tensor(inputs):
    if len(inputs) == 1:
        return inputs[0]
    return inputs
