"""Generative Adversarial Networks."""
import numpy as np
import torch
import torch.nn as nn
import time
from typing import Callable
from deepchem.models.torch_models.torch_model import TorchModel


class GAN(nn.Module):
    """Builder class for Generative Adversarial Networks.

    A Generative Adversarial Network (GAN) is a type of generative model. It
    consists of two parts called the "generator" and the "discriminator". The
    generator takes random noise as input and transforms it into an output that
    (hopefully) resembles the training data. The discriminator takes a set of
    samples as input and tries to distinguish the real training samples from the
    ones created by the generator. Both of them are trained together. The
    discriminator tries to get better and better at telling real from false data,
    while the generator tries to get better and better at fooling the discriminator.

    """

    def __init__(self,
                 noise_input_shape: tuple,
                 data_input_shape: list,
                 conditional_input_shape: list,
                 generator_fn: Callable,
                 discriminator_fn: Callable,
                 n_generators: int = 1,
                 n_discriminators: int = 1):
        """Construct a GAN.

        In addition to the parameters listed below, this class accepts all the
        keyword arguments from KerasModel.

        Parameters
        ----------
        noise_input_shape: tuple
            the shape of the noise input to the generator.  The first dimension
            (corresponding to the batch size) should be omitted.
        data_input_shape: list of tuple
            the shapes of the inputs to the discriminator.  The first dimension
            (corresponding to the batch size) should be omitted.
        conditional_input_shape: list of tuple
            the shapes of the conditional inputs to the generator and discriminator.
            The first dimension (corresponding to the batch size) should be omitted.
            If there are no conditional inputs, this should be an empty list.
        generator_fn: Callable
            a function that returns a generator.  It will be called with no arguments.
            The returned value should be a nn.Module whose input is a list
            containing a batch of noise, followed by any conditional inputs.  The
            number and shapes of its outputs must match the return value from
            get_data_input_shapes(), since generated data must have the same form as
            training data.
        discriminator_fn: Callable
            a function that returns a discriminator.  It will be called with no
            arguments.  The returned value should be a nn.Module whose input is a
            list containing a batch of data, followed by any conditional inputs.  Its
            output should be a one dimensional tensor containing the probability of
            each sample being a training sample.
        n_generators: int
            the number of generators to include
        n_discriminators: int
            the number of discriminators to include
        """
        super(GAN, self).__init__()
        self.n_generators = n_generators
        self.n_discriminators = n_discriminators
        self.noise_input_shape = noise_input_shape
        self.data_input_shape = data_input_shape
        self.conditional_input_shape = conditional_input_shape
        self.create_generator = generator_fn
        self.create_discriminator = discriminator_fn

        # Inputs
        # Noise Input
        self.noise_input = nn.Parameter(torch.empty(self.noise_input_shape))
        # Data Input
        self.data_inputs = [
            nn.Parameter(torch.ones(s)) for s in self.data_input_shape
        ]
        # Conditional Input
        self.conditional_inputs = [
            nn.Parameter(torch.empty(s)) for s in self.conditional_input_shape
        ]

        self.data_input_layers = []
        for idx, shape in enumerate(self.data_input_shape):
            self.data_input_layers.append(nn.Parameter(torch.empty(shape)))
        self.conditional_input_layers = []
        for idx, shape in enumerate(self.conditional_input_shape):
            self.conditional_input_layers.append(
                nn.Parameter(torch.empty(shape)))

        # Generators
        self.generators = nn.ModuleList()
        self.gen_variables = nn.ParameterList()
        for i in range(n_generators):
            generator = self.create_generator()
            self.generators.append(generator)
            self.gen_variables += list(generator.parameters())

        # Discriminators
        self.discriminators = nn.ModuleList()
        self.discrim_variables = nn.ParameterList()

        for i in range(n_discriminators):
            discriminator = self.create_discriminator()
            self.discriminators.append(discriminator)
            self.discrim_variables += list(discriminator.parameters())

    def forward(self, inputs):
        """Compute the output of the GAN.

        Parameters
        ----------
        inputs: list of Tensor
            the inputs to the GAN. The first element must be a batch of noise,
            followed by data inputs and any conditional inputs.

        Returns
        -------
        total_gen_loss: Tensor
            the total loss for the generator
        total_discrim_loss: Tensor
            the total loss for the discriminator
        """

        n_generators = self.n_generators
        n_discriminators = self.n_discriminators
        noise_input, data_input_layers, conditional_input_layers = inputs[
            0], inputs[1], inputs[2]

        self.noise_input.data = noise_input
        self.conditional_input_layers = [conditional_input_layers]
        self.data_input_layers = [data_input_layers]

        # Forward pass through generators
        generator_outputs = [
            gen(_list_or_tensor([[noise_input] + self.conditional_input_layers
                                ])) for gen in self.generators
        ]

        # Forward pass through discriminators
        discrim_train_outputs = [
            self._call_discriminator(disc, self.data_input_layers, True)
            for disc in self.discriminators
        ]

        discrim_gen_outputs = [
            self._call_discriminator(disc, [gen_output], False)
            for disc in self.discriminators
            for gen_output in generator_outputs
        ]

        # Compute loss functions
        gen_losses = [
            self.create_generator_loss(d) for d in discrim_gen_outputs
        ]
        discrim_losses = [
            self.create_discriminator_loss(
                discrim_train_outputs[i],
                discrim_gen_outputs[i * self.n_generators + j])
            for i in range(self.n_discriminators)
            for j in range(self.n_generators)
        ]

        # Compute the weighted errors
        if n_generators == 1 and n_discriminators == 1:
            total_gen_loss = gen_losses[0]
            total_discrim_loss = discrim_losses[0]
        else:
            # Create learnable weights for the generators and discriminators.

            gen_alpha = nn.Parameter(torch.ones(1, n_generators))
            gen_weights = nn.Parameter(torch.softmax(gen_alpha, dim=1))

            discrim_alpha = nn.Parameter(torch.ones(1, n_discriminators))
            discrim_weights = nn.Parameter(torch.softmax(discrim_alpha, dim=1))

            # Compute the weighted errors

            discrim_weights_n = discrim_weights.view(-1, self.n_discriminators,
                                                     1)
            gen_weights_n = gen_weights.view(-1, 1, self.n_generators)

            weight_products = torch.mul(discrim_weights_n, gen_weights_n)
            weight_products = weight_products.view(
                -1, self.n_generators * self.n_discriminators)

            stacked_gen_loss = torch.stack(gen_losses, axis=0)
            stacked_discrim_loss = torch.stack(discrim_losses, axis=0)

            total_gen_loss = torch.sum(stacked_gen_loss * weight_products)
            total_discrim_loss = torch.sum(stacked_discrim_loss *
                                           weight_products)

            self.gen_variables += [gen_alpha]
            self.discrim_variables += [gen_alpha]
            self.discrim_variables += [discrim_alpha]

            # Add an entropy term to the loss.

            entropy = -(
                torch.sum(torch.log(gen_weights)) / n_generators +
                torch.sum(torch.log(discrim_weights)) / n_discriminators)
            total_discrim_loss = total_discrim_loss + entropy

        return total_gen_loss, total_discrim_loss

    def _call_discriminator(self, discriminator, inputs, train):
        """Invoke the discriminator on a set of inputs.

        This is a separate method so WGAN can override it and also return the
        gradient penalty.

        Parameters
        ----------
        discriminator: nn.Module
            the discriminator to invoke
        inputs: list of Tensor
            the inputs to the discriminator.  The first element must be a batch of
            data, followed by any conditional inputs.
        train: bool
            if True, the discriminator should be invoked in training mode. If False,
            it should be invoked in inference mode.
        """
        return discriminator(
            _list_or_tensor(inputs + self.conditional_input_layers))

    def get_noise_batch(self, batch_size):
        """Get a batch of random noise to pass to the generator.

        This should return a NumPy array whose shape matches the one returned by
        get_noise_input_shape().  The default implementation returns normally
        distributed values.  Subclasses can override this to implement a different
        distribution.

        Parameters
        ----------
        batch_size: int
            the number of samples to generate

        Returns
        -------
        a batch of random noise
        """
        size = list(self.noise_input_shape)
        size = [batch_size] + size[1:]
        return np.random.normal(size=size)

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

    def discrim_loss_fn_wrapper(self, outputs, labels, weights):
        """Wrapper around create_discriminator_loss for use with fit_generator.

        Parameters
        ----------
        outputs: Tensor
            the output from the discriminator on a batch of training data.  This is
            its estimate of the probability that each sample is training data.
        labels: Tensor
            the labels for the batch.  These are ignored.
        weights: Tensor
            the weights for the batch.  These are ignored.

        Returns
        -------
        the value of the discriminator loss function for this input.
        """
        discrim_output_train, discrim_output_gen = outputs
        return self.create_discriminator_loss(discrim_output_train,
                                              discrim_output_gen)

    def gen_loss_fn_wrapper(self, outputs, labels, weights):
        """Wrapper around create_generator_loss for use with fit_generator.

        Parameters
        ----------
        outputs: Tensor
            the output from the discriminator on a batch of generated data.  This is
            its estimate of the probability that each sample is training data.
        labels: Tensor
            the labels for the batch.  These are ignored.
        weights: Tensor
            the weights for the batch.  These are ignored.

        Returns
        -------
        the value of the generator loss function for this input.
        """
        discrim_output_train, discrim_output_gen = outputs
        return self.create_generator_loss(discrim_output_gen)


def _list_or_tensor(inputs):
    if len(inputs) == 1:
        return inputs[0]
    return inputs


