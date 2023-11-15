"""Generative Adversarial Networks."""
import numpy as np
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')
import time
from typing import Callable, Any
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

    Examples
    --------
    Importing necessary modules

    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.gan import GAN
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F

    Creating a Generator

    >>> class Generator(nn.Module):
    ...     def __init__(self, noise_input_shape, conditional_input_shape):
    ...         super(Generator, self).__init__()
    ...         self.noise_input_shape = noise_input_shape
    ...         self.conditional_input_shape = conditional_input_shape
    ...         self.noise_dim = noise_input_shape[1:]
    ...         self.conditional_dim = conditional_input_shape[1:]
    ...         input_dim = sum(self.noise_dim) + sum(self.conditional_dim)
    ...         self.output = nn.Linear(input_dim, 1)
    ...     def forward(self, input):
    ...         noise_input, conditional_input = input
    ...         inputs = torch.cat((noise_input, conditional_input), dim=1)
    ...         output = self.output(inputs)
    ...         return output

    Creating a Discriminator

    >>> class Discriminator(nn.Module):
    ...     def __init__(self, data_input_shape, conditional_input_shape):
    ...         super(Discriminator, self).__init__()
    ...         self.data_input_shape = data_input_shape
    ...         self.conditional_input_shape = conditional_input_shape
    ...         # Extracting the actual data dimension
    ...         data_dim = data_input_shape[1:]
    ...         # Extracting the actual conditional dimension
    ...         conditional_dim = conditional_input_shape[1:]
    ...         input_dim = sum(data_dim) + sum(conditional_dim)
    ...         # Define the dense layers
    ...         self.dense1 = nn.Linear(input_dim, 10)
    ...         self.dense2 = nn.Linear(10, 1)
    ...     def forward(self, input):
    ...         data_input, conditional_input = input
    ...         # Concatenate data_input and conditional_input along the second dimension
    ...         discrim_in = torch.cat((data_input, conditional_input), dim=1)
    ...         # Pass the concatenated input through the dense layers
    ...         x = F.relu(self.dense1(discrim_in))
    ...         output = torch.sigmoid(self.dense2(x))
    ...         return output

    Defining an Example GAN class

    >>> class ExampleGAN(dc.models.torch_models.GAN):
    ...    def get_noise_input_shape(self):
    ...        return (16,2,)
    ...    def get_data_input_shapes(self):
    ...        return [(16,1,)]
    ...    def get_conditional_input_shapes(self):
    ...        return [(16,1,)]
    ...    def create_generator(self):
    ...        noise_dim = self.get_noise_input_shape()
    ...        conditional_dim = self.get_conditional_input_shapes()[0]
    ...        return nn.Sequential(Generator(noise_dim, conditional_dim))
    ...    def create_discriminator(self):
    ...        data_input_shape = self.get_data_input_shapes()[0]
    ...        conditional_input_shape = self.get_conditional_input_shapes()[0]
    ...        return nn.Sequential(
    ...            Discriminator(data_input_shape, conditional_input_shape))

    Defining the GAN

    >>> batch_size = 16
    >>> noise_shape = (batch_size, 2,)
    >>> data_shape = [(batch_size, 1,)]
    >>> conditional_shape = [(batch_size, 1,)]
    >>> def create_generator(noise_dim, conditional_dim):
    ...     noise_dim = noise_dim
    ...     conditional_dim = conditional_dim[0]
    ...     return nn.Sequential(Generator(noise_dim, conditional_dim))
    >>> def create_discriminator(data_input_shape, conditional_input_shape):
    ...     data_input_shape = data_input_shape[0]
    ...     conditional_input_shape = conditional_input_shape[0]
    ...     return nn.Sequential(
    ...         Discriminator(data_input_shape, conditional_input_shape))
    >>> gan = ExampleGAN(noise_shape, data_shape, conditional_shape,
    ...                  create_generator(noise_shape, conditional_shape),
    ...                  create_discriminator(data_shape, conditional_shape))
    >>> noise = torch.rand(*gan.noise_input_shape)
    >>> real_data = torch.rand(*gan.data_input_shape[0])
    >>> conditional = torch.rand(*gan.conditional_input_shape[0])
    >>> gen_loss, disc_loss = gan([noise, real_data, conditional])

    References
    ----------
    .. [1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in
         neural information processing systems. 2014.

    .. [2] Arora et al., “Generalization and Equilibrium in Generative
         Adversarial Nets (GANs)” (https://arxiv.org/abs/1703.00573)

    Notes
    -----
    This class is a subclass of TorchModel.  It accepts all the keyword arguments
    from TorchModel.
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
        # Data Input Name
        self.data_input_names = [
            "data_input_%d" % i for i in range(len(self.data_inputs))
        ]
        # print("data_input_names: ", self.data_input_names)
        # Conditional Input
        self.conditional_inputs = [
            nn.Parameter(torch.empty(s)) for s in self.conditional_input_shape
        ]
        # Conditional Input Name
        self.conditional_input_names = [
            "conditional_input_%d" % i
            for i in range(len(self.conditional_inputs))
        ]
        # print("conditional_input_names: ", self.conditional_input_names)

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

            stacked_gen_loss = torch.stack(gen_losses, dim=0)
            stacked_discrim_loss = torch.stack(discrim_losses, dim=0)

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

        Returns
        -------
        the output from the discriminator
        """
        return discriminator(
            _list_or_tensor(inputs + self.conditional_input_layers))

    def get_noise_batch(self, batch_size: int) -> np.ndarray:
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
        random_noise: ndarray
            a batch of random noise
        """
        size = list(self.noise_input_shape)
        size = [batch_size] + size[1:]
        return np.random.normal(size=size)

    def create_generator_loss(self,
                              discrim_output: torch.Tensor) -> torch.Tensor:
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
        output: Tensor
            A Tensor equal to the loss function to use for optimizing the generator.
        """
        return -torch.mean(torch.log(discrim_output + 1e-10))

    def create_discriminator_loss(
            self, discrim_output_train: torch.Tensor,
            discrim_output_gen: torch.Tensor) -> torch.Tensor:
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
        output: Tensor
            A Tensor equal to the loss function to use for optimizing the discriminator.
        """
        return -torch.mean(
            torch.log(discrim_output_train + 1e-10) +
            torch.log(1 - discrim_output_gen + 1e-10))

    def discrim_loss_fn_wrapper(self, outputs: list, labels: torch.Tensor,
                                weights: torch.Tensor) -> Any:
        """Wrapper to get the discriminator loss from the fit_generator output

        Parameters
        ----------
        outputs: list of Tensor
            the output from the discriminator on a batch of training data.  This is
            its estimate of the probability that each sample is training data.
        labels: Tensor
            the labels for the batch.  These are ignored.
        weights: Tensor
            the weights for the batch.  These are ignored.

        Returns
        -------
        the value of the discriminator loss from the fit_generator output.
        """
        discrim_output_train, discrim_output_gen = outputs
        return discrim_output_gen

    def gen_loss_fn_wrapper(self, outputs: list, labels: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
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


class GANModel(TorchModel):
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
    known as MIX+GAN.  It is described in [2]
    This can lead to better models, and is especially useful for reducing mode
    collapse, since different generators can learn different parts of the
    distribution.  To use this technique, simply specify the number of generators
    and discriminators when calling the constructor.  You can then tell
    predict_gan_generator() which generator to use for predicting samples.

    Examples
    --------
    Importing necessary modules

    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.gan import GAN
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F

    Creating a Generator

    >>> class Generator(nn.Module):
    ...     def __init__(self, noise_input_shape, conditional_input_shape):
    ...         super(Generator, self).__init__()
    ...         self.noise_input_shape = noise_input_shape
    ...         self.conditional_input_shape = conditional_input_shape
    ...         self.noise_dim = noise_input_shape[1:]
    ...         self.conditional_dim = conditional_input_shape[1:]
    ...         input_dim = sum(self.noise_dim) + sum(self.conditional_dim)
    ...         self.output = nn.Linear(input_dim, 1)
    ...     def forward(self, input):
    ...         noise_input, conditional_input = input
    ...         inputs = torch.cat((noise_input, conditional_input), dim=1)
    ...         output = self.output(inputs)
    ...         return output

    Creating a Discriminator

    >>> class Discriminator(nn.Module):
    ...     def __init__(self, data_input_shape, conditional_input_shape):
    ...         super(Discriminator, self).__init__()
    ...         self.data_input_shape = data_input_shape
    ...         self.conditional_input_shape = conditional_input_shape
    ...         # Extracting the actual data dimension
    ...         data_dim = data_input_shape[1:]
    ...         # Extracting the actual conditional dimension
    ...         conditional_dim = conditional_input_shape[1:]
    ...         input_dim = sum(data_dim) + sum(conditional_dim)
    ...         # Define the dense layers
    ...         self.dense1 = nn.Linear(input_dim, 10)
    ...         self.dense2 = nn.Linear(10, 1)
    ...     def forward(self, input):
    ...         data_input, conditional_input = input
    ...         # Concatenate data_input and conditional_input along the second dimension
    ...         discrim_in = torch.cat((data_input, conditional_input), dim=1)
    ...         # Pass the concatenated input through the dense layers
    ...         x = F.relu(self.dense1(discrim_in))
    ...         output = torch.sigmoid(self.dense2(x))
    ...         return output

    Defining an Example GAN class

    >>> class ExampleGANModel(dc.models.torch_models.GANModel):
    ...    def get_noise_input_shape(self):
    ...        return (100,2,)
    ...    def get_data_input_shapes(self):
    ...        return [(100,1,)]
    ...    def get_conditional_input_shapes(self):
    ...        return [(100,1,)]
    ...    def create_generator(self):
    ...        noise_dim = self.get_noise_input_shape()
    ...        conditional_dim = self.get_conditional_input_shapes()[0]
    ...        return nn.Sequential(Generator(noise_dim, conditional_dim))
    ...    def create_discriminator(self):
    ...        data_input_shape = self.get_data_input_shapes()[0]
    ...        conditional_input_shape = self.get_conditional_input_shapes()[0]
    ...        return nn.Sequential(
    ...            Discriminator(data_input_shape, conditional_input_shape))

    Defining a function to generate data

    >>> def generate_batch(batch_size):
    ...     means = 10 * np.random.random([batch_size, 1])
    ...     values = np.random.normal(means, scale=2.0)
    ...     return means, values

    >>> def generate_data(gan, batches, batch_size):
    ...     for _ in range(batches):
    ...         means, values = generate_batch(batch_size)
    ...         batch = {
    ...             gan.data_input_names[0]: values,
    ...             gan.conditional_input_names[0]: means
    ...         }
    ...         yield batch

    Defining the GANModel

    >>> batch_size = 100
    >>> noise_shape = (batch_size, 2,)
    >>> data_shape = [(batch_size, 1,)]
    >>> conditional_shape = [(batch_size, 1,)]
    >>> gan = ExampleGANModel(learning_rate=0.01)
    >>> data = generate_data(gan, 500, 100)
    >>> gan.fit_gan(data, generator_steps=0.5, checkpoint_interval=0)
    >>> means = 10 * np.random.random([1000, 1])
    >>> values = gan.predict_gan_generator(conditional_inputs=[means])

    References
    ----------
    .. [1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in
         neural information processing systems. 2014.

    .. [2] Arora et al., “Generalization and Equilibrium in Generative
         Adversarial Nets (GANs)” (https://arxiv.org/abs/1703.00573)

    """

    def __init__(self, n_generators=1, n_discriminators=1, **kwargs):
        """
        Parameters
        ----------
        n_generators: int
            the number of generators to include
        n_discriminators: int
            the number of discriminators to include
        """
        self.n_generators = n_generators
        self.n_discriminators = n_discriminators

        model = GAN(noise_input_shape=self.get_noise_input_shape(),
                    data_input_shape=self.get_data_input_shapes(),
                    conditional_input_shape=self.get_conditional_input_shapes(),
                    generator_fn=self.create_generator,
                    discriminator_fn=self.create_discriminator,
                    n_generators=n_generators,
                    n_discriminators=n_discriminators)
        self.gen_loss_fn = model.create_generator_loss
        self.discrim_loss_fn = model.create_discriminator_loss
        self.discrim_loss_fn_wrapper = model.discrim_loss_fn_wrapper
        self.gen_loss_fn_wrapper = model.gen_loss_fn_wrapper
        self.data_inputs = model.data_inputs
        self.data_input_names = model.data_input_names
        self.conditional_inputs = model.conditional_inputs
        self.conditional_input_names = model.conditional_input_names
        self.data_input_layers = model.data_input_layers
        self.conditional_input_layers = model.conditional_input_layers
        self.generators = model.generators
        self.discriminators = model.discriminators
        self.gen_variables = model.gen_variables
        self.discrim_variables = model.discrim_variables
        self.get_noise_batch = model.get_noise_batch

        super(GANModel, self).__init__(model,
                                       loss=self.gen_loss_fn_wrapper,
                                       **kwargs)

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

    def fit_gan(self,
                batches,
                generator_steps=1,
                max_checkpoints_to_keep=5,
                checkpoint_interval=1000,
                restore=False) -> None:
        """Train this model on data.

        Parameters
        ----------
        batches: iterable
            batches of data to train the discriminator on, each represented as a dict
            that maps Inputs to values.  It should specify values for all members of
            data_inputs and conditional_inputs.
        generator_steps: float
            the number of training steps to perform for the generator for each batch.
            This can be used to adjust the ratio of training steps for the generator
            and discriminator.  For example, 2.0 will perform two training steps for
            every batch, while 0.5 will only perform one training step for every two
            batches.
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in batches.  Set
            this to 0 to disable automatic checkpointing.
        restore: bool
            if True, restore the model from the most recent checkpoint before training
            it.
        """
        self._ensure_built()

        gen_train_fraction = 0.0
        discrim_error = 0.0
        gen_error = 0.0
        discrim_average_steps = 0
        gen_average_steps = 0
        time1 = time.time()

        for feed_dict in batches:
            # Every call to fit_generator() will increment global_step, but we only
            # want it to get incremented once for the entire batch, so record the
            # value and keep resetting it.

            global_step = self.get_global_step()

            # Train the discriminator.

            inputs = [self.get_noise_batch(self.batch_size)]

            for input in self.data_input_names:
                inputs.append(feed_dict[input])
            for input in self.conditional_input_names:
                inputs.append(feed_dict[input])
            discrim_error += self.fit_generator(
                [(inputs, [], [])],
                variables=self.discrim_variables,
                loss=self.discrim_loss_fn_wrapper,
                checkpoint_interval=0,
                restore=restore)
            restore = False
            discrim_average_steps += 1

            # Train the generator.

            if generator_steps > 0.0:
                gen_train_fraction += generator_steps
                while gen_train_fraction >= 1.0:
                    inputs = [self.get_noise_batch(self.batch_size)
                             ] + inputs[1:]

                    gen_error += self.fit_generator(
                        [(inputs, [], [])],
                        variables=self.gen_variables,
                        loss=self.gen_loss_fn_wrapper,
                        checkpoint_interval=0)
                    gen_average_steps += 1
                    gen_train_fraction -= 1.0
            self._global_step = global_step + 1

            # Write checkpoints and report progress.

            if discrim_average_steps == checkpoint_interval:
                self.save_checkpoint(max_checkpoints_to_keep)
                discrim_loss = discrim_error / max(1, discrim_average_steps)
                gen_loss = gen_error / max(1, gen_average_steps)
                print(
                    'Ending global_step %d: generator average loss %g, discriminator average loss %g'
                    % (global_step, gen_loss, discrim_loss))
                discrim_error = 0.0
                gen_error = 0.0
                discrim_average_steps = 0
                gen_average_steps = 0

        # Write out final results.

        if checkpoint_interval > 0:
            if discrim_average_steps > 0 and gen_average_steps > 0:
                discrim_loss = discrim_error / discrim_average_steps
                gen_loss = gen_error / gen_average_steps
                print(
                    'Ending global_step %d: generator average loss %g, discriminator average loss %g'
                    % (global_step, gen_loss, discrim_loss))
            self.save_checkpoint(max_checkpoints_to_keep)
            time2 = time.time()
            print("TIMING: model fitting took %0.3f s" % (time2 - time1))

    def predict_gan_generator(self,
                              batch_size=1,
                              noise_input=None,
                              conditional_inputs=[],
                              generator_index=0):
        """Use the GAN to generate a batch of samples.

        Parameters
        ----------
        batch_size: int
            the number of samples to generate.  If either noise_input or
            conditional_inputs is specified, this argument is ignored since the batch
            size is then determined by the size of that argument.
        noise_input: array
            the value to use for the generator's noise input.  If None (the default),
            get_noise_batch() is called to generate a random input, so each call will
            produce a new set of samples.
        conditional_inputs: list of arrays
            the values to use for all conditional inputs.  This must be specified if
            the GAN has any conditional inputs.
        generator_index: int
            the index of the generator (between 0 and n_generators-1) to use for
            generating the samples.

        Returns
        -------
        An array (if the generator has only one output) or list of arrays (if it has
        multiple outputs) containing the generated samples.
        """
        if noise_input is not None:
            batch_size = len(noise_input)
        elif len(conditional_inputs) > 0:
            batch_size = len(conditional_inputs[0])
        if noise_input is None:
            noise_input = self.get_noise_batch(batch_size)
        inputs = [noise_input]
        inputs += conditional_inputs
        inputs = [i.astype(np.float32) for i in inputs]
        inputs = [torch.from_numpy(i) for i in inputs]
        pred = self.generators[generator_index](_list_or_tensor(inputs))
        pred = pred.detach().numpy()
        return pred
