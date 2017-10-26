"""Generative Adversarial Networks."""

from deepchem.models import TensorGraph
from deepchem.models.tensorgraph import layers
from collections import Sequence
import numpy as np
import tensorflow as tf
import time


class GAN(TensorGraph):
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
  """

  def __init__(self, **kwargs):
    """Construct a GAN.

    This class accepts all the keyword arguments from TensorGraph.
    """
    super(GAN, self).__init__(use_queue=False, **kwargs)

    # Create the inputs.

    self.noise_input = layers.Feature(shape=self.get_noise_input_shape())
    self.data_inputs = []
    for shape in self.get_data_input_shapes():
      self.data_inputs.append(layers.Feature(shape=shape))
    self.conditional_inputs = []
    for shape in self.get_conditional_input_shapes():
      self.conditional_inputs.append(layers.Feature(shape=shape))

    # Create the generator.

    self.generator = self.create_generator(self.noise_input,
                                           self.conditional_inputs)
    if not isinstance(self.generator, Sequence):
      raise ValueError('create_generator() must return a list of Layers')
    if len(self.generator) != len(self.data_inputs):
      raise ValueError(
          'The number of generator outputs must match the number of data inputs'
      )
    for g, d in zip(self.generator, self.data_inputs):
      if g.shape != d.shape:
        raise ValueError(
            'The shapes of the generator outputs must match the shapes of the data inputs'
        )
    for g in self.generator:
      self.add_output(g)

    # Create the discriminator.

    self.discrim_train = self.create_discriminator(self.data_inputs,
                                                   self.conditional_inputs)

    # Make a copy of the discriminator that takes the generator's output as
    # its input.

    replacements = {}
    for g, d in zip(self.generator, self.data_inputs):
      replacements[d] = g
    for c in self.conditional_inputs:
      replacements[c] = c
    self.discrim_gen = self.discrim_train.copy(replacements, shared=True)

    # Make a list of all layers in the generator and discriminator.

    def add_layers_to_set(layer, layers):
      if layer not in layers:
        layers.add(layer)
        for i in layer.in_layers:
          add_layers_to_set(i, layers)

    gen_layers = set()
    for layer in self.generator:
      add_layers_to_set(layer, gen_layers)
    discrim_layers = set()
    add_layers_to_set(self.discrim_train, discrim_layers)
    discrim_layers -= gen_layers

    # Create submodels for training the generator and discriminator.

    gen_loss = self.create_generator_loss(self.discrim_gen)
    discrim_loss = self.create_discriminator_loss(self.discrim_train,
                                                  self.discrim_gen)
    self.generator_submodel = self.create_submodel(
        layers=gen_layers, loss=gen_loss)
    self.discriminator_submodel = self.create_submodel(
        layers=discrim_layers, loss=discrim_loss)

  def get_noise_input_shape(self):
    """Get the shape of the generator's noise input layer.

    Subclasses must override this to return a tuple giving the shape of the
    noise input.  The actual Input layer will be created automatically.  The
    first dimension must be None, since it will correspond to the batch size.
    """
    raise NotImplementedError("Subclasses must implement this.")

  def get_data_input_shapes(self):
    """Get the shapes of the inputs for training data.

    Subclasses must override this to return a list of tuples, each giving the
    shape of one of the inputs.  The actual Input layers will be created
    automatically.  This list of shapes must also match the shapes of the
    generator's outputs.  The first dimension of each shape must be None, since
    it will correspond to the batch size.
    """
    raise NotImplementedError("Subclasses must implement this.")

  def get_conditional_input_shapes(self):
    """Get the shapes of any conditional inputs.

    Subclasses may override this to return a list of tuples, each giving the
    shape of one of the conditional inputs.  The actual Input layers will be
    created automatically.  The first dimension of each shape must be None,
    since it will correspond to the batch size.

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
    size[0] = batch_size
    return np.random.normal(size=size)

  def create_generator(self, noise_input, conditional_inputs):
    """Create the generator.

    Subclasses must override this to construct the generator and return its
    output layers.

    Parameters
    ----------
    noise_input: Input
      the Input layer from which the generator can read random noise.  The shape
      will match the return value from get_noise_input_shape().
    conditional_inputs: list
      the Input layers for any conditional inputs to the network.  The number
      and shapes of these inputs will match the return value from
      get_conditional_input_shapes().

    Returns
    -------
    A list of Layer objects that produce the generator's outputs.  The number and
    shapes of these layers must match the return value from get_data_input_shapes(),
    since generated data must have the same form as training data.
    """
    raise NotImplementedError("Subclasses must implement this.")

  def create_discriminator(self, data_inputs, conditional_inputs):
    """Create the discriminator.

    Subclasses must override this to construct the discriminator and return its
    output layer.

    Parameters
    ----------
    data_inputs: list
      the Input layers from which the discriminator can read the input data.
      The number and shapes of these inputs will match the return value from
      get_data_input_shapes().  The samples read from these layers may be either
      training data or generated data.
    conditional_inputs: list
      the Input layers for any conditional inputs to the network.  The number
      and shapes of these inputs will match the return value from
      get_conditional_input_shapes().

    Returns
    -------
    A Layer object that outputs the probability of each sample being a training
    sample.  The shape of this layer must be [None].  That is, it must output a
    one dimensional tensor whose length equals the batch size.
    """
    raise NotImplementedError("Subclasses must implement this.")

  def create_generator_loss(self, discrim_output):
    """Create the loss function for the generator.

    The default implementation is appropriate for most cases.  Subclasses can
    override this if the need to customize it.

    Parameters
    ----------
    discrim_output: Layer
      the output from the discriminator on a batch of generated data.  This is
      its estimate of the probability that each sample is training data.

    Returns
    -------
    A Layer object that outputs the loss function to use for optimizing the
    generator.
    """
    return -layers.ReduceMean(layers.Log(discrim_output + 1e-10))

  def create_discriminator_loss(self, discrim_output_train, discrim_output_gen):
    """Create the loss function for the discriminator.

    The default implementation is appropriate for most cases.  Subclasses can
    override this if the need to customize it.

    Parameters
    ----------
    discrim_output_train: Layer
      the output from the discriminator on a batch of generated data.  This is
      its estimate of the probability that each sample is training data.
    discrim_output_gen: Layer
      the output from the discriminator on a batch of training data.  This is
      its estimate of the probability that each sample is training data.

    Returns
    -------
    A Layer object that outputs the loss function to use for optimizing the
    discriminator.
    """
    training_data_loss = layers.Log(discrim_output_train + 1e-10)
    gen_data_loss = layers.Log(1 - discrim_output_gen + 1e-10)
    return -layers.ReduceMean(training_data_loss + gen_data_loss)

  def fit_gan(self,
              batches,
              generator_steps=1.0,
              max_checkpoints_to_keep=5,
              checkpoint_interval=1000,
              restore=False):
    """Train this model on data.

    Parameters
    ----------
    batches: iterable
      batches of data to train the discriminator on, each represented as a dict
      that maps Layers to values.  It should specify values for all members of
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
    if not self.built:
      self.build()
    if restore:
      self.restore()
    gen_train_fraction = 0.0
    discrim_error = 0.0
    gen_error = 0.0
    discrim_average_steps = 0
    gen_average_steps = 0
    time1 = time.time()
    with self._get_tf("Graph").as_default():
      if checkpoint_interval > 0:
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
      for feed_dict in batches:
        # Every call to fit_generator() will increment global_step, but we only
        # want it to get incremented once for the entire batch, so record the
        # value and keep resetting it.

        global_step = self.global_step

        # Train the discriminator.

        feed_dict = dict(feed_dict)
        feed_dict[self.noise_input] = self.get_noise_batch(self.batch_size)
        discrim_error += self.fit_generator(
            [feed_dict],
            submodel=self.discriminator_submodel,
            checkpoint_interval=0)
        self.global_step = global_step
        discrim_average_steps += 1

        # Train the generator.

        if generator_steps > 0.0:
          gen_train_fraction += generator_steps
          while gen_train_fraction >= 1.0:
            feed_dict[self.noise_input] = self.get_noise_batch(self.batch_size)
            gen_error += self.fit_generator(
                [feed_dict],
                submodel=self.generator_submodel,
                checkpoint_interval=0)
            self.global_step = global_step
            gen_average_steps += 1
            gen_train_fraction -= 1.0
        self.global_step = global_step + 1

        # Write checkpoints and report progress.

        if discrim_average_steps == checkpoint_interval:
          saver.save(self.session, self.save_file, global_step=self.global_step)
          discrim_loss = discrim_error / max(1, discrim_average_steps)
          gen_loss = gen_error / max(1, gen_average_steps)
          print(
              'Ending global_step %d: generator average loss %g, discriminator average loss %g'
              % (self.global_step, gen_loss, discrim_loss))
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
              % (self.global_step, gen_loss, discrim_loss))
        saver.save(self.session, self.save_file, global_step=self.global_step)
        time2 = time.time()
        print("TIMING: model fitting took %0.3f s" % (time2 - time1))

  def predict_gan_generator(self,
                            batch_size=1,
                            noise_input=None,
                            conditional_inputs=[]):
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
    batch = {}
    batch[self.noise_input] = noise_input
    for layer, value in zip(self.conditional_inputs, conditional_inputs):
      batch[layer] = value
    return self.predict_on_generator([batch])

  def _set_empty_inputs(self, feed_dict, layers):
    """Set entries in a feed dict corresponding to a batch size of 0."""
    for layer in layers:
      shape = list(layer.shape)
      shape[0] = 0
      feed_dict[layer] = np.zeros(shape)


class WGAN(GAN):
  """Implements Wasserstein Generative Adversarial Networks.

  This class implements Wasserstein Generative Adversarial Networks (WGANs) as
  described in Arjovsky et al., "Wasserstein GAN" (https://arxiv.org/abs/1701.07875).
  A WGAN is conceptually rather different from a conventional GAN, but in
  practical terms very similar.  It reinterprets the discriminator (often called
  the "critic" in this context) as learning an approximation to the Earth Mover
  distance between the training and generated distributions.  The generator is
  then trained to minimize that distance.  In practice, this just means using
  slightly different loss functions for training the generator and discriminator.

  WGANs have theoretical advantages over conventional GANs, and they often work
  better in practice.  In addition, the discriminator's loss function can be
  directly interpreted as a measure of the quality of the model.  That is an
  advantage over conventional GANs, where the loss does not directly convey
  information about the quality of the model.

  The theory WGANs are based on requires the discriminator's gradient to be
  bounded.  The original paper achieved this by clipping its weights.  This
  class instead does it by adding a penalty term to the discriminator's loss, as
  described in https://arxiv.org/abs/1704.00028.  This is sometimes found to
  produce better results.

  There are a few other practical differences between GANs and WGANs.  In a
  conventional GAN, the discriminator's output must be between 0 and 1 so it can
  be interpreted as a probability.  In a WGAN, it should produce an unbounded
  output that can be interpreted as a distance.

  When training a WGAN, you also should usually use a smaller value for
  generator_steps.  Conventional GANs rely on keeping the generator and
  discriminator "in balance" with each other.  If the discriminator ever gets
  too good, it becomes impossible for the generator to fool it and training
  stalls.  WGANs do not have this problem, and in fact the better the
  discriminator is, the easier it is for the generator to improve.  It therefore
  usually works best to perform several training steps on the discriminator for
  each training step on the generator.
  """

  def __init__(self, gradient_penalty=10.0, **kwargs):
    """Construct a WGAN.

    In addition to the following, this class accepts all the keyword arguments
    from TensorGraph.

    Parameters
    ----------
    gradient_penalty: float
      the magnitude of the gradient penalty loss
    """
    super(WGAN, self).__init__(**kwargs)
    self.gradient_penalty = gradient_penalty

  def create_generator_loss(self, discrim_output):
    return layers.ReduceMean(discrim_output)

  def create_discriminator_loss(self, discrim_output_train, discrim_output_gen):
    gradient_penalty = GradientPenaltyLayer(discrim_output_train, self)
    return gradient_penalty + layers.ReduceMean(discrim_output_train -
                                                discrim_output_gen)


class GradientPenaltyLayer(layers.Layer):
  """Implements the gradient penalty loss term for WGANs."""

  def __init__(self, discrim_output_train, gan):
    super(GradientPenaltyLayer, self).__init__([discrim_output_train])
    self.gan = gan

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):
    gradients = tf.gradients(self.in_layers[0], self.gan.data_inputs)
    norm2 = 0.0
    for g in gradients:
      g2 = tf.square(g)
      dims = len(g.shape)
      if dims > 1:
        g2 = tf.reduce_sum(g2, axis=list(range(1, dims)))
      norm2 += g2
    penalty = tf.square(tf.sqrt(norm2) - 1.0)
    self.out_tensor = self.gan.gradient_penalty * tf.reduce_mean(penalty)
    return self.out_tensor
