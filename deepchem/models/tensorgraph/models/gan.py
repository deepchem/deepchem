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
    self.noise_conditional_inputs = []
    for shape in self.get_conditional_input_shapes():
      self.conditional_inputs.append(layers.Feature(shape=shape))
      self.noise_conditional_inputs.append(layers.Feature(shape=shape))
    self.is_training = layers.Weights(shape=(None, 1))

    # Create the generator.

    self.generator = self.create_generator(self.noise_input,
                                           self.noise_conditional_inputs)
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

    discrim_data = []
    for g, d in zip(self.generator, self.data_inputs):
      discrim_data.append(layers.Concat([g, d], axis=0))
    discrim_conditional = []
    for n, c in zip(self.noise_conditional_inputs, self.conditional_inputs):
      discrim_conditional.append(layers.Concat([n, c], axis=0))
    self.discriminator = self.create_discriminator(discrim_data,
                                                   discrim_conditional)

    #if self.discriminator.shape != (None,):
    #  raise ValueError('Incorrect shape for discriminator output')

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
    add_layers_to_set(self.discriminator, discrim_layers)
    discrim_layers -= gen_layers

    # Create submodels for training the generator and discriminator.

    gen_loss = self.create_generator_loss(self.discriminator, self.is_training)
    discrim_loss = self.create_discriminator_loss(self.discriminator,
                                                  self.is_training)
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

  def create_generator_loss(self, discrim_output, is_training):
    """Create the loss function for the generator.

    The default implementation is appropriate for most cases.  Subclasses can
    override this if the need to customize it.

    Parameters
    ----------
    discrim_output: Layer
      the discriminator's output layer, which computes the probability that each
      sample is training data.
    is_training: Layer
      outputs a set of flags indicating whether each sample is actually training
      data (1) or generated data (0).

    Returns
    -------
    A Layer object that outputs the loss function to use for optimizing the
    generator.
    """
    prob = discrim_output + 1e-10
    return -layers.ReduceMean(layers.Log(prob) * (1 - is_training))

  def create_discriminator_loss(self, discrim_output, is_training):
    """Create the loss function for the discriminator.

    The default implementation is appropriate for most cases.  Subclasses can
    override this if the need to customize it.

    Parameters
    ----------
    discrim_output: Layer
      the discriminator's output layer, which computes the probability that each
      sample is training data.
    is_training: Layer
      outputs a set of flags indicating whether each sample is actually training
      data (1) or generated data (0).

    Returns
    -------
    A Layer object that outputs the loss function to use for optimizing the
    discriminator.
    """
    training_data_loss = layers.Log(discrim_output + 1e-10) * is_training
    gen_data_loss = layers.Log(1 - discrim_output + 1e-10) * (1 - is_training)
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

        # Train the discriminator on training data.

        feed_dict = dict(feed_dict)
        feed_dict[self.noise_input] = self.get_noise_batch(0)
        feed_dict[self.is_training] = np.ones((self.batch_size, 1))
        self._set_empty_inputs(feed_dict, self.noise_conditional_inputs)
        discrim_error += self.fit_generator(
            [feed_dict],
            submodel=self.discriminator_submodel,
            checkpoint_interval=0)
        self.global_step = global_step

        # Train the discriminator on generated data.

        feed_dict[self.noise_input] = self.get_noise_batch(self.batch_size)
        feed_dict[self.is_training] = np.zeros((self.batch_size, 1))
        for n, c in zip(self.noise_conditional_inputs, self.conditional_inputs):
          feed_dict[n] = feed_dict[c]
        self._set_empty_inputs(feed_dict, self.data_inputs)
        self._set_empty_inputs(feed_dict, self.conditional_inputs)
        discrim_error += self.fit_generator(
            [feed_dict],
            submodel=self.discriminator_submodel,
            checkpoint_interval=0)
        self.global_step = global_step
        discrim_average_steps += 1

        # Train the generator.

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
          discrim_loss = discrim_error / discrim_average_steps
          gen_loss = gen_error / gen_average_steps
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
    for layer, value in zip(self.noise_conditional_inputs, conditional_inputs):
      batch[layer] = value
    return self.predict_on_generator([batch])

  def _set_empty_inputs(self, feed_dict, layers):
    """Set entries in a feed dict corresponding to a batch size of 0."""
    for layer in layers:
      shape = list(layer.shape)
      shape[0] = 0
      feed_dict[layer] = np.zeros(shape)
