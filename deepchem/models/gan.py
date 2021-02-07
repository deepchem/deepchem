"""Generative Adversarial Networks."""

from deepchem.models import KerasModel, layers, losses
from tensorflow.keras.layers import Input, Lambda, Layer, Softmax, Reshape, Multiply
try:
  from collections.abc import Sequence
except:
  from collections import Sequence
import numpy as np
import tensorflow as tf
import time


class GAN(KerasModel):
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

  def __init__(self, n_generators=1, n_discriminators=1, **kwargs):
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

    self.noise_input = Input(shape=self.get_noise_input_shape())
    self.data_input_layers = []
    for shape in self.get_data_input_shapes():
      self.data_input_layers.append(Input(shape=shape))
    self.data_inputs = [i.ref() for i in self.data_input_layers]
    self.conditional_input_layers = []
    for shape in self.get_conditional_input_shapes():
      self.conditional_input_layers.append(Input(shape=shape))
    self.conditional_inputs = [i.ref() for i in self.conditional_input_layers]

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
          self._call_discriminator(discriminator, self.data_input_layers, True))
      for gen_output in generator_outputs:
        if tf.is_tensor(gen_output):
          gen_output = [gen_output]
        discrim_gen_outputs.append(
            self._call_discriminator(discriminator, gen_output, False))
      self.discrim_variables += discriminator.trainable_variables

    # Compute the loss functions.

    gen_losses = [self.create_generator_loss(d) for d in discrim_gen_outputs]
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

      gen_alpha = layers.Variable(np.ones((1, n_generators)), dtype=tf.float32)
      # We pass an input to the Variable layer to work around a bug in TF 1.14.
      gen_weights = Softmax()(gen_alpha([self.noise_input]))
      discrim_alpha = layers.Variable(
          np.ones((1, n_discriminators)), dtype=tf.float32)
      discrim_weights = Softmax()(discrim_alpha([self.noise_input]))

      # Compute the weighted errors

      weight_products = Reshape((n_generators * n_discriminators,))(Multiply()([
          Reshape((n_discriminators, 1))(discrim_weights),
          Reshape((1, n_generators))(gen_weights)
      ]))
      stacked_gen_loss = layers.Stack(axis=0)(gen_losses)
      stacked_discrim_loss = layers.Stack(axis=0)(discrim_losses)
      total_gen_loss = Lambda(lambda x: tf.reduce_sum(x[0] * x[1]))(
          [stacked_gen_loss, weight_products])
      total_discrim_loss = Lambda(lambda x: tf.reduce_sum(x[0] * x[1]))(
          [stacked_discrim_loss, weight_products])
      self.gen_variables += gen_alpha.trainable_variables
      self.discrim_variables += gen_alpha.trainable_variables
      self.discrim_variables += discrim_alpha.trainable_variables

      # Add an entropy term to the loss.

      entropy = Lambda(lambda x: -(tf.reduce_sum(tf.math.log(x[0]))/n_generators +
          tf.reduce_sum(tf.math.log(x[1]))/n_discriminators))([gen_weights, discrim_weights])
      total_discrim_loss = Lambda(lambda x: x[0] + x[1])(
          [total_discrim_loss, entropy])

    # Create the Keras model.

    inputs = [self.noise_input
             ] + self.data_input_layers + self.conditional_input_layers
    outputs = [total_gen_loss, total_discrim_loss]
    self.gen_loss_fn = lambda outputs, labels, weights: outputs[0]
    self.discrim_loss_fn = lambda outputs, labels, weights: outputs[1]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    super(GAN, self).__init__(model, self.gen_loss_fn, **kwargs)

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
    return Lambda(lambda x: -tf.reduce_mean(tf.math.log(x + 1e-10)))(
        discrim_output)

  def create_discriminator_loss(self, discrim_output_train, discrim_output_gen):
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
    return Lambda(lambda x: -tf.reduce_mean(tf.math.log(x[0]+1e-10) + tf.math.log(1-x[1]+1e-10)))([discrim_output_train, discrim_output_gen])

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
    if checkpoint_interval > 0:
      manager = tf.train.CheckpointManager(self._checkpoint, self.model_dir,
                                           max_checkpoints_to_keep)
    for feed_dict in batches:
      # Every call to fit_generator() will increment global_step, but we only
      # want it to get incremented once for the entire batch, so record the
      # value and keep resetting it.

      global_step = self.get_global_step()

      # Train the discriminator.

      inputs = [self.get_noise_batch(self.batch_size)]
      for input in self.data_input_layers:
        inputs.append(feed_dict[input.ref()])
      for input in self.conditional_input_layers:
        inputs.append(feed_dict[input.ref()])
      discrim_error += self.fit_generator(
          [(inputs, [], [])],
          variables=self.discrim_variables,
          loss=self.discrim_loss_fn,
          checkpoint_interval=0,
          restore=restore)
      restore = False
      discrim_average_steps += 1

      # Train the generator.

      if generator_steps > 0.0:
        gen_train_fraction += generator_steps
        while gen_train_fraction >= 1.0:
          inputs = [self.get_noise_batch(self.batch_size)] + inputs[1:]
          gen_error += self.fit_generator(
              [(inputs, [], [])],
              variables=self.gen_variables,
              checkpoint_interval=0)
          gen_average_steps += 1
          gen_train_fraction -= 1.0
      self._global_step.assign(global_step + 1)

      # Write checkpoints and report progress.

      if discrim_average_steps == checkpoint_interval:
        manager.save()
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
      manager.save()
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
    pred = self.generators[generator_index](
        _list_or_tensor(inputs), training=False)
    pred = pred.numpy()
    return pred


def _list_or_tensor(inputs):
  if len(inputs) == 1:
    return inputs[0]
  return inputs


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
    from GAN and KerasModel.

    Parameters
    ----------
    gradient_penalty: float
      the magnitude of the gradient penalty loss
    """
    self.gradient_penalty = gradient_penalty
    super(WGAN, self).__init__(**kwargs)

  def _call_discriminator(self, discriminator, inputs, train):
    if train:
      penalty = GradientPenaltyLayer(self, discriminator)
      return penalty(inputs, self.conditional_input_layers)
    return discriminator(
        _list_or_tensor(inputs + self.conditional_input_layers))

  def create_generator_loss(self, discrim_output):
    return Lambda(lambda x: tf.reduce_mean(x))(discrim_output)

  def create_discriminator_loss(self, discrim_output_train, discrim_output_gen):
    return Lambda(lambda x: tf.reduce_mean(x[0] - x[1]))(
        [discrim_output_train[0], discrim_output_gen]) + discrim_output_train[1]


class GradientPenaltyLayer(Layer):
  """Implements the gradient penalty loss term for WGANs."""

  def __init__(self, gan, discriminator, **kwargs):
    super(GradientPenaltyLayer, self).__init__(**kwargs)
    self.gan = gan
    self.discriminator = discriminator

  def call(self, inputs, conditional_inputs):
    with tf.GradientTape() as tape:
      for layer in inputs:
        tape.watch(layer)
      output = self.discriminator(_list_or_tensor(inputs + conditional_inputs))
    gradients = tape.gradient(output, inputs)
    gradients = [g for g in gradients if g is not None]
    if len(gradients) > 0:
      norm2 = 0.0
      for g in gradients:
        g2 = tf.square(g)
        dims = len(g.shape)
        if dims > 1:
          g2 = tf.reduce_sum(g2, axis=list(range(1, dims)))
        norm2 += g2
      penalty = tf.square(tf.sqrt(norm2) - 1.0)
      penalty = self.gan.gradient_penalty * tf.reduce_mean(penalty)
    else:
      penalty = 0.0
    return [output, penalty]
