import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deepchem as dc
from deepchem.models.torch_models import TorchModel
from deepchem.models.torch_models.layers import Lambda


def _list_or_tensor(inputs):

  return inputs[0] if len(inputs) == 1 else inputs


class GAN(nn.Module):

  def __init__(self, n_generators, n_discriminators):

    self.n_generators, self.n_discriminators = n_generators, n_discriminators

    self.generators = []
    self.gen_variables = []

    for i in range(n_generators):
      generator = self.create_generator()
      self.generators.append(generator)
      #generator_outputs.append(generator(_list_or_tensor([self.noise_input] + self.conditional_input_layers)))
      self.gen_variables += generator.trainable_variables

    self.discriminators = []
    self.discrim_variables = []

    for _ in range(n_discriminators):
      discriminator = self.create_discriminator()
      self.discriminators.append(discriminator)

      self.discrim_variables += discriminator.trainable_variables

  def forward(self, inputs):

    noise_input, data_input, conditional_input = inputs
    generator_outputs = []

    for generator in self.generators:
      generator_outputs.append(
          generator(_list_or_tensor([noise_input] + conditional_input)))

    discrim_train_outputs = []
    discrim_gen_outputs = []

    for discriminator in self.discriminators:
      discrim_train_outputs.append(
          self._forward_discriminator(discriminator, data_input, True))

      for gen_output in generator_outputs:
        if isinstance(gen_output, torch.Tensor):
          gen_output = [gen_output]
        discrim_gen_outputs.append(
            self._forward_discriminator(discriminator, gen_output, False))

    gen_losses = [self.create_generator_loss(d) for d in discrim_gen_outputs]

    discrim_losses = []

    for i in range(self.n_discriminators):
      for j in range(self.n_generators):
        discrim_losses.append(
            self.create_discriminator_loss(
                discrim_train_outputs[1],
                discrim_gen_outputs[i * self.n_generators + j]))

    if self.n_generators == 1 and self.n_discriminators == 1:
      total_gen_loss = gen_losses[0]
      total_discrim_loss = discrim_losses[0]
    else:
      gen_alpha = torch.Tensor(torch.ones(1, self.n_generators))
      gen_weights = F.softmax(gen_alpha[noise_input])

      discrim_alpha = torch.Tensor(torch.ones(1, self.n_discriminators))
      discrim_weights = F.softmax(discrim_alpha[noise_input])

  def create_generator(self):
    raise NotImplementedError()

  def create_discriminator(self):
    raise NotImplementedError()

  def _forward_discriminator(self, discriminator, inputs, train):
    return discriminator(inputs + self.conditional_input_layers)

  def create_generator_loss(self, discrim_output):
    return Lambda(lambda x: -torch.mean(torch.log(x + 1e-10)))(discrim_output)

  def create_discriminator_loss(self, discrim_output_train, discrim_output_gen):
    return Lambda(lambda x: -torch.mean(
        torch.log(x[0] + 1e-10) + torch.log(1 - x[1] + 1e-10)))(
            [discrim_output_train, discrim_output_gen])
