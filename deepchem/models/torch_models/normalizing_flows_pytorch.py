"""Normalizing flows for transforming probability distributions using PyTorch.

class NormalizingFlow(nn.Module):
  Normalizing flows are widley used to perform generative models.
  Normalizing flow models gives advantages over variational autoencoders
  (VAE) because of ease in sampling by applying invertible transformations
  (Frey, Gadepally, & Ramsundar, 2022).

  def __init__(self, transform, base_distribution, dim):
      super().__init__()
      self.dim = dim
      self.tranforms = nn.ModuleList(transform)
      self.base_distribution = base_distribution

  def log_prob(self, inputs):
      log_prob = torch.zeros(inputs.shape[0])
      for biject in reversed(self.tranforms):
          inputs, inverse_log_det_jacobian = biject.inverse(inputs)
          log_prob += inverse_log_det_jacobian

      return log_prob

  def sample(self, n_samples):
      outputs = self.base_distribution.sample((n_samples, ))
      log_prob = self.base_distribution.log_prob(outputs)

      for biject in self.transforms:
          output, log_det_jacobian = biject.forward(output)
          log_prob += log_det_jacobian

      return output, log_prob
"""
