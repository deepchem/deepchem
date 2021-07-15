import math
try:
  import torch
  import torch.nn as nn
except:
  raise ImportError('These classes require Torch to be installed.')


class ScaleNorm(nn.Module):
  """Apply Scale Normalization to input.

    All G values are initialized to sqrt(d).

    References
    ----------
    .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
    """

  def __init__(self, scale, eps=1e-5):
    """Initialize a ScaleNorm layer.

        Parameters
        ----------
        scale: Real number or single element tensor
          Scale magnitude.
        eps: float
          Epsilon value.
        """

    super(ScaleNorm, self).__init__()
    self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
    self.eps = eps

  def forward(self, x):
    norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
    return x * norm
