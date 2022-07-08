import math
from typing import Tuple

from attr import has

try:
  import torch
  import torch.optim as optim
  has_torch = True

except ModuleNotFoundError:
  has_torch = False

class KFACOptimizer(optim.Optimizer):
  """"
  This class implement the second order optimizer - KFAC, which uses Kronecker factor products of inputs and the gradients to
  get the approximate inverse fisher matrix, which is used to update the model parameters. Presently this optimizer works only
  on liner and 2D convolution layers. If you want to know more details about KFAC, please check the paper [1]_.

  References:
  -----------
  Martens, James, and Roger Grosse. Optimizing Neural Networks with Kronecker-Factored Approximate Curvature.
  arXiv:1503.05671, arXiv, 7 June 2020. arXiv.org, http://arxiv.org/abs/1503.05671.
  """

  def __init__(self,
               model:torch.nn.Module,
               lr: float = 0.001,
               momentum: float = 0.9,
               stat_decay: float = 0.95,
               damping: float = 0.001,
               kl_clip: float = 0.001,
               weight_decay: float = 0,
               TCov: int = 10,
               TInv: int = 100,
               batch_averaged: bool = True,
               mean:bool=False):
    """
    Parameters:
    -----------
    model: torch.nn.Module
    The model to be optimized.
    lr: float
    Learning rate for the optimizer.
    momentum: float
    Momentum for the optimizer.
    stat_decay: float
    Decay rate for the update of covariance matrix with mean.
    damping: float
    damoing factor for the update of covariance matrix.
    kl_clip: float
    Clipping value for the update of covariance matrix.
    weight_decay: float
    weight decay for the optimizer.
    Tcov: int
    The number of steps to update the covariance matrix.
    Tinv: int
    The number of steps to calculate the inverse of covariance matrix.
    batch_averaged: bool
    States whether to use batch averaged covariance matrix.
    mean: bool
    States whether to use mean centered covariance matrix.
    """

    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
    defaults = dict(lr=lr,
                    momentum=momentum,
                    damping=damping,
                    weight_decay=weight_decay)
    super(KFACOptimizer, self).__init__(model.parameters(), defaults)
    self.batch_averaged = batch_averaged

    self.known_modules = {'Linear', 'Conv2d'}

    self.modules = []
    self.grad_outputs = {}

    self.model = model
    self._prepare_model()

    self.steps = 0

    self.m_aa, self.m_gg = {}, {}
    self.Q_a, self.Q_g = {}, {}
    self.d_a, self.d_g = {}, {}
    self.stat_decay = stat_decay

    self.kl_clip = kl_clip
    self.TCov = TCov
    self.TInv = TInv
    
    self.mean=mean

  def try_contiguous(self,x: torch.Tensor) -> torch.Tensor:
    """
    x: torch.Tensor
    The input tensor to be made contiguous in memory, if it is not so.
    """
    if not x.is_contiguous():
      x = x.contiguous()

    return x


  def _extract_patches(self,x: torch.Tensor, kernel_size: Tuple[int, int],
                     stride: Tuple[int, int],
                     padding: Tuple[int, int]) -> torch.Tensor:
    """
    
    Parameters:
    -----------
    x: Tuple[int, int, int, int]
    The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
      x = torch.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
      x = x.unfold(2, kernel_size[0], stride[0])
      x = x.unfold(3, kernel_size[1], stride[1])
      x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
      x = x.view(x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4) * x.size(5))
    return x
  
  def ComputeCovA(self, a: torch.Tensor , layer: torch.nn.Module) -> torch.Tensor:
    """
    Compute the covariance matrix of the A matrix (the output of each layer).
    
    Parameters:
    -----------
    a: torch.Tensor
    It is the output of the layer for which the covariance matrix should be calculated.
    layer: torch.nn.Module
    It specifies the type of layer from which the output of the layer is taken.
    
    Returns:
    --------
    torch.Tensor
    The covariance matrix of the A matrix.
    """
    if isinstance(layer, torch.linear):
      batch_size = a.size(0)
      if layer.bias is not None:
        a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
      return a.t() @ (a / batch_size)

    elif isinstance(self, layer, torch.Conv2d):
      batch_size = a.size(0)
      a = self._extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
      spatial_size = a.size(1) * a.size(2)
      a = a.view(-1, a.size(-1))
      if layer.bias is not None:
        a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
      a = a / spatial_size
    return a.t() @ (a / batch_size)

  def ComputeCovG(self, g:torch.Tensor, layer:torch.nn.Module) -> torch.Tensor:
    """
    Compute the covariance matrix of the G matrix (the gradient of the layer).
    
    Parameters:
    -----------
    g: torch.Tensor
    It is the gradient of the layer for which the covariance matrix should be calculated.
    layer: torch.nn.Module
    It specifies the type of layer from which the output of the layer is taken.
    
    Returns:
    --------
    torch.Tensor
    The covariance matrix of the G matrix.
    """
    if isinstance(layer, torch.linear):
      batch_size = g.size(0)
      if self.batch_averaged:
        cov_g = g.t() @ (g * batch_size)
      else:
        cov_g = g.t() @ (g / batch_size)
      return cov_g

    elif isinstance(layer, torch.Conv2d):
      spatial_size = g.size(2) * g.size(3)
      batch_size = g.shape[0]
      g = g.transpose(1, 2).transpose(2, 3)
      g = self.try_contiguous(g)
      g = g.view(-1, g.size(-1))
      if self.batch_averaged:
        g = g * batch_size
      g = g * spatial_size
      cov_g = g.t() @ (g / g.size(0))

      return cov_g

  def _save_input(self, module:str, input: torch.Tensor):
    """
    Saves the input of the layer.
    """
    if torch.is_grad_enabled() and self.steps % self.TCov == 0:
      aa = self.ComputeCovA(input[0].data, module)
      # Initialize buffers
      if self.steps == 0:
        self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
      self.m_aa[module] *= self.stat_decay + aa * (1 - self.stat_decay)

  def _save_grad_output(self, module: str, grad_input:torch.Tensor, grad_output:torch.Tensor):
    # Accumulate statistics for Fisher matrices
    if self.acc_stats and self.steps % self.TCov == 0:
      gg = self.ComputeCovG(grad_output[0].data, module, self.batch_averaged)
      # Initialize buffers
      if self.steps == 0:
        self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
      self.m_gg[module] *= self.stat_decay + gg * (1 - self.stat_decay)

  def _prepare_model(self):
    count = 0
    print(self.model)
    print("=> We keep following layers in KFAC. ")
    for module in self.model.modules():
      classname = module.__class__.__name__
      # print('=> We keep following layers in KFAC. <=')
      if classname in self.known_modules:
        self.modules.append(module)
        module.register_forward_pre_hook(self._save_input)
        module.register_backward_hook(self._save_grad_output)
        print('(%s): %s' % (count, module))
        count += 1

  def _update_inv(self, m):
    """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
    eps = 1e-10  # for numerical stability

    if self.mean:
      self.d_a[m], self.Q_a[m] = torch.symeig(self.m_aa[m] -
                                              torch.mean(self.m_aa[m]),
                                              eigenvectors=True)
      self.d_g[m], self.Q_g[m] = torch.symeig(self.m_gg[m] -
                                              torch.mean(self.m_gg[m]),
                                              eigenvectors=True)
    else:
      self.d_a[m], self.Q_a[m] = torch.symeig(self.m_aa[m], eigenvectors=True)
      self.d_g[m], self.Q_g[m] = torch.symeig(self.m_gg[m], eigenvectors=True)

    self.d_a[m].mul_((self.d_a[m] > eps).float())
    self.d_g[m].mul_((self.d_g[m] > eps).float())

  @staticmethod
  def _get_matrix_form_grad(m, classname):
    """
    :param m: the layer
    :param classname: the class name of the layer
    :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
    """
    if classname == 'Conv2d':
      p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0),
                                           -1)  # n_filters * (in_c * kw * kh)
    else:
      p_grad_mat = m.weight.grad.data
    if m.bias is not None:
      p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
    return p_grad_mat

  def _get_natural_grad(self, m, p_grad_mat, damping):
    """
    :param m:  the layer
    :param p_grad_mat: the gradients in matrix form
    :return: a list of gradients w.r.t to the parameters in `m`
    """
    # p_grad_mat is of output_dim * input_dim
    # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
    v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
    v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
    v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
    if m.bias is not None:
      # we always put gradient w.r.t weight in [0]
      # and w.r.t bias in [1]
      v = [v[:, :-1], v[:, -1:]]
      v[0] = v[0].view(m.weight.grad.data.size())
      v[1] = v[1].view(m.bias.grad.data.size())
    else:
      v = [v.view(m.weight.grad.data.size())]

    return v

  def _kl_clip_and_update_grad(self, updates, lr):
    # do kl clip
    vg_sum = 0
    for m in self.modules:
      v = updates[m]
      vg_sum += (v[0] * m.weight.grad.data * lr**2).sum().item()
      if m.bias is not None:
        vg_sum += (v[1] * m.bias.grad.data * lr**2).sum().item()
    nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

    for m in self.modules:
      v = updates[m]
      m.weight.grad.data.copy_(v[0])
      m.weight.grad.data.mul_(nu)
      if m.bias is not None:
        m.bias.grad.data.copy_(v[1])
        m.bias.grad.data.mul_(nu)

  def _step(self, closure):
    for group in self.param_groups:
      weight_decay = group['weight_decay']
      momentum = group['momentum']

      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data
        if weight_decay != 0 and self.steps >= 20 * self.TCov:
          d_p.add_(weight_decay, p.data)
        if momentum != 0:
          param_state = self.state[p]
          if 'momentum_buffer' not in param_state:
            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            buf.mul_(momentum).add_(d_p)
          else:
            buf = param_state['momentum_buffer']
            buf.mul_(momentum).add_(1, d_p)
          d_p = buf

        p.data.add_(-group['lr'], d_p)

  def step(self, closure: bool = None):
    """
    This is the function that gets called in each step of the optimizer to update the weights and biases of the model.
    
    Parameters:
    -----------
    closure: bool
    Gives the closure
    """
    group = self.param_groups[0]
    lr = group['lr']
    damping = group['damping']
    updates = {}
    for m in self.modules:
      classname = m.__class__.__name__
      if self.steps % self.TInv == 0:
        self._update_inv(m)
      p_grad_mat = self._get_matrix_form_grad(m, classname)
      v = self._get_natural_grad(m, p_grad_mat, damping)
      updates[m] = v
    self._kl_clip_and_update_grad(updates, lr)

    self._step(closure)
    self.steps += 1