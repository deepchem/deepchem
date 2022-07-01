import math

import torch
import torch.optim as optim

# Must add covariance matrix calculations. Also should fix todos and fixmes


class KFACOptimizer(optim.Optimizer):

  def __init__(self,
               model,
               lr=0.001,
               momentum=0.9,
               stat_decay=0.95,
               damping=0.001,
               kl_clip=0.001,
               weight_decay=0,
               TCov=10,
               TInv=100,
               batch_averaged=True,
               mean=True):
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
    # TODO (CW): KFAC optimizer now only support model as input
    super(KFACOptimizer, self).__init__(model.parameters(), defaults)
    self.CovAHandler = ComputeCovA()
    self.CovGHandler = ComputeCovG()
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

    self.mean = True

  def _save_input(self, module, input):
    """
    	This function lets the update of the expected matrix of the ouptut of a layer and its activation.
    	"""
    if torch.is_grad_enabled() and self.steps % self.TCov == 0:
      aa = self.CovAHandler(input[0].data, module)
      # Initialize buffers
      if self.steps == 0:
        self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
      update_running_stat(aa, self.m_aa[module], self.stat_decay)

  def _save_grad_output(self, module, grad_input, grad_output):
    # Accumulate statistics for Fisher matrices
    if self.acc_stats and self.steps % self.TCov == 0:
      gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
      # Initialize buffers
      if self.steps == 0:
        self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
      update_running_stat(gg, self.m_gg[module], self.stat_decay)

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

    if self.mean:  # Modifications for Ferminet
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
    # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
    # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
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

  def step(self, closure=None):
    # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
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
