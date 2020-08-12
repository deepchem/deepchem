import torch
import torch.nn as nn
import torch.nn.functional as F


class CGCNNLayer(nn.Module):
  def __init__(self, atom_fea_len: int, nbr_fea_len: int, batch_norm: bool = True):
    """
    Parameters
    ----------
    atom_fea_len: int
      Number of atom hidden features.
    nbr_fea_len: int
      Number of edge features.
    batch_norm: bool, default True
      Whether to apply batch normalization or not.
    """
    super(CGCNNLayer, self).__init__()
    z_dim = 2 * atom_fea_len + nbr_fea_len
    self.linear_with_sigmoid = nn.Linear(z_dim, atom_fea_len)
    self.linear_with_softplus = nn.Linear(z_dim, atom_fea_len)
    self.batch_norm = nn.BatchNorm1d(atom_fea_len) if batch_norm else None

  def message_func(self, edges):
    z = torch.cat([edges.src['x'], edges.dst['x'], edges.data], dim=1)
    gated_z = F.sigmoid(self.linear_with_sigmoid(z))
    message_z = F.softplus(self.linear_with_softplus(z))
    return {'gated_z': gated_z, 'message_z': message_z}

  def reduce_func(self, nodes):
    new_h = nodes.data + torch.sum(nodes.mailbox['gated_z'] * nodes.mailbox['message_z'], dim=1)
    if self.batch_norm is not None:
        new_h = self.batch_norm(new_h)
    return {'h': new_h}

  def forward(self, dgl_graph):
    dgl_graph.update_all(self.message_func, self.reduce_func)
    return dgl_graph
