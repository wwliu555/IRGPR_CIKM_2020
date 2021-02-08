import math
import random
random.seed(1234)

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, GRU
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch_geometric.utils import to_undirected

from torch_geometric.nn.inits import reset

from util import recommend

EPS = 1e-15
MAX_LOGVAR = 10
edge_type = 4



def negative_sampling(pos_edge_index, num_nodes):
    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes**2)
    perm = torch.tensor(random.sample(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0)


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def __init__(self, node_hidden_dim):
        super(InnerProductDecoder, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.fc = nn.Linear(self.node_hidden_dim * 2, 1)


    def forward(self, z, edge_index):
        z_in = torch.cat((z[edge_index[0]], z[edge_index[1]]), dim=1)
        value = self.fc(z_in)
        return value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj



class GNNRanker(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, node_hidden_dim, encoder, decoder=None):
        super(GNNRanker, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.encoder = encoder
        self.decoder = InnerProductDecoder(self.node_hidden_dim) if decoder is None else decoder
        GNNRanker.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def split_edges(self, data):
        data.test_data = data.test_data.long()
        data.test_edge_index = data.test_data[:, :2].transpose(0, 1)
        data.test_y = data.test_data[:, 2]
        data.train_edge_index = data.train_data[:, :2].transpose(0, 1)
        data.train_y = data.train_data[:, 2] 
        return data

    def get_batch(self, data, i, bs):
        return data.train_pos_edge_index[:, i*bs:(i+1)*bs], data.train_pos_edge_attr[i*bs:(i+1)*bs, :]


    def cross_entropy_loss(self, z, edge_index, label):
        l = F.binary_cross_entropy_with_logits(
            self.decoder(z, edge_index), torch.reshape(label.float(), (-1, 1)))
        return l

    def bpr_loss(self, z, edge_index, label):
        pred = torch.sigmoid(self.decoder(z, edge_index))
        mask_pos = (label == 1)
        mask_neg = (label == 0)
        length = min(mask_pos.sum(), mask_neg.sum())
        l = -torch.mean(torch.log(torch.sigmoid(pred[mask_pos][:length] - pred[mask_neg][:length])))
        return l


    def test(self, z, edge_index, y):
        pred = self.decoder(z, edge_index)
        edge_index, y, pred = edge_index.detach().cpu().numpy(), y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        precision_at_5, map_at_5, precision_at_10, map_at_10, precision_at_20, map_at_20 = recommend(edge_index, y, pred)
        
        return precision_at_5, map_at_5, precision_at_10, map_at_10, precision_at_20, map_at_20
















