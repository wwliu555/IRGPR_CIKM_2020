import os
import os.path as osp
import numpy as np
import pickle as pkl
from tqdm import tqdm
import math
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential, Linear, ReLU, GRU
from nn_rerank_conv import IRGPRConv
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from model import GNNRanker
from sampler_sampling import NeighborSampler


from amazon_data_loader import AmazonDataset


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--node_emb', type=int, default=4)
args = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# model parameters
n_epoch = 300
num_edge_type = 4
node_hidden_dim = edge_hidden_dim = args.node_emb
num_step_message_passing = 1
lr = args.lr

# print("learning rate: {}".format(lr))
# print("node embedding dim: {}".format(node_hidden_dim))

save_dir = "run/models_mpnn/"
if not osp.exists(save_dir):
  os.makedirs(save_dir)
  print('made directory {}'.format(save_dir))


dataset = AmazonDataset("./data/Amazon")
data = torch.load(dataset.processed_paths[0])
data = dataset[0]



class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, node_hidden_dim)
        self.rating_network = Linear(node_hidden_dim, node_hidden_dim * node_hidden_dim)
        self.personalized_network = Linear(num_edge_type, node_hidden_dim * node_hidden_dim)
        self.edge_network = Linear(num_edge_type + node_hidden_dim, node_hidden_dim * node_hidden_dim)
        self.conv = IRGPRConv(node_hidden_dim, node_hidden_dim, self.rating_network, 
            self.personalized_network, self.edge_network, aggr='mean', root_weight=False)
        self.gru = GRU(node_hidden_dim, node_hidden_dim)


    def forward(self, x, edge_index, edge_attr, is_user):
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(num_step_message_passing):
            prev = h
            m = F.relu(self.conv(out, edge_index, edge_attr, is_user))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = F.relu(out)

        return out

model = GNNRanker(node_hidden_dim, Encoder()).to(dev)

data.train_mask = data.val_mask = data.test_mask = None
data = model.split_edges(data).to(dev)

optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index, data.edge_attr, data.is_user)
    loss = model.cross_entropy_loss(z, data.train_edge_index, data.train_y)
    loss.backward()
    optimizer.step()

    return loss, z


def test(z, edge_index, y):
    model.eval()    
    return model.test(z, edge_index, y)


for epoch in range(1, n_epoch + 1):
    tr_loss, z = train()
    precision_at_5, map_at_5, precision_at_10, map_at_10, precision_at_20, map_at_20 = test(z, data.test_edge_index, data.test_y)
    result = 'Epoch: {:03d}, tr-loss: {:.4f}, precision@5: {:.4f}, map@5: {:.4f},\
    precision@10: {:.4f},  map@10: {:.4f},\
    precision@20: {:.4f},  map@20: {:.4f}'
    print(result.format(epoch, tr_loss, precision_at_5, map_at_5, precision_at_10, map_at_10, precision_at_20, map_at_20))




