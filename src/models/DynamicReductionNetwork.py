import os
import os.path as osp
import math

import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph

from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool.edge_pool import EdgePooling

class DynamicReductionNetwork(nn.Module):
    # This model iteratively contracts nearest neighbour graphs 
    # until there is one output node.
    # The latent space trained to group useful features at each level
    # of aggregration.
    # This allows single quantities to be regressed from complex point counts
    # in a location and orientation invariant way.
    # One encoding layer is used to abstract away the input features.
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=1, k=8, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(DynamicReductionNetwork, self).__init__()

        self.datanorm = nn.Parameter(norm)
        
        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, 2*hidden_dim),            
            nn.Tanh(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.Tanh(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Tanh(),
        )        
        convnn = nn.Sequential(nn.Linear(start_width, middle_width),
                               nn.ReLU(),
                               nn.Linear(middle_width, hidden_dim),                                             
                               nn.ReLU()
                               )
        
        self.edgeconv = EdgeConv(nn=convnn, aggr=aggr)
        
        self.reducer = EdgePooling(hidden_dim)
        
        self.output = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(2*hidden_dim, 2*hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(2*hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim//2, output_dim))
        
        
    def forward(self, data, batch=None):        
        x_norm = self.datanorm * data.x
        H = self.inputnet(x_norm)
        n_batches = torch.unique(batch)
        out = None
        for i in n_batches:
            batch_mask = (batch == i)
            H_i = H[batch_mask]
            ibatch = torch.zeros(H_i.shape[0], dtype=torch.int64).to(H_i.device)
            while H_i.shape[0] > 1:
                edge_index = knn_graph(H_i, self.k, ibatch, loop=False, flow=self.edgeconv.flow)
                Hprime = self.edgeconv(H_i, edge_index)
                H_i, _, ibatch, _ = self.reducer(Hprime, edge_index, ibatch)
            if out is None:
                out = H_i
            else:
                out = torch.cat([out, H_i], dim=0)
        return self.output(out).squeeze(-1)
