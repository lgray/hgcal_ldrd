import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import EdgeConv

class EdgeNet2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=1, n_iters=1, aggr='add'):
        super(EdgeNet2, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(hidden_dim + input_dim), (3*hidden_dim + 2*input_dim) // 2),
                               nn.ReLU(),
                               nn.Dropout(),
                               nn.Linear((3*hidden_dim + 2*input_dim) // 2, hidden_dim),
                               nn.ReLU()
        )
        
        self.n_iters = n_iters
        
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()
        )

        self.edgenetwork = nn.Sequential(nn.Linear(2*(n_iters*hidden_dim+input_dim), output_dim),
                                         nn.Sigmoid())
        
        self.nodenetwork = EdgeConv(nn=convnn,aggr=aggr)

    def forward(self, data):
        X = data.x
        H = self.inputnet(X)
        data.x = torch.cat([H, X], dim=-1)
        H_cat = X
        for i in range(self.n_iters):            
            H = self.nodenetwork(data.x, data.edge_index)
            H_cat = torch.cat([H, H_cat], dim=-1)
            data.x = torch.cat([H, X], dim=-1)
        row,col = data.edge_index
        return self.edgenetwork(torch.cat([H_cat[row], H_cat[col]], dim=-1)).squeeze(-1)
