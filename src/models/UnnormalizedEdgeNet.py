import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import EdgeConv

class UnnormalizedEdgeNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=4, n_iters=1, aggr='add'):
        super(UnnormalizedEdgeNet, self).__init__()

        start_width = 2*(hidden_dim + input_dim)
        middle_width = (3*hidden_dim + 2*input_dim) // 2

        self.n_iters = n_iters
        
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(inplace=True)
        )

        self.edgenetwork = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                         nn.Linear(2 * (n_iters * hidden_dim + input_dim), output_dim)
        )

        for i in range(n_iters):
            setattr(self,'convnn%d' % i, nn.Sequential(nn.Linear(start_width, middle_width),
                                                       nn.Linear(middle_width, hidden_dim),
                                                       nn.ELU(inplace=True)))
            setattr(self, 'nodenetwork%d' % i, EdgeConv(nn=getattr(self, 'convnn%d' % i), aggr=aggr))
        
    def forward(self, data):
        row,col = data.edge_index
        H = self.inputnet(data.x)
        H_cat = data.x
        for i in range(self.n_iters):            
            H = getattr(self,'nodenetwork%d' % i)(torch.cat([H, data.x], dim=-1), data.edge_index)
            H_cat = torch.cat([H, H_cat], dim=-1)
        return self.edgenetwork(torch.cat([H_cat[row],H_cat[col]], dim=-1)).squeeze(-1)
