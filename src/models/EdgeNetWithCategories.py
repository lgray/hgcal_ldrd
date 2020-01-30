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

from torch_geometric.nn import EdgeConv

class EdgeNetWithCategories(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=4, n_iters=1, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(EdgeNetWithCategories, self).__init__()

        self.datanorm = nn.Parameter(norm)
        
        start_width = 2 * (hidden_dim + input_dim)
        middle_width = (3 * hidden_dim + 2*input_dim) // 2
        
        self.n_iters = n_iters
                
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, 2*hidden_dim),            
            nn.Tanh(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.Tanh(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.edgenetwork = nn.Sequential(nn.Linear(2*n_iters*hidden_dim, 2*hidden_dim),
                                         nn.ELU(),  
                                         nn.Linear(2*hidden_dim, 2*hidden_dim),                                         
                                         nn.ELU(),
                                         nn.Linear(2*hidden_dim, output_dim),
                                         nn.LogSoftmax(dim=-1),
        )
        
        for i in range(n_iters):
            convnn = nn.Sequential(nn.Linear(start_width, middle_width),
                                   nn.ELU(),
                                   #nn.Dropout(p=0.5, inplace=False),
                                   nn.Linear(middle_width, hidden_dim),                                             
                                   nn.ELU()                                   
                                  )
            setattr(self, 'nodenetwork%d' % i, EdgeConv(nn=convnn, aggr=aggr))
        
    def forward(self, data):        
        row,col = data.edge_index
        x_norm = self.datanorm * data.x
        H = self.inputnet(x_norm)
        H = getattr(self,'nodenetwork0')(torch.cat([H, x_norm], dim=-1), data.edge_index)
        H_cat = H
        for i in range(1,self.n_iters):            
            H = getattr(self,'nodenetwork%d' % i)(torch.cat([H, x_norm], dim=-1), data.edge_index)
            H_cat = torch.cat([H, H_cat], dim=-1)                    
        return self.edgenetwork(torch.cat([H_cat[row],H_cat[col]],dim=-1)).squeeze(-1)
