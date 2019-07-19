"""
This module implements the PyTorch modules that define the
message-passing graph neural networks for hit or segment classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_geometric.nn import NNConv
from torch_scatter import scatter_add

class EdgeNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh):
        super(EdgeNetwork, self).__init__()
        self.edgec = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        self.sigm_out = nn.Sequential(nn.Sigmoid())
        self.network = NNConv(input_dim * 2, 1, self.edgec, aggr='add')

    def forward(self, data):
        row,col = data.edge_index
        #bi = data.x[col]
        #bo = data.x[row]
        # the original network constantly updates the edge network
        # the neural networks used are actually edge attributes
        #data.edge_attr = torch.cat([bo,bi],dim=-1)
        #print('EdgeNetworkG forward:',data.edge_attr.shape)
        B = torch.cat([data.x[col],data.x[row]],dim=-1).detach()
        return self.edgec(B) #self.network(data.x, data.edge_index, data.edge_attr)

class NodeNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh):
        super(NodeNetwork, self).__init__()
        self.nodec = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            hidden_activation())
        self.network = NNConv(input_dim * 3, output_dim, self.nodec, aggr='add')

    def forward(self, data):
        row,col = data.edge_index
        mi = data.x.new_zeros(data.x.shape)
        mo = data.x.new_zeros(data.x.shape)
        mi = scatter_add(data.edge_attr*data.x[row],col,dim=0,out=mi)
        mo = scatter_add(data.edge_attr*data.x[col],row,dim=0,out=mo)
        
        M = torch.cat([mi,mo,data.x],dim=-1)
        
        return self.nodec(M)

class GNNSegmentClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, n_iters=3, hidden_activation=nn.Tanh):
        super(GNNSegmentClassifier, self).__init__()
        self.n_iters = n_iters
        # Setup the input network
        self.input_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            hidden_activation())
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim + hidden_dim, hidden_dim,
                                        hidden_activation)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim + hidden_dim, hidden_dim,
                                        hidden_activation)

    def forward(self, data):
        """Apply forward pass of the model"""
        X = data.x
        # Apply input network to get hidden representation
        H = self.input_network(X)
        # Shortcut connect the inputs onto the hidden representation
        data.x = torch.cat([H, X], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_iters):
            # Apply edge network, update edge_attrs
            data.edge_attr = self.edge_network(data)
            # Apply node network
            H = self.node_network(data)
            # Shortcut connect the inputs onto the hidden representation
            data.x = torch.cat([H, X], dim=-1)
            # Apply final edge network
        return self.edge_network(data).squeeze(-1)
