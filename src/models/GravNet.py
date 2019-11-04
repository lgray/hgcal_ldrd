# torch geometric implementation of network introduced in arXiv:1902.07987

import torch
import torch.nn as nn
import numpy as np

# this network has two outputs with different numbers of features
# the two outputs are called "spatial" and "learned" based on usage in GravNet
class DoubleOutputNetwork(nn.Module):
    def __init__(self, spatial, learned):
        super(DoubleOutputNetwork, self).__init__()        
        self.spatial = spatial
        self.learned = learned

    def forward(self, x):
        return self.spatial(x), self.learned(x)

# performs message passing with input vector of weights that modify the features
from torch_geometric.nn import MessagePassing
class WeightedMessagePassing(MessagePassing):        
    def message(self, x_j, weights):
        return np.multiply(x_j, weights)

# potential functions for GravNet
# (essentially kernel functions of distance in latent space)
def gaussian_kernel(d_ij):
    return np.exp(-d_ij**2)
def exponential_kernel(d_ij):
    return np.exp(-np.abs(d_ij))
_allowed_kernels = {
    'gaussian': gaussian_kernel,
    'exponential': exponential_kernel,
}

# the full GravNet layer
# this is a sandwich of dense NN + neighbor assignment & message passing + dense NN
# the first dense NN should be a DoubleOutputNetwork or similar (or a sequence that ends in such)
from torch_geometric.nn import knn_graph
from torch import cdist, index_select
class GravNetLayer(nn.Module):
    def __init__(self,first_dense,n_neighbors,aggrs,second_dense,kernel='gaussian'):
        self.first_dense = first_dense
        self.n_neighbors = n_neighbors
        self.second_dense = second_dense

        if kernel not in _allowed_kernels:
            raise ValueError("Unrecognized kernel "+kernel+" (allowed values: "+', '.join(allowed_kernels)+")")
        self.kernel = _allowed_kernels[kernel]
        
        self.messengers = []
        for aggr in aggrs:
            self.messengers.append(WeightedMessagePassing(aggr=aggr,flow="target_to_source"))
        
    def forward(self, x, batch=None):
        # apply first dense NN to derive spatial and learned features
        spatial, learned = self.first_dense(x)
        
        # use spatial to generate edge index
        edge_index = knn_graph(spatial, self.n_neighbors, batch, loop=False)
        
        # make the vector of distance weights using kernel
        neighbors = index_select(spatial,0,edge_index[1])
        distances = cdist(spatial,neighbors,metric='euclidean')
        weights = self.kernel(distances)
        
        # use learned for message passing
        messages = [x]
        for messenger in self.messengers:
            messages.append(messenger(learned,weights))
            
        # concatenate features, keep input
        all_features = torch.cat(messages, dim=1)
        
        # apply second dense to get final set of features
        final = self.second_dense(all_features)
        
        return final

# a single block for the network
class GravBlock(nn.Module):
    def __init__(self, input_dim = 10, dense_dim = 64, spatial_dim = 4, learned_dim = 22, out_dim = 48, n_neighbors = 40, aggrs = ['add','mean','max']):
        self.layers = nn.Sequential(
            # first section: 3 dense layers w/ 64 nodes, tanh activation
            # add 1 to input_dim b/c concatenation of mean
            nn.Linear(in_features=input_dim+1,out_features=dense_dim),
            nn.Tanh(),
            nn.Linear(in_features=dense_dim,out_features=dense_dim),
            nn.Tanh(),
            nn.Linear(in_features=dense_dim,out_features=dense_dim),
            nn.Tanh(),
            # second section: GravNetLayer
            GravNetLayer(
                first_dense = DoubleOutputNetwork(
                    spatial = nn.Linear(in_features=dense_dim,out_features=spatial_dim),
                    learned = nn.Linear(in_features=dense_dim,out_features=learned_dim),
                ),
                n_neighbors = n_neighbors,
                aggrs = aggrs,
                second_dense = nn.Sequential(
                    nn.Linear(in_features=learned_dim,out_features=out_dim),
                    nn.Tanh(),
                ),
            ),
            nn.BatchNorm1d(out_dim)
        )
        # keep track of this in order to chain blocks together
        self.out_dim = out_dim
        
    def forward(self, x):
        # concatenate mean of features
        x = torch.cat([x, np.mean(x)], dim=1)
        
        # apply layers
        x = self.layers(x)
        return x
        
# the full network, with multiple blocks
class GravNet(nn.Module):
    # kwargs passed to GravBlocks
    def __init__(self, n_blocks = 4, final_dim = 128, n_clusters = 2, **kwargs):
        #batch norm for the input data
        self.inputnorm = nn.BatchNorm1d(kwargs['input_dim'])
        # first block just takes kwargs
        self.blocks = [GravBlock(**kwargs)]
        # subsequent blocks need to know the first block's output
        self.blocks.extend([GravBlock(input_dim = self.blocks[0].out_dim, **kwargs) for n in range(1, n_blocks)])                
        # final set of layers: dense ReLU, input from all blocks -> small dense ReLU -> small dense softmax        
        self.final = nn.Sequential(
            nn.Linear(in_features=n_blocks*self.blocks[0].out_dim,out_features=final_dim),
            nn.ReLU(),
            nn.Linear(in_features=final_dim,out_features=n_clusters+1),
            nn.ReLU(),
            nn.Linear(in_features=n_clusters+1,out_features=n_clusters),
            nn.Softmax(),
        )
    
    def forward(self, x):
        # apply batch norm to input (and then to all block outputs)
        x = self.inputnorm(x)
        # feed each block's output to the next
        all_output = [x]
        for block in self.blocks:
            block_output = block(all_output[-1])            
            all_output.append(block_output)
        # concatenate output from all blocks
        all_output = torch.cat(all_output[1:], dim=1)
        return self.final(all_output)

# define the loss function
# expects features in target: BxVx[energy, truth fraction, truth fraction, ...]
def energy_fraction_loss(pred, target, weight=None):
    energy = target[:,0]
    truth = target[:,1:]
    # used for per-sensor energy weighting w/in cluster
    total_energy_cluster = torch.sqrt(energy*truth)
    # get numer and denom terms for each shower
    numers = torch.sum(total_energy_cluster*(pred-truth)**2,axis=1)
    denoms = torch.sum(total_energy_cluster,axis=1)
    # sum of weighted differences
    loss = torch.sum(numers/denoms)
    return loss
    
# make a module
class EnergyFractionLoss(nn.Module):
    def forward(self, energy, pred, truth):
        return energy_fraction_loss(energy, pred, truth)