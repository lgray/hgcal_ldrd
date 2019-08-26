import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)

from datasets.hitgraphs import HitGraphDataset

import tqdm
import argparse
directed = False
sig_weight = 1.0
bkg_weight = 0.15
batch_size = 32
n_epochs = 20
lr = 0.01
hidden_dim = 64
n_iters = 6

from training.gnn import GNNTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

import logging
    
def main(args):    

    path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'single_mu')
    print(path)
    full_dataset = HitGraphDataset(path, directed=directed)
    fulllen = len(full_dataset)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-tv_num,0,tv_num])
    print(fulllen, splits)
    
    train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
    valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)

    d = full_dataset
    num_features = d.num_features
    num_classes = d[0].y.max().item() + 1 if d[0].y.dim() == 1 else d[0].y.size(1)

    trainer = GNNTrainer(real_weight=sig_weight, fake_weight=bkg_weight, 
                         output_dir='/home/lagray/hgcal_ldrd/', device=device)

    trainer.logger.setLevel(logging.DEBUG)
    strmH = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    strmH.setFormatter(formatter)
    trainer.logger.addHandler(strmH)
        
    #example lr scheduling definition
    def lr_scaling(optimizer):
        from torch.optim.lr_scheduler import LambdaLR
        
        lr_type = 'linear'
        lr_warmup_epochs = 0
        
        warmup_factor = 0.
        if lr_scaling == 'linear':
            warmup_factor = 1.
        
        # LR ramp warmup schedule
        def lr_warmup(epoch, warmup_factor=warmup_factor,
                      warmup_epochs=lr_warmup_epochs):
            if epoch < warmup_epochs:
                return (1. - warmup_factor) * epoch / warmup_epochs + warmup_factor
            else:
                return 1.

        # give the LR schedule to the trainer
        return LambdaLR(optimizer, lr_warmup)
    
    trainer.build_model(name='EdgeNet', loss_func='binary_cross_entropy',
                        optimizer='Adam', learning_rate=0.01, lr_scaling=lr_scaling,
                        input_dim=num_features, hidden_dim=hidden_dim, n_iters=n_iters)
    
    print('made the hep.trkx trainer!')
    
    train_summary = trainer.train(train_loader, n_epochs, valid_data_loader=valid_loader)
    
    print(train_summary)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
                                                
