import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from datasets.hitgraphs import HitDataset

import tqdm
import argparse

sig_weight = 1.0
bkg_weight = 1.0
lr = 0.0001
n_epochs = 40
batch_size = 4


from training.pointnet import PointNetTrainer

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

import logging
    
def main(args):    

    path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'single_tau')
    print(path)
    full_dataset = HitDataset(path)
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

    trainer = PointNetTrainer(real_weight=sig_weight, fake_weight=bkg_weight, 
                         output_dir='/home/scratch/sitonga/hgcal_ldrd/checkpoints', device=device)

    trainer.logger.setLevel(logging.DEBUG)
    strmH = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    strmH.setFormatter(formatter)
    trainer.logger.addHandler(strmH)

    
    trainer.build_model(loss_func='binary_cross_entropy',
                        optimizer='Adam', learning_rate=0.0001, lr_scaling=None)
    
    print('made the PointNet trainer!')
    
    train_summary = trainer.train(train_loader, n_epochs, valid_data_loader=valid_loader)
    
    print(train_summary)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
                                                
