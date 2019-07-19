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

from ..datasets.hitgraphs import HitGraphDataset

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

from models.gnn_geometric import GNNSegmentClassifier as Net
from EdgeNet import EdgeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def get_model_fname(model):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']
    
    model_fname = '%s_%d_%s_%s'%(model_name,model_params,
                                 model_cfghash, model_user)
    return model_fname

def train(model, optimizer, epoch, loader, total):
    model.train()
    model_fname = get_model_fname(model)

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        batch_target = data.y
        batch_weights_real = batch_target*sig_weight
        batch_weights_fake = (1 - batch_target)*bkg_weight
        batch_weights = batch_weights_real + batch_weights_fake
        optimizer.zero_grad()
        batch_output = model(data)
        batch_loss = F.binary_cross_entropy(batch_output, batch_target, weight=batch_weights)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("batch loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

    modpath = osp.join(os.getcwd(),model_fname+'.%d.pth'%epoch)
    torch.save(model.state_dict(),modpath)
    
    return sum_loss/(i+1)


@torch.no_grad()
def test(model,loader,total):
    model.eval()
    correct = 0

    sum_loss = 0
    sum_correct = 0
    sum_truepos = 0
    sum_trueneg = 0
    sum_falsepos = 0
    sum_falseneg = 0
    sum_true = 0
    sum_false = 0
    sum_total = 0
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        batch_target = data.y
        batch_output = model(data)
        batch_loss_item = F.binary_cross_entropy(batch_output, batch_target).item()
        t.set_description("batch loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        matches = ((batch_output > 0.5) == (batch_target > 0.5))
        true_pos = ((batch_output > 0.5) & (batch_target > 0.5))
        true_neg = ((batch_output < 0.5) & (batch_target < 0.5))
        false_pos = ((batch_output > 0.5) & (batch_target < 0.5))
        false_neg = ((batch_output < 0.5) & (batch_target > 0.5))
        sum_truepos += true_pos.sum().item()
        sum_trueneg += true_neg.sum().item()
        sum_falsepos += false_pos.sum().item()
        sum_falseneg += false_neg.sum().item()
        sum_correct += matches.sum().item()
        sum_true += batch_target.sum().item()
        sum_false += (batch_target < 0.5).sum().item()
        sum_total += matches.numel()

    print('scor', sum_correct,
          'stru', sum_true,
          'stp', sum_truepos,
          'stn', sum_trueneg,
          'sfp', sum_falsepos,
          'sfn', sum_falseneg,
          'stot', sum_total)
    return sum_loss/(i+1), sum_correct / sum_total, sum_truepos/sum_true, sum_falsepos / sum_false, sum_falseneg / sum_true, sum_truepos/(sum_truepos+sum_falsepos + 1e-6)

def main(args):    

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'training_data', 'single_mu')
    full_dataset = HitGraphDataset(path, directed=directed)
    fulllen = len(full_dataset)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    
    train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
    valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)

    d = full_dataset
    num_features = d.num_features
    num_classes = d[0].y.max().item() + 1 if d[0].y.dim() == 1 else d[0].y.size(1)

    model = EdgeNet(input_dim=num_features,hidden_dim=hidden_dim,n_iters=n_iters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model_fname = get_model_fname(model)
    
    print('Model: \n%s\nParameters: %i' %
          (model, sum(p.numel()
                      for p in model.parameters())))

    best_valid_loss = 99999
    print('Training with %s samples'%train_samples)
    print('Validating with %s samples'%valid_samples)

    for epoch in range(0, n_epochs):
        epoch_loss = train(model, optimizer, epoch, train_loader, train_samples)
        valid_loss, valid_acc, valid_eff, valid_fp, valid_fn, valid_pur = test(model, valid_loader, valid_samples)
        print('Epoch: {:02d}, Training Loss: {:.4f}'.format(epoch, epoch_loss))
        print('               Validation Loss: {:.4f}, Eff.: {:.4f}, FalsePos: {:.4f}, FalseNeg: {:.4f}, Purity: {:,.4f}'.format(valid_loss,
                                                                                                                                 valid_eff,
                                                                                                                                 valid_fp,
                                                                                                                                 valid_fn,
                                                                                                                                 valid_pur))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            modpath = osp.join(os.getcwd(),model_fname+'.best.pth')
            print('New best model saved to:',modpath)
            torch.save(model.state_dict(),modpath)
        
    modpath = osp.join(os.getcwd(),model_fname+'.final.pth')
    print('Final model saved to:',modpath)
    torch.save(model.state_dict(),modpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
                                                
