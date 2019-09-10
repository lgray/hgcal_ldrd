import os
import os.path as osp

import numpy as np
import torch
from datasets.hitgraphs import HitDataset

from torch_geometric.data import DataLoader
from models import get_model

import tqdm


model_fname = "../model_files/denoiser_pointnet_reference_model.pth"


path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'single_tau')
full_dataset = HitDataset(path)
fulllen = len(full_dataset)

d = full_dataset

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = get_model(name='PointNet').to(device)
model.load_state_dict(torch.load(model_fname))

evt_index = 1

test_dataset = torch.utils.data.Subset(full_dataset,[evt_index])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)




for i,data in enumerate(test_loader):
    assert (i == 0);
    x=(data.x.cpu().detach().numpy())
    pos =(data.pos.cpu().detach().numpy())
    y=(data.y.cpu().detach().numpy() >0.5)
    data = data.to(device)
    out = model(data).cpu().detach().numpy() > 0.5
    assert(y.shape == out.shape)
    


print(y)
print(out)

truepos = np.logical_and(y, out)
falseneg = np.logical_and(np.logical_not(out), y)
falsepos = np.logical_and(out, np.logical_not(y))
trueneg = np.logical_and(np.logical_not(y), np.logical_not(out))

assert(y.shape[0] == truepos.sum() + falseneg.sum() + falsepos.sum() + trueneg.sum())
print("truepos", truepos.sum(), "trueneg", trueneg.sum(), "falsepos", falsepos.sum(), "falseneg", falseneg.sum())

print("efficiency", truepos.sum()/y.sum())
print("purity", truepos.sum()/(truepos.sum()+falsepos.sum()))
print("true negative rate", (trueneg.sum())/(falsepos.sum() + trueneg.sum()))