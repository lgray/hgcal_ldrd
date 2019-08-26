import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.hitgraphs import HitGraphDatasetG
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
import tqdm
import argparse

from models.gnn_geometric import GNNSegmentClassifierG as Net
from EdgeNet import EdgeNet

from heptrx_nnconv import test

from datasets.graph import draw_sample

import awkward
import matplotlib.pyplot as plt
import scipy.stats as stats

batch_size = 32
hidden_dim = 64
n_iters = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def main(args):
    
    directed = False
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'training_data', 'single_mu')
    full_dataset = HitGraphDatasetG(path, directed=directed)
    fulllen = len(full_dataset)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    
    test_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[0],stop=splits[1]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_samples = len(test_dataset)

    d = full_dataset
    num_features = d.num_features
    num_classes = d[0].y.max().item() + 1 if d[0].y.dim() == 1 else d[0].y.size(1)
    
    model = EdgeNet(input_dim=num_features,hidden_dim=hidden_dim,n_iters=n_iters).to(device)
    model_fname = args.model
    print('Model: \n%s\nParameters: %i' %
          (model, sum(p.numel()
                      for p in model.parameters())))
    print('Testing with %s samples'%test_samples)
    
    model.load_state_dict(torch.load(model_fname))

    test_loss, test_acc, test_eff, test_fp, test_fn, test_pur = test(model, test_loader, test_samples)
    print('Testing: Loss: {:.4f}, Eff.: {:.4f}, FalsePos: {:.4f}, FalseNeg: {:.4f}, Purity: {:,.4f}'.format(test_loss, test_eff,
                                                                                                            test_fp, test_fn, test_pur))


    # plotting:
    figs = []
    t = tqdm.tqdm(enumerate(test_loader),total=test_samples/batch_size)
    out = []
    y = []
    x = []
    edge_index = []
    simmatched = []
    for i,data in t:
        x.append(data.x.cpu().detach().numpy())
        y.append(data.y.cpu().detach().numpy())
        edge_index.append(data.edge_index.cpu().detach().numpy())
        simmatched.append(data.simmatched.cpu().detach().numpy())
        data = data.to(device)

    out = awkward.fromiter(out)
    x = awkward.fromiter(x)
    y = awkward.fromiter(y)
    simmatched = awkward.fromiter(simmatched)
    edge_index = awkward.fromiter(edge_index)

    predicted_edge = (out > 0.5)
    truth_edge = (y > 0.5)
    node_energy = x[:,:,4]
    node_layer = x[:,:,2]

    predicted_connected_node_indices = awkward.JaggedArray.concatenate([edge_index[:,0][predicted_edge], edge_index[:,1][predicted_edge]], axis=1)
    predicted_connected_node_indices = awkward.fromiter(map(np.unique, predicted_connected_node_indices))
    predicted_energy_sum = node_energy[predicted_connected_node_indices].sum()
    truth_connected_node_indices = awkward.JaggedArray.concatenate([edge_index[:,0][truth_edge],edge_index[:,1][truth_edge]], axis=1)
    truth_connected_node_indices = awkward.fromiter(map(np.unique, truth_connected_node_indices))
    truth_energy_sum = node_energy[truth_connected_node_indices].sum()
    
    nonzeromask = (truth_energy_sum !=0.0)
    energy_captured_ratio = predicted_energy_sum[nonzeromask]/truth_energy_sum[nonzeromask]
    
    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist(energy_captured_ratio, bins=100)
    plt.title("Ratio of energy sum for predicted hits/truth (preprocessed) hits")
    plt.ylabel("events (pos+neg)")
    plt.xlabel("Ratio")
    cut = energy_captured_ratio[(energy_captured_ratio>0.975) & (energy_captured_ratio < 1.025)] #NB: restricted fit
    (mu, sigma) = stats.norm.fit(cut)
    c_paras = stats.crystalball.fit(cut)
    lnspc = np.linspace(bins[0], bins[-1], len(bins))
    pdf_g = stats.norm.pdf(lnspc, mu, sigma)
    pdf_c = stats.crystalball.pdf(lnspc, *c_paras)
    pdf_g = pdf_g / pdf_g.sum() * len(energy_captured_ratio)
    pdf_c = pdf_c / pdf_c.sum() * len(energy_captured_ratio)
    plt.plot(lnspc, pdf_g, label="Norm, restricted fit")
    plt.plot(lnspc, pdf_c, label="Crystalball, restricted fit")
    plt.legend(loc='upper left')
    figs.append(fig)


    # visualisation
    idxs = [0]
    for idx in idxs:
        fig = draw_sample(x[idx].regular(), edge_index[idx].regular()[0], edge_index[idx].regular()[1], y[idx], out[idx], sim_list=simmatched[idx])
        figs.append(fig)
    
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages("test_plots.pdf")
    for fig in figs: 
        pdf.savefig(fig)
    pdf.close()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("model", help="model PyTorch state dict file [*.pth]")
    args = parser.parse_args()
    main(args)
