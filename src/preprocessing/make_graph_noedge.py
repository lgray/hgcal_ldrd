import numpy as np
import awkward
from datasets.graph import Graph
from scipy.sparse import csr_matrix, find


def make_graph_noedge(arrays, valid_sim_indices, ievt, mask):
    x = arrays[b'rechit_x'][ievt][mask]
    y = arrays[b'rechit_y'][ievt][mask]
    z = arrays[b'rechit_z'][ievt][mask]
    layer = arrays[b'rechit_layer'][ievt][mask]
    time = arrays[b'rechit_time'][ievt][mask]
    energy = arrays[b'rechit_energy'][ievt][mask]    
    feats = np.stack((x,y,layer,time,energy)).T
    
    all_sim_hits = np.unique(valid_sim_indices[ievt].flatten())
    sim_hits_mask = np.zeros(arrays[b'rechit_z'][ievt].size)
    sim_hits_mask[all_sim_hits] = 1
    
    y_label = sim_hits_mask[mask]
    simmatched = np.where(sim_hits_mask[mask])[0]
    
    
    
    return Graph(feats, [], [], y_label, simmatched)
