import numpy as np
import awkward
from datasets.graph import Graph
from scipy.sparse import csr_matrix, find


def make_graph_etaphi(arrays, valid_sim_indices, ievt, mask, layered_norm, algo, preprocessing_args):
   
    x = arrays[b'rechit_x'][ievt][mask]
    y = arrays[b'rechit_y'][ievt][mask]
    z = arrays[b'rechit_z'][ievt][mask]
    layer = arrays[b'rechit_layer'][ievt][mask]
    time = arrays[b'rechit_time'][ievt][mask]
    energy = arrays[b'rechit_energy'][ievt][mask]    
    feats = np.stack((x,y,layer,time,energy)).T

    eta = arrays[b'rechit_eta'][ievt][mask]
    phi = arrays[b'rechit_phi'][ievt][mask]
    layer_normed = layer / layered_norm
    
    all_sim_hits = np.unique(valid_sim_indices[ievt].flatten())
    sim_hits_mask = np.zeros(arrays[b'rechit_z'][ievt].size, dtype=np.bool)
    sim_hits_mask[all_sim_hits] = True
    simmatched = np.where(sim_hits_mask[mask])[0]
    
    #Ri, Ro, y_label = make_graph_kdtree(np.stack((eta, phi, layer_normed)).T, layer, simmatched, r=r)
    Ri, Ro, y_label = algo(np.stack((eta, phi, layer_normed)).T, layer, simmatched, **preprocessing_args)
    
    return Graph(feats, Ri, Ro, y_label, simmatched)
