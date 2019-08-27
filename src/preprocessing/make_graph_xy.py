import numpy as np
import awkward
from datasets.graph import Graph
from scipy.sparse import csr_matrix, find

from .algo_kdtree import algo_kdtree
from .algo_knn import algo_knn


def make_graph_xy(arrays, valid_sim_indices, ievt, mask, layered_norm, algo, **preprocessing_args):
   
    x = arrays[b'rechit_x'][ievt][mask]
    y = arrays[b'rechit_y'][ievt][mask]
    z = arrays[b'rechit_z'][ievt][mask]
    layer = arrays[b'rechit_layer'][ievt][mask]
    time = arrays[b'rechit_time'][ievt][mask]
    energy = arrays[b'rechit_energy'][ievt][mask]    
    feats = np.stack((x,y,layer,time,energy)).T


    all_sim_hits = np.unique(valid_sim_indices[ievt].flatten())
    sim_hits_mask = np.zeros(arrays[b'rechit_z'][ievt].size, dtype=np.bool)
    sim_hits_mask[all_sim_hits] = True
    simmatched = np.where(sim_hits_mask[mask])[0]
    
    
    if algo == 'kdtree':
        Ri, Ro, y_label = algo_kdtree(np.stack((x,y,layer)).T, layer, simmatched, **preprocessing_args)
    elif algo == 'knn':
        Ri, Ro, y_label = algo_knn(np.stack((x,y,layer)).T, layer, simmatched, **preprocessing_args)
    else:
        raise Exception('Edge construction algo %s unknown' % algo)
    return Graph(feats, Ri, Ro, y_label, simmatched)
