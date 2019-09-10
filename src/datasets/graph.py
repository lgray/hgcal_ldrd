"""
This module contains code for interacting with hit graphs.
A Graph is a namedtuple of matrices X, Ri, Ro, y.
"""

from collections import namedtuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

# A Graph is a namedtuple of matrices (X, Ri, Ro, y)

Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y', 'simmatched'])



def graph_to_sparse(graph):
    Ri_rows, Ri_cols = graph.Ri.nonzero()
    Ro_rows, Ro_cols = graph.Ro.nonzero()
    return dict(X=graph.X, y=graph.y,
                Ri_rows=Ri_rows, Ri_cols=Ri_cols,
                Ro_rows=Ro_rows, Ro_cols=Ro_cols,
                simmatched=graph.simmatched
               )

def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, simmatched, dtype=np.float32):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    spRi_idxs = np.stack([Ri_rows.astype(np.int64), Ri_cols.astype(np.int64)])
    # Ri_rows and Ri_cols have the same shape
    spRi_vals = np.ones((Ri_rows.shape[0],), dtype=dtype)
    spRi = (spRi_idxs,spRi_vals,n_nodes,n_edges)

    spRo_idxs = np.stack([Ro_rows.astype(np.int64), Ro_cols.astype(np.int64)])
    # Ro_rows and Ro_cols have the same shape
    spRo_vals = np.ones((Ro_rows.shape[0],), dtype=dtype)
    spRo = (spRo_idxs,spRo_vals,n_nodes,n_edges)

    if y.dtype != np.uint8:
        y = y.astype(np.uint8)

    return Graph(X, spRi, spRo, y, simmatched)


def save_graph(graph, filename, sparse):
    """Write a single graph to an NPZ file archive"""
    if sparse:
        np.savez(filename, **graph_to_sparse(graph))
    else:
        np.savez(filename, **graph._asdict())


def save_graphs(graphs, filenames, sparse =True):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename, sparse)


def load_graph(filename, sparse =True):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        if sparse:
            return sparse_to_graph(**dict(f.items()))
        else:
            return Graph(**dict(f.items()))


def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f, graph_type) for f in filenames]


#thanks Steve :-)
def draw_sample(X, Ri, Ro, y, out,
                cmap='bwr_r', 
                skip_false_edges=True,
                alpha_labels=False, 
                sim_list=None): 
    # Select the i/o node features for each segment    
    feats_o = X[Ro]
    feats_i = X[Ri]    
    # Prepare the figure
    fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(20,12))
    cmap = plt.get_cmap(cmap)
    
    
    #if sim_list is None:    
        # Draw the hits (layer, x, y)
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    #else:        
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    #    ax0.scatter(X[sim_list,0], X[sim_list,2], c='b')
    #    ax1.scatter(X[sim_list,1], X[sim_list,2], c='b')
    
    # Draw the segments
        
     
    if out is not None:
        t = tqdm.tqdm(range(out.shape[0]))
        for j in t:       
            if y[j] and out[j]>0.5: 
                seg_args = dict(c='purple', alpha=0.2)
            elif y[j] and out[j]<0.5: 
                seg_args = dict(c='blue', alpha=0.2)
            elif out[j]>0.5:
                seg_args = dict(c='red', alpha=0.2)
            else:
                    continue #false edge

            ax0.plot([feats_o[j,0], feats_i[j,0]],
                     [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            ax1.plot([feats_o[j,1], feats_i[j,1]],
                     [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    else:
        t = tqdm.tqdm(range(y.shape[0]))
        for j in t:
            if y[j]:
                seg_args = dict(c='b', alpha=0.4)
            elif not skip_false_edges:
                seg_args = dict(c='black', alpha=0.4)
            else: continue
                
            ax0.plot([feats_o[j,0], feats_i[j,0]],
                     [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            ax1.plot([feats_o[j,1], feats_i[j,1]],
                     [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
        
    # Adjust axes
    ax0.set_xlabel('$x$ [cm]')
    ax1.set_xlabel('$y$ [cm]')
    ax0.set_ylabel('$layer$ [arb]')
    ax1.set_ylabel('$layer$ [arb]')
    plt.tight_layout()
    return fig;