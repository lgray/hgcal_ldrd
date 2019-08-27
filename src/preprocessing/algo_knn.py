import numpy as np
import awkward
from datasets.graph import Graph
from scipy.sparse import csr_matrix, find
from sklearn.neighbors import NearestNeighbors


def algo_knn(coords, layers, sim_indices, k):
    
    nbrs = NearestNeighbors(algorithm='kd_tree').fit(coords)
    nbrs_sm = nbrs.kneighbors_graph(coords, k)
    nbrs_sm.setdiag(0) #remove self-loop edges
    nbrs_sm.eliminate_zeros() 
    nbrs_sm = nbrs_sm + nbrs_sm.T
    pairs_sel = np.array(nbrs_sm.nonzero()).T
    first,second = pairs_sel[:,0],pairs_sel[:,1]  
    #selected index pair list that we label as connected
    data_sel = np.ones(pairs_sel.shape[0])
    
    #prepare the input and output matrices (already need to store sparse)
    r_shape = (coords.shape[0],pairs_sel.shape[0])
    eye_edges = np.arange(pairs_sel.shape[0])
    
    R_i = csr_matrix((data_sel,(pairs_sel[:,1],eye_edges)),r_shape,dtype=np.uint8)
    R_o = csr_matrix((data_sel,(pairs_sel[:,0],eye_edges)),r_shape,dtype=np.uint8)
        
    #now make truth graph y (i.e. both hits are sim-matched)    
    y = (np.isin(pairs_sel,sim_indices).astype(np.int8).sum(axis=-1) == 2)
    return R_i,R_o,y    
