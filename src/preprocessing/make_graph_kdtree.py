import numpy as np
import awkward
from datasets.graph import Graph
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, find

def make_graph_kdtree(coords,layers,sim_indices,r):
    #setup kd tree for fast processing
    the_tree = cKDTree(coords)
    
    #define the pre-processing (all layer-adjacent hits in ball R < r)
    #and build a sparse matrix representation, then blow it up 
    #to the full R_in / R_out definiton
    pairs = the_tree.query_pairs(r=r,output_type='ndarray')
    pairs = pairs[np.argsort(pairs[:,0])]
    first,second = pairs[:,0],pairs[:,1]  
    #selected index pair list that we label as connected
    #pairs_sel  = pairs[( (np.abs(layers[(second,)]-layers[(first,)]) <= 1)  )]
    neighbour_counts = np.unique(pairs[:,0], return_counts=True)[1]
    neighbour_counts = np.repeat(neighbour_counts, neighbour_counts)
    pairs_sel  = pairs[(np.abs(layers[(second,)]-layers[(first,)]) <= 1) | (neighbour_counts == 1)]
    #pairs_sel  = pairs
    data_sel = np.ones(pairs_sel.shape[0])
    
    #prepare the input and output matrices (already need to store sparse)
    r_shape = (coords.shape[0],pairs.shape[0])
    eye_edges = np.arange(pairs_sel.shape[0])
    
    R_i = csr_matrix((data_sel,(pairs_sel[:,1],eye_edges)),r_shape,dtype=np.uint8)
    R_o = csr_matrix((data_sel,(pairs_sel[:,0],eye_edges)),r_shape,dtype=np.uint8)
        
    #now make truth graph y (i.e. both hits are sim-matched)    
    y = (np.isin(pairs_sel,sim_indices).astype(np.int8).sum(axis=-1) == 2)
    
    return R_i,R_o,y
