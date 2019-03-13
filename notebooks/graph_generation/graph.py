import numpy as np
from collections import namedtuple
from scipy.sparse import csr_matrix

Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','simmatched'])

SparseGraph = namedtuple('SparseGraph',
        ['X', 'Ri_rows', 'Ri_cols', 'Ro_rows', 'Ro_cols', 'y', 'simmatched'])

def make_sparse_graph(X, Ri, Ro, y,simmatched=None):
    Ri_rows, Ri_cols = Ri.nonzero()
    Ro_rows, Ro_cols = Ro.nonzero()
    return SparseGraph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, simmatched)

def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph._asdict())
    #np.savez(filename, X=graph.X, Ri=graph.Ri, Ro=graph.Ro, y=graph.y)

def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)

def load_graph(filename, graph_type=SparseGraph):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return graph_type(**dict(f.items()))

def load_graphs(filenames, graph_type=SparseGraph):
    return [load_graph(f, graph_type) for f in filenames]

def graph_from_sparse(sparse_graph, dtype=np.uint8):
    n_nodes = sparse_graph.X.shape[0]
    n_edges = sparse_graph.Ri_rows.shape[0]
    mat_shape = (n_nodes,n_edges)
    data = np.ones(n_edges)
    Ri = csr_matrix((data,(sparse_graph.Ri_rows,sparse_graph.Ri_cols)),mat_shape,dtype=dtype)
    Ro = csr_matrix((data,(sparse_graph.Ro_rows,sparse_graph.Ro_cols)),mat_shape,dtype=dtype)
    return Graph(sparse_graph.X, Ri, Ro, sparse_graph.y, sparse_graph.simmatched)