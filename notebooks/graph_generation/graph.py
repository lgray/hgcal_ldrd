import numpy as np
from collections import namedtuple

SparseGraph = namedtuple('SparseGraph',
        ['X', 'Ri_rows', 'Ri_cols', 'Ro_rows', 'Ro_cols', 'y','simmatched'])

def make_sparse_graph(X, Ri, Ro, y):
    Ri_rows, Ri_cols = Ri.nonzero()
    Ro_rows, Ro_cols = Ro.nonzero()
    return SparseGraph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y)

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