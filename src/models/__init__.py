"""
Python module for holding our PyTorch models.
"""

from .EdgeNet import EdgeNet
from .EdgeNet2 import EdgeNet2
from .UnnormalizedEdgeNet import UnnormalizedEdgeNet
from .gnn_geometric import GNNSegmentClassifier    

_models = {'UnnormalizedEdgeNet': UnnormalizedEdgeNet,
           'EdgeNet2': EdgeNet2,
           'EdgeNet': EdgeNet,
           'heptrkx_segment_classifier': GNNSegmentClassifier}

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name in _models:
        return _models[name](**model_args)
    else:
        raise Exception('Model %s unknown' % name)
