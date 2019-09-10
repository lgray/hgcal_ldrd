"""
Python module for holding our PyTorch models.
"""

from .EdgeNet import EdgeNet
from .gnn_geometric import GNNSegmentClassifier    
from .PointNet import PointNet

_models = {'EdgeNet': EdgeNet,
           'heptrkx_segment_classifier': GNNSegmentClassifier,
           'PointNet': PointNet}

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name in _models:
        return _models[name](**model_args)
    else:
        raise Exception('Model %s unknown' % name)
