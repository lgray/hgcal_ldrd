"""
Python module for holding our PyTorch models.
"""

from .EdgeNet import EdgeNet
from .EdgeNet2 import EdgeNet2
from .EdgeNetWithCategories import EdgeNetWithCategories
from .gnn_geometric import GNNSegmentClassifier    
from .GravNet import GravNet, energy_fraction_loss

_models = {'EdgeNetWithCategories': EdgeNetWithCategories,
           'EdgeNet2': EdgeNet2,
           'EdgeNet': EdgeNet,
           'heptrkx_segment_classifier': GNNSegmentClassifier,
           'GravNet': GravNet,
}

_losses = {'energy_fraction_loss': energy_fraction_loss,
}

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name in _models:
        return _models[name](**model_args)
    else:
        raise Exception('Model %s unknown' % name)

# attach custom losses to functional
from torch import nn
def get_losses():
    for loss,fn in _losses.items():
        setattr(nn.functional, loss, fn)
