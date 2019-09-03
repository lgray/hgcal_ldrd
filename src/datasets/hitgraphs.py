"""
    PyTorch specification for the hit graph dataset.
    """

# System imports
import os
import glob
import os.path as osp

# External imports
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import (Data, Dataset)

# Local imports
from datasets.graph import load_graph

class HitGraphDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def __init__(self, root,
                 directed = True,
                 categorical = False,
                 transform = None,
                 pre_transform = None):
        self._directed = directed
        self._categorical = categorical
        super(HitGraphDataset, self).__init__(root, transform, pre_transform)
    
    def download(self):
        pass #download from xrootd or something later
    
    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = glob.glob(self.raw_dir+'/*.npz')
        return [f.split('/')[-1] for f in self.input_files]
    
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            proc_names = ['data_{}.pt'.format(idx) for idx in range(len(self.raw_file_names))]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data
    
    def process(self):
        #convert the npz into pytorch tensors and save them
        path = self.processed_dir
        for idx,raw_path in enumerate(tqdm(self.raw_paths)):

            g = load_graph(raw_path)
            
            Ro = g.Ro[0].T.astype(np.int64)
            Ri = g.Ri[0].T.astype(np.int64)
            
            i_out = Ro[Ro[:,1].argsort(kind='stable')][:,0]
            i_in  = Ri[Ri[:,1].argsort(kind='stable')][:,0]
                        
            x = g.X.astype(np.float32)
            edge_index = np.stack((i_out,i_in))
            y = g.y.astype(np.int64)
            if not self._categorical:
                y = g.y.astype(np.float32)
            #print('y type',y.dtype)
            outdata = Data(x=torch.from_numpy(x),
                           edge_index=torch.from_numpy(edge_index),
                           y=torch.from_numpy(y))
            
            if not self._directed and not outdata.is_undirected():
                rows,cols = outdata.edge_index
                temp = torch.stack((cols,rows))
                outdata.edge_index = torch.cat([outdata.edge_index,temp],dim=-1)
                outdata.y = torch.cat([outdata.y,outdata.y])
        
            torch.save(outdata, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
