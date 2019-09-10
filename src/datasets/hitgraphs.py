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

import xgboost as xgb

# Local imports
from datasets.graph import load_graph

class HitGraphDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def __init__(self, root,
                 directed = True,
                 transform = None,
                 pre_transform = None):
        self._directed = directed
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
            
            i_out = Ro[Ro[:,1].argsort()][:,0]
            i_in  = Ri[Ri[:,1].argsort()][:,0]
            
            x = g.X.astype(np.float32)
            edge_index = np.stack((i_out,i_in))
            y = g.y.astype(np.float32)
            outdata = Data(x=torch.from_numpy(x),
                           edge_index=torch.from_numpy(edge_index),
                           y=torch.from_numpy(y))
            if not self._directed and not outdata.is_undirected():
                rows,cols = outdata.edge_index
                temp = torch.stack((cols,rows))
                outdata.edge_index = torch.cat([outdata.edge_index,temp],dim=-1)
                outdata.y = torch.cat([outdata.y,outdata.y])
        
            torch.save(outdata, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))

class HitDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def __init__(self, root,
                 transform = None,
                 pre_transform = None):
        super(HitDataset, self).__init__(root, transform, pre_transform)

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
        
        bst = xgb.Booster()
        bst.load_model('../model_files/xgboost_denoiser_reference.model')
        bst.set_param({'tree_method':'hist'})  #force inference on CPU
        
        for idx,raw_path in tqdm(enumerate(self.raw_paths), desc='event processed'):
            g = load_graph(raw_path, sparse=False)

            x = g.X.astype(np.float32)[:,3:]
            pos = g.X.astype(np.float32)[:,:3]
            y = g.y.astype(np.float32)
            if (y.sum() != 0):
                weight_in_event = (y * (1 / y.sum()) + (1-y) * (1 / (y.shape[0] - y.sum()))).astype(np.float32)
            else:
                weight_in_event = np.ones(y.shape[0]).astype(np.float32)
            dmatrix = xgb.DMatrix(g.X.astype(np.float32))
            pred = bst.predict(dmatrix).astype(np.float32)
            del(dmatrix)
            
            
            outdata = Data(x=torch.from_numpy(x),
                            pos=torch.from_numpy(pos),
                            y=torch.from_numpy(y),
                            weight_in_event=torch.from_numpy(weight_in_event),
                            xgboost_score=torch.from_numpy(pred))
            
            
            torch.save(outdata, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))