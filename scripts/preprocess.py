#!/usr/bin/env python
import uproot
import awkward
import numpy as np
import os
from scipy.sparse import csr_matrix, find
from scipy.spatial import cKDTree
from tqdm import tqdm

from datasets.graph import graph_to_sparse,sparse_to_graph, save_graph

from preprocessing import *


# for PointNet
preprocessing_algo = make_graph_noedge

# for EdgeNet
#preprocessing_algo = make_graph_etaphi
#grouping_algo = 'knn' #or 'kdtree'
#preprocessing_args= dict(k=4)
#preprocessing_args= dict(r = 0.07) #if algo == 'kdtree'
#layer_norm = 150 #only used for etaphi, no effect for other preprocessors

fname = '../data/ntup/partGun_PDGid15_x1000_Pt3.0To100.0_NTUP_1.root'

test = uproot.open(fname)['ana']['hgc']

#example of generating a binary ground-truth adjacency matrix 
#for both endcaps in all events for all clusters
#truth is now that hits in adjacent layers are connected 
#and so are hits in the same layer within delta-R < 2 
arrays = test.arrays([b'simcluster_hits_indices'])
rechit = test.arrays([b'rechit_x',b'rechit_y', b'rechit_z', b'rechit_eta', b'rechit_phi',
                      b'rechit_layer',b'rechit_time',b'rechit_energy'])
NEvents = rechit[b'rechit_z'].shape[0]
rechit[b'rechit_x'].content[rechit[b'rechit_z'].content < 0] *= -1
sim_indices = awkward.fromiter(arrays[b'simcluster_hits_indices'])
valid_sim_indices = sim_indices[sim_indices > -1]


for ievt in tqdm(range(NEvents),desc='events processed'):
    #make input graphs
    
    # for EdgeNet
    #pos_graph = preprocessing_algo(rechit, valid_sim_indices, ievt = ievt, mask = rechit[b'rechit_z'][ievt] > 0,
    #                               layered_norm = layer_norm, algo=grouping_algo, preprocessing_args=preprocessing_args)
    #neg_graph = preprocessing_algo(rechit, valid_sim_indices, ievt = ievt, mask = rechit[b'rechit_z'][ievt] < 0,
    #                               layered_norm = layer_norm, algo=grouping_algo, preprocessing_args=preprocessing_args)
    # for PointNet
    pos_graph = preprocessing_algo(rechit, valid_sim_indices, ievt = ievt, mask = rechit[b'rechit_z'][ievt] > 0)
    neg_graph = preprocessing_algo(rechit, valid_sim_indices, ievt = ievt, mask = rechit[b'rechit_z'][ievt] < 0)
    
    #write the graph and truth graph out
    outbase = fname.split('/')[-1].replace('.root','')
    outdir = "/".join(fname.split('/')[:-2]) + "/npz/" + outbase + "/raw"
    if not os.path.exists( outdir):
        os.makedirs(outdir)

    # for EdgeNet
    #save_graph(pos_graph, '%s/%s_hgcal_graph_pos_evt%d.npz'%(outdir,outbase,ievt))
    #save_graph(neg_graph, '%s/%s_hgcal_graph_neg_evt%d.npz'%(outdir,outbase,ievt))
    #saved as sparse
    
    # for PointNet
    save_graph(pos_graph, '%s/%s_hgcal_graph_pos_evt%d.npz'%(outdir,outbase,ievt), sparse=False)
    save_graph(neg_graph, '%s/%s_hgcal_graph_neg_evt%d.npz'%(outdir,outbase,ievt), sparse=False)