# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-05-24
File: data_convert.py
"""
import os
import numpy as np
import scipy.sparse as sp
import argparse
import networkx as nx

from data_processor import DataProcess
from LINKX.dataset import load_nc_dataset, rand_train_test_idx

DATAPATH = 'data'

def get_idx_split(label, split_type='random', train_prop=.5, valid_prop=.25):
    ignore_negative = True
    train_idx, valid_idx, test_idx = rand_train_test_idx(
        label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
    split_idx = {'train': train_idx,
                 'valid': valid_idx,
                 'test': test_idx}
    return split_idx

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str)
args = parser.parse_args()

namestr = args.data
dataset = load_nc_dataset(namestr)
graph, labels = dataset.graph, dataset.label
print("Name of dataset: ", namestr)
print("Number of raw nodes: ", dataset.graph['num_nodes'])

adjtxt = dataset.graph['edge_index'].numpy().T
G = nx.from_edgelist(adjtxt, nx.Graph)
G.remove_edges_from(nx.selfloop_edges(G))

gcc = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
g0 = G.subgraph(gcc[0])
g0 = g0.to_directed()
nodelst = np.array(sorted(g0.nodes()))
g1 = nx.convert_node_labels_to_integers(g0, ordering='sorted')
adjtxt = np.array(g1.edges())

print("Nodes", len(nodelst), np.max(nodelst), len(g1.nodes()), np.max(g1.nodes()))
print("Edges", g1.number_of_edges(), len(g1.edges()), adjtxt.shape)

dp = DataProcess(namestr, path=DATAPATH)
os.makedirs(os.path.join(DATAPATH, namestr), exist_ok=False)
dp._n = len(nodelst)
dp._m = adjtxt.shape[0]
ones = np.ones((dp.m), dtype=np.int8)
dp.adj_matrix = sp.coo_matrix(
    (ones, (adjtxt[:, 0], adjtxt[:, 1])),
    shape=(dp.n, dp.n))
print("Shape of adj_matrix: ", dp.adj_matrix.shape)
# print("Number of raw edges: ", dp.adj_matrix.nnz)
dp.adj_matrix = dp.adj_matrix.tocsr()
dp.adj_matrix.setdiag(0)
dp.adj_matrix[dp.adj_matrix > 1] = 1
dp.adj_matrix.eliminate_zeros()
dp._m = dp.adj_matrix.nnz
print("Number of edges: ", dp.adj_matrix.nnz, dp.m)
assert dp.adj_matrix.diagonal().sum() == 0, "adj_matrix error"
assert (dp.adj_matrix - dp.adj_matrix.T).sum() == 0, "adj_matrix error"
assert (labels.dim()==2 and labels.shape[1]==1) or labels.dim()==1, "label shape error"

labels = labels[nodelst]
split_idx = get_idx_split(labels)
idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]

dp.attr_matrix = dataset.graph['node_feat'].numpy()[nodelst]
dp.labels = labels.numpy().flatten()
dp.idx_train = idx_train.numpy()
dp.idx_train.sort()
dp.idx_val = idx_val.numpy()
dp.idx_val.sort()
dp.idx_test = idx_test.numpy()
dp.idx_test.sort()
print(dp.labels.shape, dp.attr_matrix.shape, np.min(dp.labels), np.max(dp.labels))
print(dp.idx_train.shape, dp.idx_val.shape, dp.idx_test.shape)
print(dp)

dp.calculate(['deg'])
dp.output(['adjnpz', 'attr_matrix', 'labels', 'adjl', 'attribute'])
