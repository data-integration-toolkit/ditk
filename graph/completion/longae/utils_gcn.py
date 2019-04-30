"""
The MIT License
Copyright (c) 2017 Thomas Kipf

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

Modified from https://github.com/tkipf/gae to work with citation network data.
"""

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from collections import defaultdict as dd

from graph.completion.longae.hparams import hparams

np.random.seed(1982)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_graph(file):
    graph = dd(list)
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip().split(" ")
            graph[line[0]].append(line[1])

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()
    return adj
    
def output_graph(graph):
    output = []
    for k, v in nx.to_dict_of_lists(nx.to_networkx_graph(graph)).items():
        for i in v:
            output.append((k, i))
    return output

def load_citation_data_from_file(files):
    """Load citation data."""
    with open(files[0], 'rb') as f:
        features = np.loadtxt(f)
    
    with open(files[1], 'rb') as f:
        labels = np.loadtxt(f)
    
    graph = dd(list)
    with open(files[2], 'r') as f:
        for line in f:
            line = line.rstrip().split(" ")
            graph[line[0]].append(line[1])

    if hparams.index_file == "":
        l = len(features) // 5
        test_idx_range = np.arange(l)
        test_idx_reorder = np.random.permutation(test_idx_range)
    else:
        test_idx_reorder = parse_index_file(hparams.index_file)
        test_idx_range = np.sort(test_idx_reorder)

    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).tolil()

    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(hparams.training_size)
    idx_val = range(hparams.training_size, hparams.training_size + hparams.dev_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def split_citation_data(adj):
    """
    Function to build test set with 10% positive links and
    the same number of randomly sampled negative links.
    NOTE: Splits are randomized and results might slightly deviate
    from reported numbers in the paper.
    """

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))
    if num_test == 0:
        num_test = 1
    if num_val == 0:
        num_val = 1

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    data = np.ones(train_edges.shape[0])

    # NOTE: the edge list only contains single direction of edge!
    return np.concatenate([test_edges, np.asarray(test_edges_false)], axis=0)

