"""
COMIX: Confidence-based logit-level graph mixup (2025)

Authors:
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Shihyung Park (psh0416@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""

import os
from collections import defaultdict
import numpy as np
import torch
from torch_geometric import datasets

def compute_heterophilic_ratio(edge_index, node_labels):
    """
    Compute the ratio of edges connecting nodes with different labels.

    Args:
        edge_index (Tensor): Edge index of shape [2, E].
        node_labels (Tensor): Node label tensor.

    Returns:
        float: Heterophilic ratio of the graph.
    """
    src, tgt = edge_index

    label_diff = (node_labels[src] != node_labels[tgt]).float()
    heterophilic_ratio = label_diff.mean().item()

    return heterophilic_ratio

def preprocess_edges(edges):
    """
    Remove duplicates and self-loops from edges.

    Args:
        edges (Tensor): Edge index tensor [2, E].

    Returns:
        np.ndarray: Cleaned symmetric edge list [2, E'].
    """
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if src != dst:
            m[src].add(dst)
            m[dst].add(src)

    edges = []
    for src in sorted(m):
        for dst in sorted(m[src]):
            edges.append((src, dst))
    return np.array(edges, dtype=np.int64).transpose()

def to_pu_setting(labels, pos_number):
    """
    Convert multi-class labels to PU format.

    Args:
        labels (np.ndarray): Original labels.
        pos_number (int): Number of positive classes.

    Returns:
        np.ndarray: PU-style labels (0 for unlabeled).
    """
    count = np.bincount(labels)
    pu_labels = np.zeros_like(labels)
    for i in range(pos_number):
        positive_nodes = labels == count.argmax()
        pu_labels[positive_nodes] = i+1
        count[count.argmax()]=-1

    return pu_labels

def split_nodes(labels, trn_ratio, vld_ratio, pos_number=1, seed=0):
    """
    Split nodes into train/validation/test sets.

    Args:
        labels (np.ndarray): Node labels.
        trn_ratio (float): Training ratio.
        vld_ratio (float): Validation ratio.
        pos_number (int): Number of positive classes.
        seed (int): Random seed.

    Returns:
        tuple: (train_nodes, test_nodes, val_nodes)
    """
    state = np.random.RandomState(seed)

    all_nodes = np.arange(labels.shape[0])
    pos_nodes = []
    for i in range(pos_number):
        pos_nodes.append(all_nodes[labels == i+1])
    neg_nodes = all_nodes[labels == 0]

    trn_nodes = []
    n_pos_nodes = len(all_nodes[labels != 0])
    n_trn_nodes = int(len(all_nodes) * trn_ratio)

    vld_nodes = []
    for pos_nodes_i in pos_nodes:
        trn_temp = state.choice(pos_nodes_i, size=int(pos_nodes_i.shape[0] * n_trn_nodes / n_pos_nodes), replace=False)
        pos_nodes_i = np.array(list(set(pos_nodes_i).difference(set(trn_temp))))
        vld_temp = state.choice(pos_nodes_i, size=int(pos_nodes_i.shape[0] * vld_ratio), replace=False)
        trn_nodes.append(trn_temp)
        vld_nodes.append(vld_temp)

    pos_nodes = np.concatenate(pos_nodes)
    trn_nodes = np.concatenate(trn_nodes)
    vld_nodes = np.concatenate(vld_nodes)
    test_nodes = np.array(list(set(pos_nodes).difference(set(trn_nodes)).difference(set(vld_nodes))))
    vld_neg_nodes = state.choice(neg_nodes, size=int(neg_nodes.shape[0] * vld_ratio), replace=False)
    test_neg_nodes = np.array(list(set(neg_nodes).difference(set(vld_neg_nodes))))
    test_nodes = np.concatenate([test_nodes, test_neg_nodes])
    vld_nodes = np.concatenate([vld_nodes, vld_neg_nodes])

    return trn_nodes, test_nodes, vld_nodes

def read_data(dataset, trn_ratio, vld_ratio=0.0, pos_class=1, verbose=False):
    """
    Load and preprocess dataset for PU learning.

    Args:
        dataset (str): Dataset name (Cora, CiteSeer, etc.).
        trn_ratio (float): Training ratio.
        vld_ratio (float): Validation ratio.
        pos_class (int): Number of positive classes.
        verbose (bool): If True, print dataset statistics.

    Returns:
        tuple: (features, labels, edges, train_nodes, test_nodes, val_nodes)
    """
    root = '../data'
    root_cached = os.path.join(root, 'cached', dataset)

    if not os.path.exists(root_cached):
        if dataset in ['CiteSeer']:
            data = datasets.Planetoid(root, dataset)
        elif dataset in ['Cora_ML', 'CiteSeer_full', 'DBLP']:
            dataset = dataset.replace('_full', '')
            data = datasets.CitationFull(root, dataset)
        elif dataset in ['Computers']:
            data = datasets.amazon.Amazon(root, dataset)

        node_x = data.data.x
        #if getattr(node_x, "layout", None) in (torch.sparse_csr, torch.sparse_coo) or torch.is_sparse(node_x):
        #    node_x = node_x.to_dense()
        if hasattr(node_x, "to_dense"): 
            node_x = node_x.to_dense()

        node_x[node_x.sum(dim=1) == 0] = 1
        node_x = node_x / node_x.sum(dim=1, keepdim=True)
        node_y = to_pu_setting(data.data.y, pos_number=pos_class)
        edges = preprocess_edges(data.data.edge_index)

        os.makedirs(root_cached, exist_ok=True)
        np.save(os.path.join(root_cached, 'x'), node_x)
        np.save(os.path.join(root_cached, 'y'), node_y)
        np.save(os.path.join(root_cached, 'edges'), edges)

    node_x = torch.from_numpy(np.array(np.load(os.path.join(root_cached, 'x.npy'), allow_pickle=True), dtype=np.float32))
    node_y = torch.from_numpy(np.array(np.load(os.path.join(root_cached, 'y.npy'), allow_pickle=True), dtype=np.int64))
    unique, counts = np.unique(node_y.numpy(), return_counts=True)
    print(dict(zip(unique, counts)))
    edges = torch.from_numpy(np.load(os.path.join(root_cached, 'edges.npy'), allow_pickle=True))
    trn_nodes, test_nodes, vld_nodes = split_nodes(node_y, trn_ratio, vld_ratio, pos_number=pos_class)

    if verbose:
        classes = ['N', 'P1', 'P2', 'P3']
        unique_values, counts = torch.unique(node_y, return_counts=True)

        print(f'----------------- Data statistics of {dataset} -----------------')
        print(f'Heterophilic ratio: {compute_heterophilic_ratio(edge_with_noise, node_y):.4f}')
        print('# of nodes:', node_x.size(0))
        print('# of features:', node_x.size(1))
        print(f'# of edges: {edges.size(1)} -> {edge_with_noise.size(1)}')
        print('# of instances for each class:', ', '.join(f"{cnt}({cls})" for cls, cnt in zip(classes, counts.tolist())))
        print(f'# of train / val / test instances: {len(trn_nodes)} / {len(vld_nodes)} / {len(test_nodes)}\n')

    return node_x, node_y, edges, trn_nodes, test_nodes, vld_nodes

