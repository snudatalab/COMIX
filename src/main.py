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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.sparse")

import argparse
import data
import os
import random
import time
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch_geometric.nn as gnn

import models


def to_device(gpu):
    """
    Select and return computation device (GPU or CPU).

    Args:
        gpu (int): GPU index number.

    Returns:
        torch.device: Target device for model and data.
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def parse_args():
    """
    Parse input arguments for experiment configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CiteSeer')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--trn-ratio', type=float, default=0.2)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--units', type=int, default=16)
    return parser.parse_args()


def main():
    """
    Main execution function for COMIX PU-GNN training.
    Loads dataset, initializes model, and runs training loop.
    """
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = to_device(args.gpu)

    features, labels, edges, trn_nodes, test_nodes, vld_nodes = data.read_data(
        args.data, args.trn_ratio, pos_class=3)
    num_nodes, num_features, num_class = features.size(0), features.size(1), 4

    trn_labels = torch.zeros(num_nodes, dtype=torch.long)
    trn_labels[trn_nodes] = labels[trn_nodes].long()

    model = models.GCN(num_features, num_class, args.units, args.layers).to(device)
    optimizer = optim.Adam(model.parameters())
    loss = models.Loss(edges.t().numpy(), num_class)

    features, edges, trn_labels = features.to(device), edges.to(device), trn_labels.to(device)

    start_time = time.time()
    trained_model = models.train_model(model, features, edges, labels,
                                       trn_nodes, test_nodes, loss, optimizer, trn_labels, args.epochs)
    end_time = time.time()
    print("Running time:", end_time - start_time)


if __name__ == '__main__':
    main()

