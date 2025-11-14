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
import torch
from torch import nn
from torch_geometric import nn as gnn

class GCN(nn.Module):
    """
    Basic Graph Convolutional Network (GCN) model.

    Args:
        num_features (int): Input feature dimension.
        num_class (int): Number of output classes.
        num_hidden (int): Hidden dimension size.
        num_layers (int): Number of GCN layers.
    """
    def __init__(self, num_features, num_class, num_hidden=16, num_layers=2):
        super().__init__()
        layers = []
        norms = []
        for i in range(num_layers):
            num_inputs = num_features if i == 0 else num_hidden
            layers.append(gnn.GCNConv(num_inputs, num_hidden, cached=True))
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(num_hidden, num_class)
        self.num_class = num_class

    def forward(self, x, edge_index, edge_weights=None):
        """
        Forward pass through GCN.

        Args:
            x (Tensor): Node features [N, F].
            edge_index (Tensor): Edge indices [2, E].
            edge_weights (Tensor, optional): Edge weights.

        Returns:
            Tensor: Logits [N, C].
        """
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out, edge_index, edge_weights)
            out = torch.relu(out)
        return self.linear(out)

