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
import numpy as np
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss between connected nodes to enforce
    proximity for positive pairs and separation for negative pairs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, hard_labels, edges):
        """
        Args:
            embeddings (Tensor): Node embeddings [N, D].
            hard_labels (Tensor): Node labels [N].
            edges (Tensor): Edge indices [2, E].

        Returns:
            Tensor: Scalar contrastive loss value.
        """
        device = embeddings.device
        edges = edges.to(device)

        positive_classes = torch.tensor([1, 2, 3], device=device)
        negative_class = 0

        pos_mask = (hard_labels[edges[0]].unsqueeze(1) == positive_classes).any(dim=1) & \
                   (hard_labels[edges[1]].unsqueeze(1) == positive_classes).any(dim=1)

        neg_mask = ((hard_labels[edges[0]].unsqueeze(1) == positive_classes).any(dim=1) & (
                    hard_labels[edges[1]] == negative_class)) | \
                   ((hard_labels[edges[1]].unsqueeze(1) == positive_classes).any(dim=1) & (
                    hard_labels[edges[0]] == negative_class))

        pos_pairs = edges[:, pos_mask]
        neg_pairs = edges[:, neg_mask]

        if pos_pairs.size(1) == 0 or neg_pairs.size(1) == 0:
            return torch.tensor(0.0, device=device)

        perm = torch.randperm(neg_pairs.size(1), device=device)[:pos_pairs.size(1)]
        neg_pairs = neg_pairs[:, perm]

        pos_dists = torch.norm(embeddings[pos_pairs[0]] - embeddings[pos_pairs[1]], dim=1)
        neg_dists = torch.norm(embeddings[neg_pairs[0]] - embeddings[neg_pairs[1]], dim=1)

        pos_loss = torch.mean(pos_dists ** 2)
        neg_loss = torch.mean(torch.clamp(1.0 - neg_dists, min=0) ** 2)

        return 0.5 * (pos_loss + neg_loss)

class Loss(nn.Module):
    """
    Combined classification and contrastive loss for PU learning.
    Integrates supervised CE loss for each class and edge-level contrastive regularization.
    """
    def __init__(self, edges,num_class=4):
        """
        Args:
            edges (array-like or Tensor): Graph edges [2, E].
            num_class (int): Number of classes (default: 4).
        """
        super().__init__()
        self.num_class = num_class
        if isinstance(edges, (float, list, np.ndarray)):
            self.edges = torch.tensor(edges.T, dtype=torch.long)
        else:
            self.edges = edges.t()

        self.loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss()

    def forward(self, predictions, hard_labels, return_components=False):
        """
        Compute total loss combining CE and contrastive terms.

        Args:
            predictions (Tensor): Logits [N, C].
            hard_labels (Tensor): Node labels [N].
            return_components (bool): Whether to return sub-losses (unused).

        Returns:
            Tensor: Total scalar loss.
        """
        device = predictions.device
        edges = self.edges.to(device)

        all_nodes = torch.arange(predictions.size(0), device=device)
        pos_nodes1 = all_nodes[hard_labels == 1]
        pos_nodes2 = all_nodes[hard_labels == 2]
        pos_nodes3 = all_nodes[hard_labels == 3]
        unl_nodes = all_nodes[hard_labels == 0]
        #supervised loss
        r_hat_p1 = self.loss(predictions[pos_nodes1], hard_labels[pos_nodes1]).mean()
        r_hat_p2 = self.loss(predictions[pos_nodes2], hard_labels[pos_nodes2]).mean()
        r_hat_p3 = self.loss(predictions[pos_nodes3], hard_labels[pos_nodes3]).mean()
        r_hat_u = self.loss(predictions[unl_nodes], hard_labels[unl_nodes]).mean()
        contrastive_loss = self.contrastive_loss(predictions, hard_labels, edges)
        
        total_loss = r_hat_p1 + r_hat_u + r_hat_p2 + r_hat_p3 + contrastive_loss
        
        return total_loss

