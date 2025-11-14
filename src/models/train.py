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
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .mixup import logit_mixup_gumbel
torch.autograd.set_detect_anomaly(True)

def evaluate_model(model, features, edges, labels, test_nodes):
    """
    Evaluate model on test nodes.

    Args:
        model (nn.Module): GCN model.
        features (Tensor): Node features.
        edges (Tensor): Edge indices.
        labels (Tensor): Ground-truth labels.
        test_nodes (Tensor): Node indices for test set.

    Returns:
        tuple: (macro_f1, accuracy, AUC)
    """
    model.eval()
    with torch.no_grad():
        out = model(features, edges).cpu()
        probs = F.softmax(out, dim=1)
        out_labels = torch.argmax(out, dim=1)

    test_f1 = f1_score(labels[test_nodes], out_labels[test_nodes], average='macro')
    test_acc = accuracy_score(labels[test_nodes], out_labels[test_nodes])
    test_auc = roc_auc_score(labels[test_nodes], probs[test_nodes], multi_class='ovr', average=
    'macro')
    return test_f1, test_acc, test_auc

def train_model(
    model, features, edges, labels, trn_nodes, test_nodes, 
    loss_func, optimizer, trn_labels, epochs
    ):
    """
    Train COSMOS GCN model with adaptive Gumbel-based mixup.

    Args:
        model (nn.Module): GCN model.
        features (Tensor): Node features.
        edges (Tensor): Edge indices.
        labels (Tensor): Ground-truth labels.
        trn_nodes (Tensor): Training node indices.
        test_nodes (Tensor): Test node indices.
        loss_func (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        trn_labels (Tensor): Training labels.
        epochs (int): Number of epochs.

    Returns:
        nn.Module: Trained model.
    """
    num_classes = trn_labels.max()+1
    tau_max = 10.0
    tau_min = 3.0
    lam_min= 0.1
    print(f"Gumbel-Softmax training activated!")

    for epoch in range(epochs + 1):
        current_tau = tau_min + (tau_max - tau_min) / (
            1 + math.exp(0.05 * (epoch - epochs / 2))
        )
        model.train()

        initial_logits = model(features, edges)
        with torch.no_grad():
            initial_probs = F.softmax(initial_logits, dim=1)

            pred_confidences, pred_labels = initial_probs.max(dim=1)

            confidences = initial_probs.gather(1, trn_labels.unsqueeze(1)).squeeze()
            new_trn_labels = trn_labels.clone()
            adaptive_lam = lam_min + (1. - 2 * lam_min) * confidences

        positive_classes = [1,2,3]
        
        mixed_logits = logit_mixup_gumbel(
            labels=new_trn_labels,
            logits=initial_logits,
            mixup_class_ids=[0, 1, 2, 3],
            lam=adaptive_lam,
            tau=current_tau
        )

        loss = loss_func(mixed_logits, new_trn_labels, return_components=True)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        test_f1, test_acc, test_auc = evaluate_model(model, features, edges, labels, test_nodes)
        print(
                f"Epoch: {epoch:03d}, Total Loss: {loss.item():.4f}, "
                f"Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}"
            )
    return model


