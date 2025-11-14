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
import torch.nn.functional as F
import torch.nn as nn

def logit_mixup_gumbel(
    labels: torch.Tensor,
    logits: torch.Tensor,
    mixup_class_ids: list,
    lam: float = 0.5,
    tau: float = 1.0,
    epoch: int = 0,
):
    """
    Perform class-wise logit-level mixup using Gumbel-Softmax.

    Args:
        labels (Tensor): Node class labels [N].
        logits (Tensor): Logits before softmax [N, C].
        mixup_class_ids (list): Target class IDs for mixup.
        lam (Tensor or float): Mixing coefficient(s).
        tau (float): Temperature for Gumbel-Softmax.

    Returns:
        Tensor: Mixed logits tensor.
    """
    device = logits.device
    mixed_logits = logits.clone()
    eps = 1e-12
    with torch.no_grad():
        confidence_scores = F.softmax(logits,dim=1)
    for cls_id in mixup_class_ids:
        class_mask = labels == cls_id
        class_indices = torch.where(class_mask)[0]

        if len(class_indices) < 2:
            continue
            
        candidate_indices = class_indices

        with torch.no_grad():
            if epoch == 0:
                uniform_logits = torch.zeros(len(candidate_indices), device=logits.device)
                score_matrix = uniform_logits.unsqueeze(0).expand(len(class_indices), -1)
                soft_partners = F.gumbel_softmax(score_matrix, tau=tau, hard=False, dim=-1)
            else:
                candidate_scores = confidence_scores[candidate_indices, cls_id]
                denom = torch.sum(candidate_scores, dim=0, keepdim=True) + eps
                candidate_scores = torch.log(candidate_scores + eps) - torch.log(denom)
                score_matrix = candidate_scores.unsqueeze(0).expand(len(candidate_indices), -1)
                soft_partners = F.gumbel_softmax(score_matrix, tau=tau, hard=False, dim=-1)

        candidate_logits = logits[candidate_indices]
        partners_logits = torch.matmul(soft_partners, candidate_logits)
        
        lam_tensor = lam[candidate_indices].unsqueeze(1)
        
        source_logits = logits[candidate_indices]
        mixed_logits[candidate_indices] = (
            lam_tensor * source_logits + (1.0 - lam_tensor) * partners_logits
        )
        
        
    return mixed_logits


