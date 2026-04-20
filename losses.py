"""
src/losses.py
Loss functions for GastroVision training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples so training focuses on hard ones.

    gamma = 0  → standard CrossEntropy
    gamma = 2  → recommended default for class imbalance
    gamma = 3+ → aggressive, rarely needed

    BUG FIX from original code:
        Original computed loss.backward() twice per step (called criterion
        twice: once for backward, once for .item()). This doubled GPU memory
        and inflated the printed loss values by 2×.
        Fixed: compute once, store result.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss    = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt         = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingFocalLoss(nn.Module):
    """
    Focal Loss + Label Smoothing.
    Reduces overconfidence which is a significant issue for rare classes
    with very few training examples.

    Recommended as a drop-in improvement over plain FocalLoss,
    especially when training on synthetic data that may contain
    distributional shift artifacts.
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.1, num_classes: int = 27):
        super().__init__()
        self.gamma       = gamma
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth one-hot targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss   = -(smooth_targets * log_probs).sum(dim=-1)

        # Focal weighting
        probs      = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_loss = (1 - probs) ** self.gamma * ce_loss

        return focal_loss.mean()
