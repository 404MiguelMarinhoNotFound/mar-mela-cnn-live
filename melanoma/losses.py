"""Loss functions for the imbalanced, F2-optimized objective (doctrine §4).

Even on a "roughly balanced" set, real inference skews melanoma-minority and we
optimize recall, so we default to **focal loss** (down-weights easy negatives,
gamma=2). A class-weighted BCE is provided as a fallback.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss for a single-logit binary classifier.

    Parameters
    ----------
    gamma : focusing parameter (doctrine suggests ~2).
    alpha : optional weight in [0,1] for the positive (melanoma) class. If None,
        the loss reduces to unweighted focal.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # prob of the true class
        focal = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * focal
        return focal.mean()


def build_loss(cfg, n_neg: int, n_pos: int) -> nn.Module:
    """Construct the configured loss, deriving weights from class counts."""
    if cfg.loss == "focal":
        alpha = cfg.focal_alpha
        if alpha is None and (n_pos + n_neg) > 0:
            # Up-weight the minority class proportionally to its rarity.
            alpha = n_neg / (n_pos + n_neg)
        return BinaryFocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
    if cfg.loss == "weighted_ce":
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"Unknown loss: {cfg.loss!r} (expected 'focal' or 'weighted_ce')")
