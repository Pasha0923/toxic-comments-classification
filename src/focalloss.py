import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Gives more attention to difficult
    and rare examples during training.
    """

    def __init__(self,alpha=1,gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,logits,targets):

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs
        )

        focal_weight = ((1-pt)** self.gamma)

        if isinstance(self.alpha,torch.Tensor):

            alpha = self.alpha.unsqueeze(0)

        else:
            alpha = self.alpha

        focal_loss = (alpha* focal_weight* bce)

        return focal_loss.mean()