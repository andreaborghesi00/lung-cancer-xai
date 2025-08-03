import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        alpha: weight for class 1 (positive class), float scalar.
        gamma: focusing parameter.
        reduction: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits, shape (N,) or (N,1)
        targets: labels, shape (N,) or (N,1), values in {0,1}
        """
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
