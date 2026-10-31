import torch
import torch.nn as nn


class BalancedDayFireLoss(nn.Module):
  """Symmetric Focal Tversky Loss for Daytime Satellite Wildfire Detection.

  Key Properties:
  1. True Negatives (the 32,000 background pixels) are completely excluded from the denominator.
     This eliminates the gradient avalanche that caused the Dead-Zero collapse.
  2. alpha = 0.5, beta = 0.5 applies symmetric penalties, preventing both the all-ones and all-zeros traps.
  3. gamma = 4/3 steepens gradients when overlap is poor, forcing the network out of early plateaus.
  """

  def __init__(self, alpha=0.5, beta=0.5, gamma=4.0 / 3.0, eps=1e-6):
    super(BalancedDayFireLoss, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.eps = eps

  def forward(self, logits, targets):
    # Map raw logits to continuous probabilities in [0.0, 1.0]
    probs = torch.sigmoid(logits)

    # Flatten spatial dimensions per image in the batch: [Batch, H * W]
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    # Compute continuous confusion components
    true_pos = torch.sum(probs_flat * targets_flat, dim=1)
    false_pos = torch.sum(probs_flat * (1.0 - targets_flat), dim=1)
    false_neg = torch.sum((1.0 - probs_flat) * targets_flat, dim=1)

    # Symmetric Tversky index
    tversky = (true_pos + self.eps) / (
        true_pos
        + self.alpha * false_pos
        + self.beta * false_neg
        + self.eps
    )

    # Focal modulation
    loss = torch.pow((1.0 - tversky), (1.0 / self.gamma))
    return torch.mean(loss)