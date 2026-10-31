import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
  """Focal Tversky Loss for highly imbalanced binary semantic segmentation.

  Formula:
      TI = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
      FTL = (1 - TI) ** (1 / gamma)
  """

  def __init__(self, alpha=0.3, beta=0.7, gamma=4.0 / 3.0, eps=1e-6):
    super(FocalTverskyLoss, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.eps = eps

  def forward(self, logits, targets):
    """Args:

        logits: Tensor of shape [B, 1, H, W] (Unconstrained raw model outputs)
        targets: Tensor of shape [B, 1, H, W] (Binary masks with values 0.0 or
        1.0)
    """
    # 1. Map raw logits to continuous probabilities via Sigmoid activation
    probs = torch.sigmoid(logits)

    # 2. Flatten spatial dimensions per batch item: [B, H*W]
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    # 3. Compute continuous confusion matrix terms
    true_pos = torch.sum(probs_flat * targets_flat, dim=1)
    false_pos = torch.sum(probs_flat * (1.0 - targets_flat), dim=1)
    false_neg = torch.sum((1.0 - probs_flat) * targets_flat, dim=1)

    # 4. Compute Tversky Index per batch item
    tversky_index = (true_pos + self.eps) / (
        true_pos
        + self.alpha * false_pos
        + self.beta * false_neg
        + self.eps
    )

    # 5. Apply Focal modulation and average across batch
    focal_tversky = torch.pow((1.0 - tversky_index), (1.0 / self.gamma))
    return torch.mean(focal_tversky)


# =====================================================================
# EVALUATION METRICS (Binarized via Threshold = 0.5)
# =====================================================================
@torch.no_grad()
def calculate_segmentation_metrics(logits, targets, threshold=0.5, eps=1e-6):
  """Computes Dice (F1), IoU (Jaccard), Precision, and Recall on binarized predictions.

  Args:
      logits: Tensor of shape [B, 1, H, W]
      targets: Tensor of shape [B, 1, H, W]
  Returns:
      dict: Averaged scalar metrics across the batch
  """
  probs = torch.sigmoid(logits)
  preds = (probs > threshold).float()

  preds_flat = preds.view(preds.size(0), -1)
  targets_flat = targets.view(targets.size(0), -1)

  tp = torch.sum(preds_flat * targets_flat, dim=1)
  fp = torch.sum(preds_flat * (1.0 - targets_flat), dim=1)
  fn = torch.sum((1.0 - preds_flat) * targets_flat, dim=1)

  # Dice Score / F1: 2*TP / (2*TP + FP + FN)
  dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

  # IoU / Jaccard Index: TP / (TP + FP + FN)
  iou = (tp + eps) / (tp + fp + fn + eps)

  # Precision: TP / (TP + FP)
  precision = (tp + eps) / (tp + fp + eps)

  # Recall / Sensitivity: TP / (TP + FN) -> Crucial for wildfire detection
  recall = (tp + eps) / (tp + fn + eps)

  return {
      "dice": torch.mean(dice).item(),
      "iou": torch.mean(iou).item(),
      "precision": torch.mean(precision).item(),
      "recall": torch.mean(recall).item(),
  }