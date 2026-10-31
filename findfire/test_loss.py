import torch
from losses_and_metrics import FocalTverskyLoss, calculate_segmentation_metrics

criterion = FocalTverskyLoss(alpha=0.3, beta=0.7)

# 1. Create a dummy ground truth mask with sparse fires (simulating 66 fire pixels)
targets = torch.zeros(4, 1, 128, 256)
targets[:, :, 20:25, 200:215] = 1.0  # Fire patch

# Case A: Perfect Predictions (Large positive logits -> Sigmoid -> ~1.0 on fires)
perfect_logits = torch.full_like(targets, -10.0)  # Background = 0.0
perfect_logits[:, :, 20:25, 200:215] = 10.0  # Fire = 1.0
loss_perfect = criterion(perfect_logits, targets)
metrics_perfect = calculate_segmentation_metrics(perfect_logits, targets)

# Case B: Degenerate "All Background" Model (Outputs zero fires everywhere)
all_zeros_logits = torch.full_like(targets, -10.0)
loss_zeros = criterion(all_zeros_logits, targets)
metrics_zeros = calculate_segmentation_metrics(all_zeros_logits, targets)

# Case C: Completely Inverted Predictions (Predicts fire where there is none)
inverted_logits = torch.full_like(targets, 10.0)
inverted_logits[:, :, 20:25, 200:215] = -10.0
loss_inverted = criterion(inverted_logits, targets)
metrics_inverted = calculate_segmentation_metrics(inverted_logits, targets)

print("--- Sanity Check: Focal Tversky Loss & Metrics ---")
print(
    f"Case A (Perfect Match)  -> Loss: {loss_perfect.item():.5f} | Dice:"
    f" {metrics_perfect['dice']:.4f} | Recall: {metrics_perfect['recall']:.4f}"
)
print(
    f"Case B (All Background) -> Loss: {loss_zeros.item():.5f} | Dice:"
    f" {metrics_zeros['dice']:.4f} | Recall: {metrics_zeros['recall']:.4f}"
)
print(
    f"Case C (Fully Inverted) -> Loss: {loss_inverted.item():.5f} | Dice:"
    f" {metrics_inverted['dice']:.4f} | Recall:"
    f" {metrics_inverted['recall']:.4f}"
)

# Mathematical assertions
assert (
    loss_perfect < 0.05
), "Loss on perfect predictions must be close to zero!"
assert (
    loss_zeros > 0.80
), "Loss on degenerate all-background predictions must be heavily penalized!"
assert (
    metrics_perfect["dice"] > 0.99
), "Dice score on perfect predictions must be ~1.0!"
assert (
    metrics_zeros["recall"] < 0.01
), "Recall on all-zero predictions must be 0.0!"
print("\nAll mathematical constraints validated successfully.")