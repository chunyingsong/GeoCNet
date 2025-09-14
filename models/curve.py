import torch
import torch.nn as nn
class CurveLoss(nn.Module):
    def __init__(self, smooth_weight=1.0, continuity_weight=1.0):
        super(CurveLoss, self).__init__()
        self.smooth_weight = smooth_weight
        self.continuity_weight = continuity_weight

    def forward(self, pred_xs, pred_zs, gt_xs, gt_zs, anchor_y_steps, loc_mask):
        """
        Calculate smoothness and continuity loss for lane curves.

        Args:
            pred_xs (torch.Tensor): Predicted x coordinates, shape [batch_size, num_y_steps].
            pred_zs (torch.Tensor): Predicted z coordinates, shape [batch_size, num_y_steps].
            gt_xs (torch.Tensor): Ground truth x coordinates, shape [batch_size, num_y_steps].
            gt_zs (torch.Tensor): Ground truth z coordinates, shape [batch_size, num_y_steps].
            anchor_y_steps (torch.Tensor): Preset y coordinates, shape [1, 1, num_y_steps].
            loc_mask (torch.Tensor): Visibility mask, shape [batch_size, num_y_steps].

        Returns:
            smooth_loss (torch.Tensor): Smoothness loss.
            continuity_loss (torch.Tensor): Continuity loss.
        """
        # Adjust anchor_y_steps shape from [1, 1, 20] to [20]
        anchor_y_steps = anchor_y_steps.squeeze()  # [num_y_steps]

        # Combine x, y, z coordinates into 3D points
        pred_points = torch.stack([
            pred_xs,  # [batch_size, num_y_steps]
            anchor_y_steps[None, :].expand_as(pred_xs),  # [batch_size, num_y_steps]
            pred_zs  # [batch_size, num_y_steps]
        ], dim=-1)  # [batch_size, num_y_steps, 3]

        gt_points = torch.stack([
            gt_xs,  # [batch_size, num_y_steps]
            anchor_y_steps[None, :].expand_as(gt_xs),  # [batch_size, num_y_steps]
            gt_zs  # [batch_size, num_y_steps]
        ], dim=-1)  # [batch_size, num_y_steps, 3]

        # Apply visibility mask
        pred_points = pred_points[loc_mask]  # [num_visible_points, 3]
        gt_points = gt_points[loc_mask]  # [num_visible_points, 3]

        # Smoothness loss: penalize the second derivative of the curve
        pred_dx = torch.diff(pred_points, dim=0)  # First derivative
        pred_ddx = torch.diff(pred_dx, dim=0)  # Second derivative
        smooth_loss = torch.mean(torch.abs(pred_ddx))

        # Continuity loss: penalize the difference between consecutive points
        pred_diff = torch.diff(pred_points, dim=0)
        gt_diff = torch.diff(gt_points, dim=0)
        continuity_loss = torch.mean(torch.abs(pred_diff - gt_diff))

        return self.smooth_weight * smooth_loss, self.continuity_weight * continuity_loss
