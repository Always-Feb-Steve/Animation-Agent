import torch
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch3d.transforms import axis_angle_to_quaternion
from math import exp

def l1_loss(network_output, gt, conf=None):
    assert network_output.shape == gt.shape
    if conf is not None:
        assert network_output.shape[0] == conf.shape[0]
        return torch.abs((network_output - gt) * conf).mean()
    else:
        return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt, conf=None):
    assert network_output.shape == gt.shape
    if conf is not None:
        assert network_output.shape[0] == conf.shape[0]
        return ((network_output - gt) ** 2 * conf).mean()
    else:
        return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def rotation_reg_loss(rotations_seq, lambda_reg=1.0, threshold=torch.pi/10.0):
    """
    Compute the regularization loss to prevent rotations from being too large.
    This function only penalizes rotations that exceed the specified threshold.
    
    Args:
        rotations_seq: Tensor of shape (..., 3) containing axis-angle rotations
        lambda_reg: Weighting factor for the regularization
        threshold: Angle threshold in radians. Only rotations larger than this will be penalized.
                  Default is π/10 (18 degrees)
        
    Returns:
        Scalar loss term penalizing large rotations
    """
    # Calculate the magnitude (angle) of each rotation
    # In axis-angle, the magnitude represents the rotation angle in radians
    rotation_magnitudes = torch.norm(rotations_seq, dim=-1)
    
    # Create a mask for rotations exceeding the threshold
    mask = rotation_magnitudes > threshold
    
    # If no rotations exceed the threshold, return zero loss
    if not torch.any(mask):
        return torch.tensor(0.0, device=rotations_seq.device)
    
    # Calculate excess rotation beyond the threshold
    excess_rotation = (rotation_magnitudes - threshold) * mask.float()
    
    # Square the excess to more heavily penalize very large rotations
    squared_excess = excess_rotation ** 2
    
    # Mean of squared excess, factoring in the weighting
    loss = lambda_reg * torch.sum(squared_excess) / (mask.sum() + 1e-8)
    
    return loss

def translation_reg_loss(translations_seq, lambda_reg=1.0, threshold=0.0):
    """
    Compute the regularization loss to prevent translations from being too large.
    This function only penalizes translations that exceed the specified threshold.
    
    Args:
        translations_seq: Tensor of shape (..., 3) containing 3D translations
        lambda_reg: Weighting factor for the regularization
        threshold: Distance threshold in world units. Only translations larger than this will be penalized.
                  Default is 0.05 (5cm if units are in meters)
        
    Returns:
        Scalar loss term penalizing large translations
    """
    # Calculate the magnitude (distance) of each translation
    translation_magnitudes = torch.norm(translations_seq, dim=-1)
    
    # Create a mask for translations exceeding the threshold
    mask = translation_magnitudes > threshold
    
    # If no translations exceed the threshold, return zero loss
    if not torch.any(mask):
        return torch.tensor(0.0, device=translations_seq.device)
    
    # Calculate excess translation beyond the threshold
    excess_translation = (translation_magnitudes - threshold) * mask.float()
    
    # Square the excess to more heavily penalize very large translations
    squared_excess = excess_translation ** 2
    
    # Mean of squared excess, factoring in the weighting
    loss = lambda_reg * torch.sum(squared_excess) / (mask.sum() + 1e-8)
    
    return loss

def rotation_smooth_loss(rotations_seq, lambda_pos=1.0, lambda_vel=1.0):
    """
    Compute the smooth loss for a sequence of rotations.
    Args:
        rotations_seq: (N, T, 3) tensor of rotations, axis-angle representation
        lambda_pos: weight for position loss
        lambda_vel: weight for velocity loss
    """
    # Compute the difference between consecutive rotations
    pos_diff = rotations_seq[:, :-1, :] - rotations_seq[:, 1:, :]
    
    vel_diff = rotations_seq[:, :-2, :] - 2 * rotations_seq[:, 1:-1, :] + rotations_seq[:, 2:, :]
    
    pos_reg = lambda_pos * l1_loss(pos_diff, torch.zeros_like(pos_diff))
    vel_reg = lambda_vel * l1_loss(vel_diff, torch.zeros_like(vel_diff))
    
    return pos_reg + vel_reg

def smooth_loss(seq, lambda_first=1.0, lambda_second=0.3):
    """
    Compute the smooth loss for a dynamic sequence.
    """
    # first_order_diff = seq[:, :-1, :] - seq[:, 1:, :]
    # second_order_diff = seq[:, :-2, :] - 2 * seq[:, 1:-1, :] + seq[:, 2:, :]
    first_order_diff = seq[:-1, :] - seq[1:, :]
    second_order_diff = seq[:-2, :] - 2 * seq[1:-1, :] + seq[2:, :]

    first_order_reg = lambda_first * l1_loss(
        first_order_diff, torch.zeros_like(first_order_diff)
    )
    second_order_reg = lambda_second * l1_loss(
        second_order_diff, torch.zeros_like(second_order_diff)
    )

    return first_order_reg + second_order_reg

def position_smooth_loss(positions_seq, lambda_pos=1.0, lambda_vel=1.0):
    """
    Compute the smooth loss for a sequence of positions.
    """
    # Compute the difference between consecutive positions
    pos_diff = positions_seq[:, :-1, :] - positions_seq[:, 1:, :]
    
    vel_diff = positions_seq[:, :-2, :] - 2 * positions_seq[:, 1:-1, :] + positions_seq[:, 2:, :]
    
    pos_reg = lambda_pos * l1_loss(pos_diff, torch.zeros_like(pos_diff))
    vel_reg = lambda_vel * l1_loss(vel_diff, torch.zeros_like(vel_diff))
    
    return pos_reg + vel_reg

def quaternion_geodesic_loss(rotations_seq):
    """
    Compute the geodesic loss for a sequence of quaternions.
    Measures the smoothness of rotation transitions using the geodesic distance
    between consecutive quaternions.
    
    Args:
        rotations_seq: Tensor of shape (T, 3) with axis-angle rotations
                        
    Returns:
        Total geodesic loss measuring rotation smoothness (lower is smoother)
    """
    if rotations_seq.shape[0] <= 1:
        return 0.0  # No loss for a single quaternion or empty sequence
    
    quaternions_seq = axis_angle_to_quaternion(rotations_seq)
    
    # Ensure quaternions are normalized
    quaternions_seq = quaternions_seq / torch.norm(quaternions_seq, dim=1, keepdim=True)
    
    # Compute inner products between consecutive quaternions
    # Note: Using the absolute value of the dot product since q and -q represent the same rotation
    dot_products = torch.abs(torch.sum(
        quaternions_seq[:-1] * quaternions_seq[1:],
        dim=1
    ))
    
    # Clamp dot products to valid range for numerical stability
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    
    # Compute the geodesic distances (arccos of dot products)
    # This gives the angle between quaternions in radians
    geodesic_distances = torch.acos(dot_products)
    
    # Sum the squared distances to get the total loss
    # Squaring emphasizes larger jumps in rotation
    loss = torch.sum(geodesic_distances ** 2)
    
    return loss


def normalized_depth_loss(depth_pred, depth_gt, conf=None, conf_thres=0.45, eps=1e-8):
    """
    Compute L1 loss between scale-normalized depth maps, processed frame by frame.
    
    For each frame, both depth maps are normalized using median centering and scale 
    normalization. The scale is computed as the difference between 95th and 5th 
    percentiles (robust to outliers). Statistics are computed only on points with 
    confidence > conf_thres.
    
    Args:
        depth_pred: Predicted depth map tensor. Can be:
                   - 1D: (N,) - single batch of N points
                   - 2D: (B, N) - B batches of N points each
                   - 2D: (T, N) - T timesteps of N points each
        depth_gt: Ground truth depth map tensor (same shape as depth_pred)
        conf: Optional confidence weights tensor (same shape as depth tensors).
              Used both for filtering statistics computation and weighting the final loss.
        conf_thres: Confidence threshold. Only points with conf > conf_thres are used
                    to compute median and scale statistics (default: 0.5)
        eps: Small epsilon value to prevent division by zero (default: 1e-8)
        
    Returns:
        Normalized L1 loss between depth maps
        
    Raises:
        AssertionError: If input tensors have mismatched shapes or are empty
    """
    assert depth_pred.shape == depth_gt.shape, f"Shape mismatch: pred {depth_pred.shape} vs gt {depth_gt.shape}"
    assert depth_pred.numel() > 0, "Input tensors cannot be empty"
    
    # Handle different tensor dimensions
    if depth_pred.dim() == 1:
        # 1D case: (N,) - compute statistics across all elements
        depth_pred = depth_pred.unsqueeze(0)
        depth_gt = depth_gt.unsqueeze(0)
        if conf is not None:
            conf = conf.unsqueeze(0)
    elif depth_pred.dim() == 2:
        # 2D case: (B, N) or (T, N) - compute statistics across the last dimension (N)
        # This preserves batch/time dimension and computes per-batch/timestep statistics
        pass
    else:
        raise ValueError(f"Unsupported tensor dimension: {depth_pred.dim()}. Expected 1D or 2D tensors.")
    
    # Process frame by frame to handle confidence filtering
    num_frames = depth_pred.shape[0]
    num_points = depth_pred.shape[1]
    
    # median_depth_pred = torch.zeros(num_frames, 1, device=depth_pred.device, dtype=depth_pred.dtype)
    # median_depth_gt = torch.zeros(num_frames, 1, device=depth_gt.device, dtype=depth_gt.dtype)
    # scale_depth_pred = torch.zeros(num_frames, 1, device=depth_pred.device, dtype=depth_pred.dtype)
    # scale_depth_gt = torch.zeros(num_frames, 1, device=depth_gt.device, dtype=depth_gt.dtype)
    total_loss = torch.tensor(0.0, device=depth_pred.device)
    for frame_idx in range(num_frames):
        # Get data for this frame
        frame_depth_pred = depth_pred[frame_idx]  # Shape: (N,)
        frame_depth_gt = depth_gt[frame_idx]      # Shape: (N,)
        
        # Create confidence mask for this frame
        if conf is not None:
            frame_conf = conf[frame_idx]
            conf_mask = frame_conf > conf_thres  # Shape: (N,)
        else:
            # If no confidence provided, use all points
            conf_mask = torch.ones(num_points, dtype=torch.bool, device=depth_pred.device)
        
        # Filter points based on confidence threshold
        valid_depth_pred = frame_depth_pred[conf_mask]
        valid_depth_gt = frame_depth_gt[conf_mask]
        
        # Handle edge case where no points pass the threshold
        if valid_depth_pred.numel() == 0:
            # Fall back to using all points for this frame
            valid_depth_pred = frame_depth_pred
            valid_depth_gt = frame_depth_gt
        
        # Compute median for this frame (only on high-confidence points)
        median_depth_pred = torch.median(valid_depth_pred)
        median_depth_gt = torch.median(valid_depth_gt)
        
        # Compute scale using 95th - 5th percentile range (more robust to outliers)
        q95_pred = torch.quantile(valid_depth_pred, 0.95)
        q05_pred = torch.quantile(valid_depth_pred, 0.05)
        scale_depth_pred = q95_pred - q05_pred
        
        q95_gt = torch.quantile(valid_depth_gt, 0.95)
        q05_gt = torch.quantile(valid_depth_gt, 0.05)
        scale_depth_gt = q95_gt - q05_gt
        
        # Add epsilon to prevent division by zero when all values are identical
        scaled_depth_pred = torch.clamp(scale_depth_pred, min=eps)
        scaled_depth_gt = torch.clamp(scale_depth_gt, min=eps)
        
        depth_pred_normalized = (valid_depth_pred - median_depth_pred) / scaled_depth_pred
        depth_gt_normalized = (valid_depth_gt - median_depth_gt) / scaled_depth_gt
        
        loss = l1_loss(depth_pred_normalized, depth_gt_normalized)
        
        total_loss += loss
        
    return total_loss / num_frames
    
    # # Add epsilon to prevent division by zero when all values are identical
    # scale_depth_pred = torch.clamp(scale_depth_pred, min=eps)
    # scale_depth_gt = torch.clamp(scale_depth_gt, min=eps)
    
    # # Normalize both depth maps using per-frame statistics
    # depth_pred_normalized = (depth_pred - median_depth_pred) / scale_depth_pred
    # depth_gt_normalized = (depth_gt - median_depth_gt) / scale_depth_gt
    
    # return l1_loss(depth_pred_normalized, depth_gt_normalized, conf=conf)
