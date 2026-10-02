import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def extract_foreground_tracking_points(
    tracking_points, silhouette, depth, depth_threshold=10, device=None
):
    """Extract tracking points that belong to the foreground from a sequence format (T x N x 3)."""
    # Get image dimensions from silhouette
    H, W = silhouette.shape

    # Extract x, y coordinates from first frame tracking points
    points_first_frame = tracking_points[0]  # N x 3
    xy_coords = points_first_frame[:, :2].long()  # N x 2

    # Check if coordinates are within image bounds
    x_valid = (xy_coords[:, 0] >= 0) & (xy_coords[:, 0] < W)
    y_valid = (xy_coords[:, 1] >= 0) & (xy_coords[:, 1] < H)
    in_bounds = x_valid & y_valid

    # Get silhouette values at point locations
    foreground_mask = torch.zeros(xy_coords.shape[0], dtype=torch.bool, device=device)
    valid_xy = xy_coords[in_bounds]

    if valid_xy.shape[0] > 0:
        sil_values = silhouette[valid_xy[:, 1], valid_xy[:, 0]]
        depth_values = depth[valid_xy[:, 1], valid_xy[:, 0]]
        foreground_mask[in_bounds] = (sil_values > 0.5) & (depth_values < depth_threshold)

    # Count foreground points
    num_fg_points = foreground_mask.sum().item()
    print(
        f"Found {num_fg_points} foreground points (out of {foreground_mask.shape[0]} total)"
    )

    if num_fg_points == 0:
        raise ValueError(
            "No foreground points found. Check tracking points and silhouette alignment."
        )

    # Extract foreground points for all frames
    foreground_points = tracking_points[:, foreground_mask]

    return foreground_points, foreground_mask


def align_tracking_points(tracking_points, ref_depth):
    """
    Aligns the tracking points with reference depth by finding optimal scale and offset
    for Z coordinates to minimize depth difference in the first frame.
    
    Args:
        tracking_points: T X N X 3, from frame 0 to T-1
        ref_depth: H X W, for frame 0
        
    Returns:
        aligned_tracking_points: T X N X 3, with adjusted Z values
    """
    # Validate inputs
    if tracking_points is None or ref_depth is None:
        raise ValueError("Inputs cannot be None")
    
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(tracking_points, torch.Tensor):
        tracking_points = torch.tensor(tracking_points, dtype=torch.float32)
    if not isinstance(ref_depth, torch.Tensor):
        ref_depth = torch.tensor(ref_depth, dtype=torch.float32)
    
    # Check input shapes
    if tracking_points.dim() != 3 or tracking_points.shape[2] != 3:
        raise ValueError(f"tracking_points should have shape TxNx3, got {tracking_points.shape}")
    if ref_depth.dim() != 2:
        raise ValueError(f"ref_depth should have shape HxW, got {ref_depth.shape}")
    
    # Check for NaN or Inf values
    if torch.isnan(tracking_points).any() or torch.isinf(tracking_points).any():
        raise ValueError("tracking_points contains NaN or Inf values")
    if torch.isnan(ref_depth).any() or torch.isinf(ref_depth).any():
        raise ValueError("ref_depth contains NaN or Inf values")
    
    # Get the first frame tracking points
    first_frame_points = tracking_points[0]  # N x 3
    
    # Extract xy coordinates (in pixels) and z values (depth)
    xy_coords = first_frame_points[:, :2]  # N x 2
    z_values = first_frame_points[:, 2]  # N x 1
    
    # Check if xy coordinates are within image bounds
    h, w = ref_depth.shape
    if (xy_coords[:, 0].min() < 0 or xy_coords[:, 0].max() >= w or
        xy_coords[:, 1].min() < 0 or xy_coords[:, 1].max() >= h):
        print("Warning: Some tracking points are outside the image boundaries")
    
    # Sample depth values from reference depth map at xy coordinates
    # Normalize xy coordinates to [-1, 1] range for grid_sample
    normalized_coords = xy_coords.clone()
    normalized_coords[:, 0] = (normalized_coords[:, 0] / (w - 1)) * 2 - 1  # x coord
    normalized_coords[:, 1] = (normalized_coords[:, 1] / (h - 1)) * 2 - 1  # y coord
    
    # Reshape for grid_sample
    grid = normalized_coords.view(1, -1, 1, 2)  # 1 x N x 1 x 2
    ref_depth_expanded = ref_depth.unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
    
    # Sample reference depth values at xy coordinates
    sampled_depth = F.grid_sample(
        ref_depth_expanded, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    sampled_depth = sampled_depth.squeeze()  # N
    
    # Remove invalid samples (zeros or NaNs from padding)
    valid_mask = (sampled_depth > 0) & ~torch.isnan(sampled_depth) & (sampled_depth < 10)
    
    # Check if we have enough valid points for alignment
    if valid_mask.sum() < 2:
        print("Warning: Not enough valid depth samples for accurate alignment, returning original points")
        return tracking_points
    
    z_values_valid = z_values[valid_mask]
    sampled_depth_valid = sampled_depth[valid_mask]
    
    # Compute scale and offset through least squares
    # We want to find s and t such that s * z_values + t ≈ sampled_depth
    # Using direct solution to least squares problem
    
    # Build the system matrix A and target vector b
    A = torch.stack([z_values_valid, torch.ones_like(z_values_valid)], dim=1)  # N' x 2
    b = sampled_depth_valid  # N'
    
    
    try:
        # Solve the least squares problem: min ||A @ [s, t] - b||^2
        # Using newer torch.linalg.lstsq instead of deprecated torch.lstsq
        solution_result = torch.linalg.lstsq(A, b.unsqueeze(1))
        solution = solution_result.solution
        
        if torch.isnan(solution).any() or torch.isinf(solution).any():
            # Fallback to a more robust but less accurate approach
            scale = sampled_depth_valid.mean() / z_values_valid.mean() if z_values_valid.mean() > 0 else 1.0
            offset = sampled_depth_valid.mean() - scale * z_values_valid.mean()
            print("Warning: Least squares solution contains NaN/Inf, using fallback scaling")
        else:
            scale, offset = solution.squeeze()
            
        # Check if scale is reasonable (not too extreme)
        if scale < 0.1 or scale > 10.0:
            print(f"Warning: Scale factor {scale:.4f} seems extreme, might indicate poor alignment")
        
    except RuntimeError as e:
        # If least squares fails, use a simpler approach
        print(f"Error in least squares: {e}")
        print("Using fallback method: matching means")
        scale = sampled_depth_valid.mean() / z_values_valid.mean() if z_values_valid.mean() > 0 else 1.0
        offset = sampled_depth_valid.mean() - scale * z_values_valid.mean()
    
    # Apply scale and offset to all frames
    aligned_tracking_points = tracking_points.clone()
    aligned_tracking_points[..., 2] = scale * tracking_points[..., 2] + offset
    
    # temporal_offset = aligned_tracking_points - aligned_tracking_points[0].unsqueeze(0)
    
    # print(temporal_offset.shape)
    # print(aligned_tracking_points.shape)
    # print(b.shape)
    
    # aligned_tracking_points[:, :, 2] = b.reshape(1, -1) + temporal_offset[:, :, 2]
    
    return aligned_tracking_points, sampled_depth

def query_depth(depth_map, tracking_points):
    """
    Query depth values from depth map at tracking points.
    
    Args:
        depth_map: HxW depth map
        tracking_points: Nx3 tracking points (x, y coordinates should be in pixel space)
        
    Returns:
        depth_values: N depth values
    """
    # Convert tracking points to normalized coordinates
    h, w = depth_map.shape
    
    # Extract x, y coordinates from tracking points
    x_coords = tracking_points[:, 0]  # N
    y_coords = tracking_points[:, 1]  # N
    
    # Convert to tensor if not already
    if not torch.is_tensor(x_coords):
        x_coords = torch.tensor(x_coords, device=depth_map.device, dtype=depth_map.dtype)
        y_coords = torch.tensor(y_coords, device=depth_map.device, dtype=depth_map.dtype)
    
    # Round to nearest pixel coordinates
    x_nearest = torch.round(x_coords).long()
    y_nearest = torch.round(y_coords).long()
    
    # Clamp coordinates to valid range
    x_nearest = torch.clamp(x_nearest, 0, w - 1)
    y_nearest = torch.clamp(y_nearest, 0, h - 1)
    
    # Sample depth values at nearest pixels
    depth_values = depth_map[y_nearest, x_nearest]
    
    return depth_values  # Return as N

def visualize_alignment(tracking_points, aligned_points, ref_depth, save_path=None):
    """
    Visualizes the alignment of tracking points with reference depth.
    
    Args:
        tracking_points: Original tracking points (TxNx3)
        aligned_points: Aligned tracking points (TxNx3)
        ref_depth: Reference depth image (HxW)
        save_path: Optional path to save the visualization
        
    Returns:
        None (displays or saves the visualization)
    """
    if not isinstance(tracking_points, torch.Tensor):
        tracking_points = torch.tensor(tracking_points)
    if not isinstance(aligned_points, torch.Tensor):
        aligned_points = torch.tensor(aligned_points)
    if not isinstance(ref_depth, torch.Tensor):
        ref_depth = torch.tensor(ref_depth)
    
    # Get first frame points
    orig_points = tracking_points[0]
    aligned_first = aligned_points[0]
    
    # Convert to numpy for plotting
    orig_z = orig_points[:, 2].cpu().numpy()
    aligned_z = aligned_first[:, 2].cpu().numpy()
    xy_coords = orig_points[:, :2].cpu().numpy().astype(int)
    
    # Sample depth values from reference
    h, w = ref_depth.shape
    ref_z = []
    valid_indices = []
    
    for i, (x, y) in enumerate(xy_coords):
        if 0 <= x < w and 0 <= y < h:
            depth_val = ref_depth[y, x].item()
            if depth_val > 0 and not np.isnan(depth_val):
                ref_z.append(depth_val)
                valid_indices.append(i)
    
    ref_z = np.array(ref_z)
    valid_indices = np.array(valid_indices)
    
    if len(valid_indices) == 0:
        print("No valid points for visualization")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot depth image
    depth_img = ref_depth.cpu().numpy()
    im = axes[0, 0].imshow(depth_img, cmap='viridis')
    axes[0, 0].scatter(xy_coords[valid_indices, 0], xy_coords[valid_indices, 1], 
                      c='r', s=10, alpha=0.7)
    axes[0, 0].set_title('Reference Depth Image with Tracking Points')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Plot depth comparison
    axes[0, 1].scatter(ref_z, orig_z[valid_indices], alpha=0.5, label='Original Z')
    axes[0, 1].scatter(ref_z, aligned_z[valid_indices], alpha=0.5, label='Aligned Z')
    min_val = min(np.min(ref_z), np.min(orig_z[valid_indices]), np.min(aligned_z[valid_indices]))
    max_val = max(np.max(ref_z), np.max(orig_z[valid_indices]), np.max(aligned_z[valid_indices]))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
    axes[0, 1].set_xlabel('Reference Depth')
    axes[0, 1].set_ylabel('Tracking Point Depth')
    axes[0, 1].set_title('Depth Comparison')
    axes[0, 1].legend()
    
    # Plot depth distributions
    axes[1, 0].hist(ref_z, bins=30, alpha=0.5, label='Reference')
    axes[1, 0].hist(orig_z[valid_indices], bins=30, alpha=0.5, label='Original')
    axes[1, 0].set_xlabel('Depth Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Original Depth Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].hist(ref_z, bins=30, alpha=0.5, label='Reference')
    axes[1, 1].hist(aligned_z[valid_indices], bins=30, alpha=0.5, label='Aligned')
    axes[1, 1].set_xlabel('Depth Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Aligned Depth Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def export_tracking_video(tracking_points, tracking_confs, rgbs, debug_dir):
    """Export tracking video with points overlaid, following demo.py logic."""
    print("Exporting tracking visualization video...")
    
    # Import necessary modules
    try:
        # Add alltracker path for imports
        import sys
        alltracker_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'alltracker')
        if alltracker_path not in sys.path:
            sys.path.append(alltracker_path)
        
        import utils.improc as alltracker_improc
        import PIL.Image
        import imageio
    except ImportError as e:
        print(f"Warning: Cannot export tracking video due to missing dependencies: {e}")
        return
    
    # tracking_points: T,N,2
    # tracking_confs: T,N  
    # rgbs: 1,T,3,H,W (0-255 range)
    
    T, N, _ = tracking_points.shape
    _, _, _, H, W = rgbs.shape
    
    # Convert rgbs back to the format expected by draw_pts_gpu
    # rgbs is [1,T,3,H,W], we need [T,3,H,W]
    rgbs_viz = rgbs[0].clone()  # T,3,H,W
    
    # Create colormap for points (similar to demo.py)
    # Get initial positions for color assignment
    xy0 = tracking_points[0].cpu().numpy()  # N,2
    colors = alltracker_improc.get_2d_colors(xy0, H, W)
    
    # Prepare data for visualization
    trajs_viz = tracking_points.unsqueeze(0)  # 1,T,N,2 (add batch dimension)
    visibilities = tracking_confs > 0.1  # T,N (confidence threshold)
    visibilities = visibilities.unsqueeze(0)  # 1,T,N (add batch dimension)
    
    # Draw tracking points on frames (following demo.py logic)
    rate = 4  # visualization subsampling rate
    bkg_opacity = 0.5  # background opacity
    frames = draw_pts_gpu(rgbs_viz, trajs_viz[0], visibilities[0], colors, rate=rate, bkg_opacity=bkg_opacity)
    
    # Save video to debug directory
    video_path = os.path.join(debug_dir, "tracking_visualization.mp4")
    temp_dir = os.path.join(debug_dir, "temp_tracking_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save individual frames
    print(f"Saving {T} frames to {temp_dir}")
    for t in range(T):
        frame_path = os.path.join(temp_dir, f"{t:03d}.jpg")
        im = PIL.Image.fromarray(frames[t])
        im.save(frame_path)
    
    # Create video using ffmpeg
    framerate = 30  # Default framerate
    ffmpeg_cmd = (
        f"/usr/bin/ffmpeg -y -hide_banner -loglevel error "
        f"-f image2 -framerate {framerate} -pattern_type glob "
        f"-i \"{temp_dir}/*.jpg\" -c:v libx264 -crf 20 -pix_fmt yuv420p \"{video_path}\""
    )
    
    print(f"Creating tracking video: {video_path}")
    os.system(ffmpeg_cmd)
    
    # Clean up temporary frames
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Tracking video saved to: {video_path}")


def draw_pts_gpu(rgbs, trajs, visibs, colormap, rate=1, bkg_opacity=0.5):
    """Draw tracking points on frames (adapted from demo.py)."""
    device = rgbs.device
    T, C, H, W = rgbs.shape
    trajs = trajs.permute(1, 0, 2)  # N,T,2
    visibs = visibs.permute(1, 0)  # N,T
    N = trajs.shape[0]
    colors = torch.tensor(colormap, dtype=torch.float32, device=device)  # [N,3]

    rgbs = rgbs * bkg_opacity  # darken, to see the point tracks better
    
    opacity = 1.0
    if rate == 1:
        radius = 1
        opacity = 0.9
    elif rate == 2:
        radius = 1
    elif rate == 4:
        radius = 2
    elif rate == 8:
        radius = 4
    else:
        radius = 6
    sharpness = 0.15 + 0.05 * np.log2(rate)
    
    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1)  # [D,D]
    icon = icon.view(1, D, D)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")  # [D,D]
    
    for t in range(T):
        mask = visibs[:, t]  # [N]
        if mask.sum() == 0:
            continue
        xy = trajs[mask, t] + 0.5  # [N,2]
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]  # [N,3]
        N_vis = xy.shape[0]
        cx = xy[:, 0].long()  # [N]
        cy = xy[:, 1].long()
        x_grid = cx[:, None, None] + disp_x  # [N,D,D]
        y_grid = cy[:, None, None] + disp_y  # [N,D,D]
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]  # [K]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(N_vis, D, D)[valid]  # [K]
        colors_valid = colors_now[:, :, None, None].expand(N_vis, 3, D, D).permute(1, 0, 2, 3)[
            :, valid
        ]  # [3, K]
        idx_flat = (y_valid * W + x_valid).long()  # [K]

        accum = torch.zeros_like(rgbs[t])  # [3, H, W]
        weight = torch.zeros(1, H * W, device=device)  # [1, H*W]
        img_flat = accum.view(C, -1)  # [3, H*W]
        weighted_colors = colors_valid * icon_weights  # [3, K]
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        alpha = weight.clamp(0, 1) * opacity
        accum = accum / (weight + 1e-6)  # [3, H, W]
        rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha
        
    rgbs = rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # T,H,W,3
    if bkg_opacity == 0.0:
        for t in range(T):
            hsv_frame = cv2.cvtColor(rgbs[t], cv2.COLOR_RGB2HSV)
            saturation_factor = 1.5
            hsv_frame[..., 1] = np.clip(hsv_frame[..., 1] * saturation_factor, 0, 255)
            rgbs[t] = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
    return rgbs 
