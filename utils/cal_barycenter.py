import torch

def extract_color_hard_silhouettes(image_rgba: torch.Tensor, torch_color_list, threshold=0.01):
    assert image_rgba.shape[0] == 1 and image_rgba.shape[-1] == 4
    _, H, W, _ = image_rgba.shape

    device = image_rgba.device

    rgb = image_rgba[..., :3]  # (1, H, W, 3)

    masks = []
    for target_rgb in torch_color_list:
        # Compute color distance
        dist = torch.norm(rgb - target_rgb.view(1, 1, 1, 3), dim=-1)  # (1, H, W)
        mask = (dist < threshold).float()  # (1, H, W)
        masks.append(mask)

    # Stack into (1, H, W, 4)
    silhouettes_tensor = torch.cat(masks, dim=0).permute(1, 2, 0).unsqueeze(0)

    return silhouettes_tensor  # (1, H, W, 4)

def extract_color_soft_silhouettes(image_rgba: torch.Tensor, torch_color_list, threshold=0.01, sharpness=100.0):
    """
    Extract differentiable silhouette masks per color from RGBA tensor.
    
    Args:
        image_rgba: (1, H, W, 4) tensor in [0, 1] range.
        threshold: color distance threshold to center soft mask.
        sharpness: controls sharpness of soft mask (higher = sharper, but still differentiable)

    Returns:
        silhouettes_tensor: (1, H, W, 4) tensor, channel order is [red, white, green, blue]
    """
    _, H, W, _ = image_rgba.shape

    device = image_rgba.device

    # Define target colors in fixed order
    

    rgb = image_rgba[..., :3]  # (1, H, W, 3)

    masks = []
    for target_rgb in torch_color_list:
        # Compute color distance
        dist = torch.norm(rgb - target_rgb.view(1, 1, 1, 3), dim=-1)  # (1, H, W)

        # Differentiable mask using sigmoid
        mask = torch.sigmoid(-sharpness * (dist - threshold))  # (1, H, W)

        masks.append(mask)

    # Stack into (1, H, W, 4)
    silhouettes_tensor = torch.cat(masks, dim=0).permute(1, 2, 0).unsqueeze(0)

    return silhouettes_tensor  

def compute_barycenters_of_separated_meshes(separated_meshes):
    """
    Compute the barycenter (center of mass) of each mesh in separated_meshes.

    Args:
        separated_meshes: list of Meshes or None

    Returns:
        barycenters: Tensor of shape (num_groups, 3), barycenter of each mesh. 
                     If mesh is None, returns (0, 0, 0) for that entry.
    """
    device = separated_meshes[0].device if separated_meshes[0] is not None else "cpu"

    barycenters = []
    for mesh in separated_meshes:
        if mesh is None or mesh.isempty():
            # If mesh is None or empty, barycenter is (0, 0, 0)
            barycenters.append(torch.zeros(3, device=device))
        else:
            verts = mesh.verts_packed()  # (V, 3)
            barycenter = verts.mean(dim=0)  # (3,)
            barycenters.append(barycenter)

    # Stack into (num_groups, 3)
    barycenters = torch.stack(barycenters, dim=0)
    return barycenters

def compute_mask_pixel_barycenters(mask_tensor):
    """
    Compute barycenter (center of mass) and mask area ratio for each object mask in each frame.
    
    Args:
        mask_tensor: Tensor (N, G, H, W), values 0~1
    
    Returns:
        barycenters: Tensor (N, G, 2), each row is (x, y) barycenter in normalized coordinates (0~1)
        ratios: Tensor (N, G, 1), each is the area ratio of this object compared to total mask area in this frame
    """
    n_frames, obj_nums, H, W = mask_tensor.shape
    device = mask_tensor.device

    # Create pixel coordinate grid
    y_coords = torch.arange(H, device=device).float().view(H, 1).expand(H, W)  # (H, W)
    x_coords = torch.arange(W, device=device).float().view(1, W).expand(H, W)  # (H, W)

    barycenters_list = []
    ratios_list = []

    for frame_idx in range(n_frames):
        frame_mask = mask_tensor[frame_idx]  # (G, H, W)

        areas = frame_mask.view(obj_nums, -1).sum(dim=1)  # (G,)
        total_area = areas.sum() + 1e-6  # prevent divide by zero

        barycenters = []
        for g in range(obj_nums):
            mask = frame_mask[g]  # (H, W)
            weight = areas[g]

            if weight < 1e-6:
                barycenter = torch.tensor([0.5, 0.5], device=device)
            else:
                x_mean = (mask * x_coords).sum() / weight
                y_mean = (mask * y_coords).sum() / weight
                x_mean = x_mean / (W - 1)
                y_mean = y_mean / (H - 1)
                barycenter = torch.stack([x_mean, y_mean], dim=0)

            barycenters.append(barycenter)

        barycenters = torch.stack(barycenters, dim=0)  # (G, 2)
        ratios = (areas / total_area).unsqueeze(1)  # (G, 1)

        barycenters_list.append(barycenters)
        ratios_list.append(ratios)

    barycenters = torch.stack(barycenters_list, dim=0)  # (N, G, 2)
    ratios = torch.stack(ratios_list, dim=0)  # (N, G, 1)
    binary_ratios = (ratios > 0.01).float()

    return barycenters, ratios, binary_ratios