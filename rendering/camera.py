import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
)

def get_camera(dist, elev, azim, fov=60, znear=1, zfar=100, device=None):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, zfar=zfar, fov=fov)
    return cameras    

def world_to_image_space(camera, vertices):
    """
    Convert vertices from world space to image space using the camera transform.

    Args:
        camera: FoVPerspectiveCameras object
        vertices: Tensor of shape (V, 3) containing vertex coordinates in world space

    Returns:
        projected_verts: Tensor of shape (V, 2) containing vertex coordinates in image space
    """
    # Project vertices to NDC space (-1 to +1)
    projected_verts = camera.transform_points_ndc(vertices)

    # Convert from NDC to image space (0 to image_size)
    projected_verts = -projected_verts[..., :2]  # Keep only x,y coordinates and negate
    projected_verts = (projected_verts + 1) * 0.5  # Map from [-1,1] to [0,1]

    return projected_verts


def projection_matrix_to_intrinsic(projection_matrix, image_size):
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size

    if projection_matrix.dim() == 2:
        projection_matrix = projection_matrix.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = projection_matrix.shape[0]
    device = projection_matrix.device

    s1 = projection_matrix[:, 0, 0]
    s2 = projection_matrix[:, 1, 1]
    w1 = projection_matrix[:, 0, 2]
    h1 = projection_matrix[:, 1, 2]

    fx = s1 * width / 2.0
    fy = s2 * height / 2.0
    cx = width / 2.0 + w1 * width / 2.0
    cy = height / 2.0 + h1 * height / 2.0

    intrinsic_matrix = torch.zeros((batch_size, 3, 3), device=device, dtype=projection_matrix.dtype)
    intrinsic_matrix[:, 0, 0] = fx
    intrinsic_matrix[:, 1, 1] = fy
    intrinsic_matrix[:, 0, 2] = cx
    intrinsic_matrix[:, 1, 2] = cy
    intrinsic_matrix[:, 2, 2] = 1.0

    if squeeze_output:
        intrinsic_matrix = intrinsic_matrix.squeeze(0)
    return intrinsic_matrix


def get_camera_intrinsics(camera, image_size):
    projection_transform = camera.get_projection_transform()
    projection_matrix = projection_transform.get_matrix()
    return projection_matrix_to_intrinsic(projection_matrix, image_size)


def unproject_pixel_depth_map_to_point_map(pixel_depth_map, cameras, image_size=512):
    if not torch.is_tensor(pixel_depth_map):
        raise ValueError("pixel_depth_map must be a torch.Tensor")
    if pixel_depth_map.dim() != 2 or pixel_depth_map.shape[1] != 3:
        raise ValueError(f"pixel_depth_map must have shape (N, 3), got {pixel_depth_map.shape}")
    if pixel_depth_map.shape[0] == 0:
        return torch.empty((0, 3), dtype=pixel_depth_map.dtype, device=pixel_depth_map.device)

    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size

    pixel_x = pixel_depth_map[:, 0]
    pixel_y = pixel_depth_map[:, 1]
    depth = pixel_depth_map[:, 2]

    intrinsics = get_camera_intrinsics(cameras, image_size)
    if intrinsics.dim() == 3:
        intrinsics = intrinsics.squeeze(0)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    camera_x = (pixel_x - cx) * depth / fx
    camera_y = (pixel_y - cy) * depth / fy
    camera_z = depth

    return torch.stack([camera_x, camera_y, camera_z], dim=-1)


