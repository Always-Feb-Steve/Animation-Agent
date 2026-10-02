import torch
import numpy as np
import pyvista as pv
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pathlib import Path

def load_mesh(obj_path, load_materials=True, device=None):
    """Load mesh from obj file with materials and textures"""
    obj_path = Path(obj_path)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if load_materials:
        try:
            # Method 1: Using load_objs_as_meshes (preferred for textured meshes)
            meshes = load_objs_as_meshes([str(obj_path)], device=device)
            mesh = meshes[0]
            print("Successfully loaded mesh with materials using load_objs_as_meshes")
        except Exception as e:
            print(f"Failed to load with load_objs_as_meshes: {e}")
            print("Falling back to manual loading...")

            # Method 2: Manual loading with load_obj
            verts, faces, aux = load_obj(
                obj_path,
                device=device,
                load_textures=True,  # Important: Load texture data
                create_texture_atlas=True,  # Create texture atlas for complex materials
                texture_atlas_size=4096,  # Size of texture atlas
            )

            # Extract face data
            faces_idx = faces.verts_idx
            faces_uv = faces.textures_idx if faces.textures_idx is not None else None

            # Handle textures
            if faces_uv is not None and aux.texture_images is not None:
                # Create a TexturesUV object using the texture atlas
                texture_maps = aux.texture_images

                if isinstance(texture_maps, dict):
                    # If multiple textures, use the first one
                    texture_map = next(iter(texture_maps.values()))
                    if texture_map.device != device:
                        texture_map = texture_map.to(device)
                else:
                    texture_map = texture_maps
                    if texture_map.device != device:
                        texture_map = texture_map.to(device)

                textures = TexturesUV(
                    maps=texture_map[None],
                    faces_uvs=[faces_uv],
                    verts_uvs=[aux.verts_uvs]
                )
            else:
                # Fallback to vertex colors or white if no texture
                if 'verts_rgb' in aux:
                    verts_rgb = aux.verts_rgb[None]
                else:
                    verts_rgb = torch.ones_like(verts)[None]  # white color
                textures = TexturesVertex(verts_features=verts_rgb)

            # Create a Meshes object
            mesh = Meshes(
                verts=[verts],
                faces=[faces_idx],
                textures=textures
            )
            print("Successfully loaded mesh with materials manually")
    else:
        verts, faces, _ = load_obj(obj_path, device=device)
        faces_idx = faces.verts_idx
        mesh = Meshes(verts=[verts], faces=[faces_idx])

    return mesh


def load_tet_mesh(tet_path, device=None, dtype=None):
    """Load tet mesh from tet file"""
    tet_path = Path(tet_path)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tet_mesh = pv.read(tet_path)
    verts = np.asarray(tet_mesh.points)
    tets = np.asarray(tet_mesh.cell_connectivity).reshape(-1, 4).astype(np.int32)
    surf_tet_mesh = tet_mesh.extract_surface()
    surf_vinds = np.asarray(surf_tet_mesh.point_data["vtkOriginalPointIds"]).astype(
        np.int32
    )
    surf_face_surf_vinds = surf_tet_mesh.faces.reshape(-1, 4)[:, 1:]
    faces = surf_vinds[surf_face_surf_vinds]

    verts = torch.from_numpy(verts).clone().to(device, dtype=dtype)
    faces = torch.from_numpy(faces).clone().to(device, dtype=torch.int32)
    tets = torch.from_numpy(tets).clone().to(device, dtype=torch.int32)

    return verts, faces, tets

def load_tracking_points(tracking_points_path, image_size, device=None):
    """Load tracking points from a numpy file (by DELTA) and convert to tensor"""
    tracking_points = np.load(tracking_points_path, allow_pickle=True)
    print(tracking_points.keys())
    conf = tracking_points['conf']
    tracking_points = tracking_points['coords']
    # tracking_points = tracking_points / 960 # TODO: hardcoded
    # tracking_points = tracking_points * image_size
    tracking_points[:, :, :2] = (tracking_points[:, :, :2] + 1) / 2 * image_size
    tracking_points = torch.from_numpy(tracking_points).float().to(device)
    conf = torch.from_numpy(conf).float().to(device)
    return tracking_points, conf

def separate_mesh_by_vertex_colors(segmented_mesh, np_color_list, colors, device=None):
    """
    Separates a mesh into different meshes based on vertex colors.

    Args:
        segmented_mesh (Meshes): The segmented mesh with vertex textures
        colors: Tensor of vertex colors [1, N, 3]
        device: Torch device

    Returns:
        list: List of separated meshes for each color group
        dict: Mapping from color group index to representative name
        dict: Mapping from color group index to color value (RGB)
        dict: Mapping from color group to vertex indices in original mesh
    """
    if device is None:
        device = segmented_mesh.device
    
    # Get vertices and faces from the segmented mesh
    all_verts = segmented_mesh.verts_packed()
    all_faces = segmented_mesh.faces_packed()
    
    # Remove batch dimension
    vertex_colors = colors.squeeze(0)  # [N, 3]
    
    # Find unique colors (with some tolerance for floating point differences)
    # Convert to numpy for easier comparison
    colors_np = vertex_colors.detach().cpu().numpy()
    
    # Round colors to reduce floating-point variations
    rounded_colors = np.round(colors_np * 255) / 255
    
    # Find unique colors
    unique_colors = {}
    for i, color in enumerate(rounded_colors):
        color_tuple = tuple(color)
        if color_tuple not in unique_colors:
            unique_colors[color_tuple] = []
        unique_colors[color_tuple].append(i)

    color_groups = []
    color_to_verts = {}

    for idx, canonical_color in enumerate(np_color_list):
        
        if canonical_color in unique_colors:
            color_groups.append(canonical_color)

            indices = unique_colors[canonical_color]
            color_to_verts[idx] = torch.tensor(indices, device=device)

        else:
            color_groups.append(None)  # No such color in this image
            color_to_verts[idx] = torch.zeros(0, dtype=torch.long, device=device)  # empty tensor

    idx_to_color = {i: np.array(color_groups[i]) * 255 for i in range(len(color_groups))}
    idx_to_group_name = {i: f"color_{i}" for i in range(len(color_groups))}
    
    # Now create separate meshes for each color group
    separated_meshes = []
    color_to_vert_indices = {}  # Store the original vertex indices for each color group
    
    for color_idx in range(len(color_groups)):
        # Get vertices assigned to this color group
        vert_indices = color_to_verts[color_idx]
        
        # Save the original vertex indices for later use
        color_to_vert_indices[color_idx] = vert_indices
        
        # Create a mask for this color group
        region_mask = torch.zeros(len(all_verts), dtype=torch.bool, device=device)
        region_mask[vert_indices] = True
        
        # Create a mapping from old vertex indices to new ones
        old_to_new = torch.full((len(all_verts),), -1, dtype=torch.long, device=device)
        region_verts_idx = vert_indices
        old_to_new[region_verts_idx] = torch.arange(len(region_verts_idx), device=device)
        
        # Get vertices for this region
        region_verts = all_verts[region_mask]
        
        # Find faces that only use vertices from this region
        # A face belongs to the region if all its vertices are in the region
        face_vert_idx = all_faces.reshape(-1)  # Flatten to check all vertices in all faces
        face_region_mask = region_mask[face_vert_idx].reshape(-1, 3)  # Reshape back to (F, 3)
        valid_faces_mask = torch.all(face_region_mask, dim=1)
        region_faces = all_faces[valid_faces_mask]
        
        # Skip if no faces in this region
        if len(region_faces) == 0:
            separated_meshes.append(None)
            continue
        
        # Remap face indices to the new vertex indices
        region_faces = old_to_new[region_faces]
        
        # Get the color for this group
        color_np = color_groups[color_idx]
        region_color = torch.tensor(color_np, device=device)
        
        # Create vertex colors for this region
        region_texture = torch.ones_like(region_verts[:, :1]) * region_color.reshape(1, 3)
        region_textures = TexturesVertex(verts_features=region_texture.unsqueeze(0))
        
        # Create a mesh for this color group
        region_mesh = Meshes(
            verts=[region_verts],
            faces=[region_faces],
            textures=region_textures
        )
        
        separated_meshes.append(region_mesh)
    
    return separated_meshes, idx_to_group_name, idx_to_color, color_to_vert_indices