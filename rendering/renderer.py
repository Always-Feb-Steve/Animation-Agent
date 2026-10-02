import torch
import numpy as np
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    RasterizationSettings,
    SoftSilhouetteShader,
    MeshRendererWithFragments
)
from pytorch3d.renderer.mesh.shader import SoftDepthShader, HardDepthShader



class Renderer:
    def __init__(self, image_size=512, device=None):
        self.image_size = image_size
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def render(self, mesh, camera, lights, R=None, T=None, background_color=None, silhouette=False, depth=False):
        if background_color is None:
            background_color = torch.tensor([0.0, 0.0, 0.0])

        # Set up rasterization settings
        if silhouette:
            blend_params = BlendParams(
                sigma=1e-4, gamma=1e-4
            )
            raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
                faces_per_pixel=100,
            )
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=camera, raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params),
            )
        elif depth:
            raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=10,
            )
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=camera, raster_settings=raster_settings
                ),
                shader=HardDepthShader(
                    device=self.device,
                    cameras=camera
                ),
            )
        else:
            # Standard RGB rendering
            blend_params = BlendParams(
                sigma=1e-4, gamma=1e-4, background_color=background_color
            )
            raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=camera, raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=camera,
                    lights=lights,
                    blend_params=blend_params,
                ),
            )

        # Render the mesh
        if R is not None and T is not None:
            images = renderer(mesh, R=R, T=T)
        else:
            images = renderer(mesh)
            
        return images
    
    def pixel_to_triangle(self, pixel_coords, mesh, camera, R=None, T=None):
        """
        Find the triangle that each pixel belongs to and compute barycentric coordinates.
        
        Args:
            pixel_coords: Pixel coordinates (shape: (N, 2)) in image space [0, image_size]
            mesh: PyTorch3D mesh object
            camera: PyTorch3D camera object
            R: Optional rotation matrix
            T: Optional translation vector
            
        Returns:
            triangle_indices: Triangle index for each pixel (shape: (N,)), -1 if no triangle
            barycentric_coords: Barycentric coordinates for each pixel (shape: (N, 3))
        """
        # Set up rasterization settings for fragments
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
        )
        
        # Create rasterizer
        rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        )
        
        # Rasterize to get fragments
        if R is not None and T is not None:
            fragments = rasterizer(mesh, R=R, T=T)
        else:
            fragments = rasterizer(mesh)
        
        # Extract fragment information
        pix_to_face = fragments.pix_to_face  # (1, H, W, 1)
        barycentric_coords_full = fragments.bary_coords  # (1, H, W, 1, 3)
        
        # Convert pixel coordinates to integer indices
        # pixel_coords should be in [0, image_size] range
        pixel_coords = pixel_coords.round().long()
        
        # Clamp coordinates to valid range
        pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, self.image_size - 1)
        pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, self.image_size - 1)
        
        # Extract triangle indices and barycentric coordinates for specified pixels
        # Note: PyTorch3D uses (y, x) indexing for images
        y_coords = pixel_coords[:, 1]  # y coordinates
        x_coords = pixel_coords[:, 0]  # x coordinates
        
        # Get triangle indices for the specified pixels
        triangle_indices = pix_to_face[0, y_coords, x_coords, 0]  # (N,)
        
        # Get barycentric coordinates for the specified pixels
        barycentric_coords = barycentric_coords_full[0, y_coords, x_coords, 0, :]  # (N, 3)
        
        # Handle pixels that don't belong to any triangle (set to -1)
        # PyTorch3D uses -1 to indicate no face
        no_face_mask = triangle_indices == -1
        barycentric_coords[no_face_mask] = 0.0  # Set barycentric coords to 0 for invalid pixels
        
        return triangle_indices, barycentric_coords
    
    def triangle_to_pixel(self, triangle_indices, barycentric_coords, mesh, camera, R=None, T=None):
        """
        Convert triangle indices and barycentric coordinates back to pixel coordinates.
        
        Args:
            triangle_indices: Triangle indices (shape: (N,))
            barycentric_coords: Barycentric coordinates (shape: (N, 3))
            mesh: PyTorch3D mesh object
            camera: PyTorch3D camera object
            R: Optional rotation matrix
            T: Optional translation vector
            
        Returns:
            pixel_coords: Pixel coordinates and depth in image space (shape: (N, 3))
        """
        # Get mesh vertices and faces
        verts = mesh.verts_packed()  # (V, 3)
        faces = mesh.faces_packed()  # (F, 3)
        
        # Handle invalid triangle indices
        valid_mask = triangle_indices >= 0
        num_points = triangle_indices.shape[0]
        
        # Initialize output
        pixel_coords = torch.zeros((num_points, 3), device=self.device, dtype=torch.float32)
        
        if not valid_mask.any():
            # All invalid triangles, return zeros
            return pixel_coords
        
        # Get valid indices
        valid_triangle_indices = triangle_indices[valid_mask]
        valid_barycentric_coords = barycentric_coords[valid_mask]
        
        # Get the vertices of the triangles
        triangle_faces = faces[valid_triangle_indices]  # (N_valid, 3)
        triangle_verts = verts[triangle_faces]  # (N_valid, 3, 3)
        
        # Compute 3D world coordinates using barycentric coordinates
        # world_coords = sum(barycentric_coords * triangle_vertices)
        world_coords = torch.sum(
            valid_barycentric_coords.unsqueeze(2) * triangle_verts, dim=1
        )  # (N_valid, 3)
        
        # Apply transformation if provided
        if R is not None and T is not None:
            # Transform vertices: world_coords = R @ world_coords + T
            world_coords = torch.matmul(world_coords, R.transpose(-1, -2)) + T
        
        # Convert world coordinates to screen/pixel coordinates using camera
        # Add batch dimension for camera projection
        world_coords_batch = world_coords.unsqueeze(0)  # (1, N_valid, 3)
        
        # Project to screen space using camera
        screen_coords = camera.transform_points_screen(world_coords_batch, image_size=(self.image_size, self.image_size))  # (1, N_valid, 3)
        screen_coords = screen_coords.squeeze(0)  # (N_valid, 3)
        
        depth_coords = camera.get_world_to_view_transform().transform_points(world_coords_batch)
        depth_coords = depth_coords.squeeze(0)
        
        # Extract x, y pixel coordinates (z is depth)
        valid_pixel_coords = screen_coords[:, :2]  # (N_valid, 2)
        valid_depth_coords = depth_coords[:, 2]  # (N_valid, )
        
        # Store valid pixel coordinates
        pixel_coords[valid_mask] = torch.cat([valid_pixel_coords, valid_depth_coords.unsqueeze(1)], dim=1)
        
        return pixel_coords
    
    def check_triangle_visibility_v1(self, triangle_indices, pixel_coords, mesh, camera, R=None, T=None):
        """
        Check if triangles are visible (not occluded) at given pixel locations.
        
        Args:
            triangle_indices: Triangle indices to check (shape: (N,))
            pixel_coords: Pixel coordinates where to check visibility (shape: (N, 2)) in [0, image_size]
            mesh: PyTorch3D mesh object
            camera: PyTorch3D camera object
            R: Optional rotation matrix
            T: Optional translation vector
            
        Returns:
            visibility_mask: Boolean tensor indicating if each triangle is visible at its pixel location (shape: (N,))
        """
        # Set up rasterization settings for fragments
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
        )
        
        # Create rasterizer
        rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        )
        
        # Rasterize to get fragments
        if R is not None and T is not None:
            fragments = rasterizer(mesh, R=R, T=T)
        else:
            fragments = rasterizer(mesh)
        
        # Extract visible face indices at each pixel
        pix_to_face = fragments.pix_to_face  # (1, H, W, 1)
        
        # Convert pixel coordinates to integer indices
        pixel_coords_int = pixel_coords.round().long()
        
        # Clamp coordinates to valid range
        pixel_coords_int[:, 0] = torch.clamp(pixel_coords_int[:, 0], 0, self.image_size - 1)
        pixel_coords_int[:, 1] = torch.clamp(pixel_coords_int[:, 1], 0, self.image_size - 1)
        
        # Extract visible triangle at each query point's pixel location
        # Note: PyTorch3D uses (y, x) indexing for images
        y_coords = pixel_coords_int[:, 1]  # y coordinates
        x_coords = pixel_coords_int[:, 0]  # x coordinates
        
        visible_triangles = pix_to_face[0, y_coords, x_coords, 0]  # (N,)
        
        # Check if the query point's triangle matches the visible triangle
        # If they match, the triangle is visible (not occluded)
        visibility_mask = (triangle_indices == visible_triangles)
        
        return visibility_mask
    
    def check_triangle_visibility_v2(self, triangle_indices, mesh, camera, R=None, T=None):
        """
        Check if triangles are visible (not occluded) by first projecting their centroids,
        then checking visibility at those projected pixel locations.
        
        Args:
            triangle_indices: Triangle indices to check (shape: (N,))
            mesh: PyTorch3D mesh object
            camera: PyTorch3D camera object
            R: Optional rotation matrix
            T: Optional translation vector
            
        Returns:
            visibility_mask: Boolean tensor indicating if each triangle is visible (shape: (N,))
        """
        # Get triangle vertices
        faces = mesh.faces_packed()  # (F, 3)
        verts = mesh.verts_packed()  # (V, 3)
        
        # Get vertices for the specified triangles
        triangle_faces = faces[triangle_indices]  # (N, 3)
        triangle_verts = verts[triangle_faces]  # (N, 3, 3)
        
        # Compute triangle centroids
        triangle_centroids = triangle_verts.mean(dim=1)  # (N, 3)
        
        # Apply transformation if provided
        if R is not None and T is not None:
            triangle_centroids = torch.matmul(triangle_centroids, R.transpose(-1, -2)) + T
        
        # Project centroids to screen space
        triangle_centroids_batch = triangle_centroids.unsqueeze(0)  # (1, N, 3)
        screen_coords = camera.transform_points_screen(
            triangle_centroids_batch, 
            image_size=(self.image_size, self.image_size)
        )  # (1, N, 3)
        screen_coords = screen_coords.squeeze(0)  # (N, 3)
        
        # Extract pixel coordinates
        pixel_coords = screen_coords[:, :2]  # (N, 2)
        
        # Set up rasterization settings
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
        )
        
        # Create rasterizer and rasterize mesh
        rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        )
        
        if R is not None and T is not None:
            fragments = rasterizer(mesh, R=R, T=T)
        else:
            fragments = rasterizer(mesh)
        
        # Extract visible face indices at each pixel
        pix_to_face = fragments.pix_to_face  # (1, H, W, 1)
        
        # Convert pixel coordinates to integer indices
        pixel_coords_int = pixel_coords.round().long()
        
        # Clamp coordinates to valid range
        pixel_coords_int[:, 0] = torch.clamp(pixel_coords_int[:, 0], 0, self.image_size - 1)
        pixel_coords_int[:, 1] = torch.clamp(pixel_coords_int[:, 1], 0, self.image_size - 1)
        
        # Extract visible triangle at each query point's pixel location
        # Note: PyTorch3D uses (y, x) indexing for images
        y_coords = pixel_coords_int[:, 1]  # y coordinates
        x_coords = pixel_coords_int[:, 0]  # x coordinates
        
        visible_triangles = pix_to_face[0, y_coords, x_coords, 0]  # (N,)
        
        # Check if there's a valid triangle visible at each pixel location
        # visible_triangles == -1 indicates background (no triangle visible)
        visibility_mask = (visible_triangles == triangle_indices)
        
        return visibility_mask
        