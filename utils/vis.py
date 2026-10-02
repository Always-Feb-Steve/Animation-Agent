import torch
import cv2
import numpy as np
from rendering.camera import world_to_image_space
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


def vis_skeleton(
    images,
    joint_positions_world,
    hierarchy,
    camera,
    joint_color=(0, 0, 215),
    joint_thickness=2,
    joint_radius=3,
    draw_dots=False,
    dot_color=(0, 255, 0),
    blend_factor=None,
):
    """
    Visualize a sequence of skeletons on a sequence of images.

    Args:
        images: list or tensor of images [T, H, W, 3] (numpy arrays in BGR format)
        joint_positions_world: dictionary of joint positions in world space
                              - For sequences: each value is [T, 3] tensor
                              - For single frame: each value is [3] tensor
        hierarchy: dictionary of parent-child relationships
        camera: camera object used for rendering
        joint_color: BGR color tuple for skeleton lines
        joint_thickness: thickness of skeleton lines
        joint_radius: radius of joint dots
        draw_dots: whether to draw dots at joint positions
        dot_color: BGR color tuple for joint dots
        blend_factor: if not None, blend the images with white background using this factor (0.0-1.0)
                      where 1.0 means use original image, 0.0 means use white background

    Returns:
        List of images with skeletons drawn on them
    """
    # Make a copy of the images to avoid modifying the originals
    result_images = [
        img.copy() if isinstance(img, np.ndarray) else img.clone() for img in images
    ]
    num_frames = len(images)
    image_size = images[0].shape[0]  # Assuming square images
    
    # Apply blending with white background if requested
    if blend_factor is not None:
        for t in range(num_frames):
            white_image = np.ones_like(result_images[t]) * 255
            result_images[t] = cv2.addWeighted(
                result_images[t], blend_factor, white_image, 1.0 - blend_factor, 0
            )

    # Check if we have a sequence or single frame of joint positions
    is_sequence = (
        len(joint_positions_world[list(joint_positions_world.keys())[0]].shape) > 1
    )

    for t in range(num_frames):
        # Get joint positions for this frame
        if is_sequence:
            # Extract frame t from each joint's position sequence
            frame_joint_positions = {
                joint: pos[t] for joint, pos in joint_positions_world.items()
            }
        else:
            # Use the same joint positions for all frames
            frame_joint_positions = joint_positions_world

        # Stack joint positions for conversion to image space
        joint_names = list(frame_joint_positions.keys())
        joint_positions_stack = torch.stack(
            [frame_joint_positions[joint] for joint in joint_names], dim=0
        )

        # Convert world space positions to image space
        joint_positions_image = world_to_image_space(camera, joint_positions_stack)

        # Create a dictionary mapping joint names to their pixel coordinates
        joint_name_to_pixel = {}
        for i, joint_name in enumerate(joint_names):
            x, y = int(joint_positions_image[i, 0] * image_size), int(
                joint_positions_image[i, 1] * image_size
            )
            joint_name_to_pixel[joint_name] = (x, y)

        # Draw lines between parent and child joints based on hierarchy
        for parent_joint, child_joints in hierarchy.items():
            if parent_joint in joint_name_to_pixel:
                parent_pos = joint_name_to_pixel[parent_joint]
                for child_joint in child_joints:
                    if child_joint in joint_name_to_pixel:
                        child_pos = joint_name_to_pixel[child_joint]
                        cv2.line(
                            result_images[t],
                            parent_pos,
                            child_pos,
                            joint_color,
                            joint_thickness,
                        )

        # Optionally draw dots at joint positions for clarity
        if draw_dots:
            for joint_name, pos in joint_name_to_pixel.items():
                if type(dot_color) == tuple:
                    cv2.circle(result_images[t], pos, joint_radius, dot_color, -1)
                else:
                    # print(dot_color[joint_name])
                    r, g, b = dot_color[joint_name]
                    r, g, b = int(r), int(g), int(b)
                    cv2.circle(result_images[t], pos, joint_radius, (b, g, r), -1)

    return result_images


def vis_skinning_weights(
    mesh,
    joints,
    skin_weights, 
    joint_colors,
    renderer, 
    camera, 
    lights, 
    background_color=None, 
    mode='max_joint',
    image_size=512,
):
    """
    Visualize the skinning weights of a mesh.

    Args:
        mesh: The input mesh (Meshes object from PyTorch3D)
        skin_weights: Dictionary mapping vertex indices to [(joint_name, weight)] pairs
        renderer: Renderer object for rendering the mesh
        camera: Camera object for rendering
        lights: Lights object for rendering
        background_color: Background color for rendering (default: black)
        mode: One of ['max_joint', 'blend']. 
              'max_joint' colors each vertex by the color of its max-weight joint
              'blend' blends colors based on joint weights
        joint_colors: Dictionary mapping joint names to colors (RGB tensors)
              If None, random colors will be generated
        image_size: Output image size

    Returns:
        Rendered image tensor [1, H, W, 3]
    """

    device = mesh.device
    
    # Set background color if not provided
    if background_color is None:
        background_color = torch.tensor([0.0, 0.0, 0.0], device=device)
    
    # Generate colors for each vertex based on skinning weights
    num_verts = mesh.verts_packed().shape[0]
    colors = []
    
    for i in range(num_verts):
        if i in skin_weights:
            if mode == 'max_joint':
                # Assign color based on max weight joint
                max_weight = 0
                max_joint = None
                for joint, weight in skin_weights[i]:
                    if weight > max_weight:
                        max_weight = weight
                        max_joint = joint
                
                # Assign color
                if max_joint and max_joint in joint_colors:
                    color = joint_colors[max_joint]
                else:
                    color = torch.zeros(3, device=device)
            
            elif mode == 'blend':
                # Blend colors based on weights
                color = torch.zeros(3, device=device)
                total_weight = 0
                
                for joint, weight in skin_weights[i]:
                    if joint in joint_colors:
                        color += weight * joint_colors[joint]
                        total_weight += weight
                
                # Normalize
                if total_weight > 0:
                    color /= total_weight
            else:
                color = torch.zeros(3, device=device)
        else:
            # No skinning weights for this vertex
            color = torch.zeros(3, device=device)
        
        colors.append(color)
    
    # Stack colors and create textures
    colors = torch.stack(colors)[None].to(device)  # [1, N, 3]
    textures = TexturesVertex(verts_features=colors)
    
    # Create new mesh with color textures
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    colored_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    
    # Render the mesh
    with torch.no_grad():
        rendered_image = renderer.render(
            colored_mesh,
            camera,
            lights,
            background_color=background_color
        )
    
    return rendered_image
