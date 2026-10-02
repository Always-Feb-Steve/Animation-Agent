import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)


class RiggingModel(nn.Module):
    def __init__(self, mesh, joints, skin_weights, hierarchy):
        super(RiggingModel, self).__init__()
        self.register_buffer(
            "_device_dummy", torch.zeros(1)
        )  # Helper for device tracking

        # Store mesh
        self.mesh = mesh

        # Convert joint positions to tensors and register as buffers
        self.joints = {}
        for name, pos in joints.items():
            if not torch.is_tensor(pos):
                pos = torch.tensor(pos, dtype=torch.float32)
            self.register_buffer(f"joint_pos_{name}", pos)
            self.joints[name] = pos

        self.skin_weights = (
            skin_weights  # This is a dict of vertex_idx -> [(joint_name, weight)]
        )
        self.hierarchy = hierarchy  # This is a dict of parent -> [children]

        # Find root joints (joints with no parents)
        self.root_joints = set(joints.keys())
        for children in hierarchy.values():
            self.root_joints -= set(children)

        # Create trainable parameters for each joint's rotation using 6D representation
        # and additional translation parameters for non-root joints
        self.joint_rotations = nn.ParameterDict()
        self.joint_translations = nn.ParameterDict()  # New parameter dict for translations
        
        for joint_name in joints.keys():
            # print(joint_name)
            if joint_name not in self.root_joints:
                # Initialize 6D rotation parameters (two orthogonal 3D vectors)
                # Default to identity rotation: [1,0,0, 0,1,0]
                self.joint_rotations[joint_name] = nn.Parameter(
                    torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
                )
                # Initialize translation parameters to zeros (no additional translation)
                self.joint_translations[joint_name] = nn.Parameter(
                    torch.zeros(3, dtype=torch.float32)
                )

        # Create trainable parameters for root transformation
        # [tx, ty, tz, r1, r2, r3, r4, r5, r6] - translation and 6D rotation
        for root in self.root_joints:
            self.register_parameter(
                f"root_transform_{root}",
                nn.Parameter(
                    torch.cat(
                        [
                            torch.zeros(3, dtype=torch.float32),
                            torch.tensor(
                                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32
                            ),
                        ]
                    )
                ),
            )

        # Pre-compute homogeneous vertices
        verts = self.mesh.verts_packed()
        V = verts.shape[0]
        verts_h = torch.cat(
            [verts, torch.ones(V, 1, dtype=verts.dtype, device=verts.device)], dim=1
        )
        self.register_buffer("verts_h", verts_h)

        # Pre-compute weight matrix and joint mapping
        all_joint_names = list(joints.keys())
        J = len(all_joint_names)
        self.register_buffer(
            "joint_names", torch.tensor([hash(name) for name in all_joint_names])
        )  # Store for validation
        self.joint_to_idx = {name: idx for idx, name in enumerate(all_joint_names)}

        # Create and fill weight matrix (V, J)
        weights = torch.zeros((V, J), dtype=torch.float32)
        for vi in range(V):
            if vi in self.skin_weights:
                for joint_name, weight in self.skin_weights[vi]:
                    if joint_name in self.joint_to_idx:
                        weights[vi, self.joint_to_idx[joint_name]] = weight
        self.register_buffer("weights", weights)

        # Store mask for unweighted vertices
        self.register_buffer("unweighted_mask", weights.sum(dim=1) < 1e-6)

    @property
    def device(self):
        """Get the current device of the model"""
        return self._device_dummy.device

    def to(self, *args, **kwargs):
        """Override to() to handle custom attributes"""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        # Call parent's to() first
        super().to(*args, **kwargs)

        if device is not None:
            # Move mesh to device if it has a to() method
            if hasattr(self.mesh, "to"):
                self.mesh = self.mesh.to(device)

            # Update joint positions from registered buffers
            for name in self.joints.keys():
                self.joints[name] = getattr(self, f"joint_pos_{name}")

        return self

    def _compute_rotation_matrix_from_6d(self, rot6d):
        """
        Convert 6D rotation representation to 4x4 rotation matrix.
        Uses PyTorch3D's rotation_6d_to_matrix function.
        """
        # Convert to 3x3 rotation matrix
        R_3x3 = rotation_6d_to_matrix(rot6d)

        # Create 4x4 rotation matrix
        R = torch.eye(4, device=self.device, dtype=R_3x3.dtype)
        R[:3, :3] = R_3x3

        return R

    def compute_transform_from_parent(self, joint_pos_child, joint_pos_parent, rot6d, translation=None):
        """
        Compute transform that rotates around the joint's position relative to parent and applies additional translation.
        
        Args:
            joint_pos_child: Position of child joint
            joint_pos_parent: Position of parent joint
            rot6d: 6D rotation parameter
            translation: Optional 3D translation parameter
        """
        # Create rotation matrix from 6D representation
        R = self._compute_rotation_matrix_from_6d(rot6d)

        # Create translation matrices
        T = torch.eye(4, device=self.device, dtype=R.dtype)
        # T[:3, 3] = joint_pos_parent
        T[:3, 3] = joint_pos_child
        
        T_additional = torch.eye(4, device=self.device, dtype=R.dtype)
        T_additional[:3, 3] = translation

        T_inv = torch.eye(4, device=self.device, dtype=R.dtype)
        # T_inv[:3, 3] = -joint_pos_parent
        T_inv[:3, 3] = -joint_pos_child
        
        # Transform = T * R * T^(-1)    
        return T @ T_additional @ R @ T_inv

    def create_root_transform(self, root_params, root_position):
        """Create 4x4 transformation matrix from root parameters [tx, ty, tz, r1, r2, r3, r4, r5, r6]"""
        translation = root_params[:3]
        rot6d = root_params[3:]

        # Create rotation matrix from 6D representation
        R = self._compute_rotation_matrix_from_6d(rot6d)

        # Add translation
        T = torch.eye(4, device=self.device, dtype=R.dtype)
        T[:3, 3] = translation
        
        # Root position
        T0 = torch.eye(4, device=self.device, dtype=R.dtype)
        T0[:3, 3] = root_position
        
        T0_inv = torch.eye(4, device=self.device, dtype=R.dtype)
        T0_inv[:3, 3] = -root_position
        
        return T @ T0 @ R @ T0_inv

    def create_joint_transforms(self):
        """Create joint transforms using trainable parameters"""
        transforms = {}

        def process_joint(joint_name, parent_transform=None):
            """Recursively process joints following the hierarchy"""
            current_joint_pos = self.joints[joint_name]

            # For root joints, apply the trainable root transformation
            if joint_name in self.root_joints:
                root_params = getattr(self, f"root_transform_{joint_name}")
                local_transform = self.create_root_transform(root_params, current_joint_pos)
            else:
                # Get parent joint position
                parent_name = None
                for p, children in self.hierarchy.items():
                    if joint_name in children:
                        parent_name = p
                        break
                parent_pos = self.joints[parent_name]

                # Compute local transform using trainable rotation and translation
                local_transform = self.compute_transform_from_parent(
                    current_joint_pos,
                    parent_pos,
                    self.joint_rotations[joint_name],  # 6D rotation for non-root joints
                    self.joint_translations[joint_name],  # Additional translation
                )

            # If this joint has a parent, combine with parent's transform
            if parent_transform is not None:
                transforms[joint_name] = parent_transform @ local_transform
            else:
                transforms[joint_name] = local_transform

            # Process children
            if joint_name in self.hierarchy:
                for child in self.hierarchy[joint_name]:
                    process_joint(child, transforms[joint_name])

        # Process from root joints
        for root in self.root_joints:
            process_joint(root)

        return transforms

    def apply_lbs(self, joint_transforms):
        """
        Apply linear blend skinning using pre-computed matrices.

        Args:
            joint_transforms: Dictionary mapping joint names to their 4x4 transformation matrices

        Returns:
            Deformed mesh with transformed vertices
        """
        # Validate joint transforms match our pre-computed weights
        joint_names = list(joint_transforms.keys())
        assert len(joint_names) == len(
            self.joint_to_idx
        ), "Number of joints doesn't match"

        # Stack transforms in the same order as our weight matrix
        transforms = torch.stack(
            [joint_transforms[name] for name in self.joint_to_idx.keys()]
        )

        # Apply transforms to vertices using pre-computed weights and homogeneous coordinates
        transformed_verts = torch.einsum(
            "vj,jmn,vn->vm", self.weights, transforms, self.verts_h
        )
        new_verts = transformed_verts[:, :3]

        # Handle unweighted vertices using pre-computed mask
        new_verts[self.unweighted_mask] = self.verts_h[self.unweighted_mask, :3]

        return self.mesh.update_padded(new_verts.unsqueeze(0))

    def get_joint_positions(self):
        """
        Get the positions of all joints in world space after applying transforms

        Returns:
            dict: Dictionary mapping joint names to their world space positions (3D vectors)
        """
        joint_transforms = self.create_joint_transforms()
        positions = {}

        # For each joint, extract position from its transform matrix
        for joint_name, transform in joint_transforms.items():
            rest_pos = self.joints[joint_name]
            transformed_pos = transform @ torch.cat(
                [rest_pos, torch.ones(1, device=self.device)]
            )
            positions[joint_name] = transformed_pos[:3]

        return positions

    def get_joint_rotations(self):
        """
        Get the rotations of all joints in axis-angle representation.

        Returns:
            Dictionary mapping joint names to their axis-angle representations
        """
        rotations = {}
        for joint_name, rot6d in self.joint_rotations.items():
            rotations[joint_name] = matrix_to_axis_angle(
                rotation_6d_to_matrix(rot6d.unsqueeze(0)), fast=False
            ).squeeze(0)
        return rotations
    
    def get_joint_translations(self):
        """
        Get the additional translations of all non-root joints.
        
        Returns:
            Dictionary mapping joint names to their translation vectors (3D)
        """
        return {name: trans.clone() for name, trans in self.joint_translations.items()}

    def get_root_rotations(self):
        """
        Get the rotations of root joints in axis-angle representation.

        Returns:
            Dictionary mapping root joint names to their axis-angle representations
        """
        rotations = {}
        for root in self.root_joints:
            root_params = getattr(self, f"root_transform_{root}")
            # Extract rotation part (last 6 elements)
            rot6d = root_params[3:]
            rotations[root] = matrix_to_axis_angle(
                rotation_6d_to_matrix(rot6d.unsqueeze(0)), fast=False
            ).squeeze(0)
        return rotations

    def get_root_positions(self):
        """
        Get the positions of root joints in world space after applying transforms.
        
        Returns:
            dict: Dictionary mapping root joint names to their world space positions (3D vectors)
        """
        joint_transforms = self.create_joint_transforms()
        positions = {}
        
        # For each root joint, extract position from its transform matrix
        for root in self.root_joints:
            transform = joint_transforms[root]
            rest_pos = self.joints[root]
            transformed_pos = transform @ torch.cat(
                [rest_pos, torch.ones(1, device=self.device)]
            )
            positions[root] = transformed_pos[:3]
            
        return positions

    def forward(self):
        """Forward pass: compute transforms and apply skinning"""
        joint_transforms = self.create_joint_transforms()
        deformed_mesh = self.apply_lbs(joint_transforms)
        return deformed_mesh


class RiggingModelSequence(RiggingModel):
    def __init__(self, mesh, joints, skin_weights, hierarchy, sequence_length):
        """
        Initialize a sequence-based rigging model that extends RiggingModel.

        Args:
            mesh: The input mesh to deform
            joints: Dictionary of joint positions
            skin_weights: Dictionary of vertex skinning weights
            hierarchy: Dictionary of joint parent-child relationships
            sequence_length: Number of frames in the animation sequence
        """
        super().__init__(mesh, joints, skin_weights, hierarchy)

        self.sequence_length = sequence_length

        # Convert single-frame parameters to sequence parameters
        with torch.no_grad():
            # Convert joint rotations to sequence
            new_joint_rotations = nn.ParameterDict()
            for joint_name, rot6d in self.joint_rotations.items():
                # Initialize sequence with identity rotation
                sequence_rot = rot6d.expand(
                    sequence_length, -1
                ).clone()  # (sequence_length, 6)
                new_joint_rotations[joint_name] = nn.Parameter(sequence_rot)
            self.joint_rotations = new_joint_rotations
            
            # Convert joint translations to sequence
            new_joint_translations = nn.ParameterDict()
            for joint_name, trans in self.joint_translations.items():
                # Initialize sequence with zero translations
                sequence_trans = trans.expand(
                    sequence_length, -1
                ).clone()  # (sequence_length, 3)
                new_joint_translations[joint_name] = nn.Parameter(sequence_trans)
            self.joint_translations = new_joint_translations

            # Convert root transforms to sequence
            for root in self.root_joints:
                root_param = getattr(self, f"root_transform_{root}")
                # Initialize sequence with identity transform
                sequence_transform = root_param.expand(
                    sequence_length, -1
                ).clone()  # (sequence_length, 9)
                self.register_parameter(
                    f"root_transform_{root}", nn.Parameter(sequence_transform)
                )

    def load_from_model_list(self, rigging_model_list):
        """
        Load parameters from a list of single-frame RiggingModel instances.

        Args:
            rigging_model_list: List of RiggingModel instances, one for each frame
                              Length must match sequence_length
        """
        # The length of rigging_model_list should be the same as the sequence_length
        assert (
            len(rigging_model_list) == self.sequence_length
        ), f"The length of rigging_model_list should be the same as the sequence_length, but got {len(rigging_model_list)}"
        state_dict = self.state_dict()
        input_state_dict_list = [
            rigging_model.state_dict() for rigging_model in rigging_model_list
        ]

        for k, v in state_dict.items():
            if k.startswith("joint_rotations") or k.startswith("joint_translations") or k.startswith("root_transform"):
                new_v = torch.zeros_like(v, device=self.device)
                for t in range(self.sequence_length):
                    new_v[t] = input_state_dict_list[t][k].to(self.device)
                state_dict[k] = new_v

        self.load_state_dict(state_dict, strict=True)

    def create_joint_transforms(self, start_idx=None, end_idx=None):
        """
        Create joint transforms for a range of frames.

        Args:
            start_idx: Starting frame index (inclusive)
            end_idx: Ending frame index (exclusive)

        Returns:
            Dictionary mapping joint names to their transforms of shape (T, 4, 4)
        """
        # Handle default range
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = self.sequence_length

        # Validate indices
        assert (
            0 <= start_idx < end_idx <= self.sequence_length
        ), f"Invalid frame range: {start_idx} to {end_idx}, sequence length is {self.sequence_length}"

        # Create transforms for specified frames
        all_transforms = {}
        for t in range(start_idx, end_idx):  # end_idx is now exclusive
            frame_transforms = self._create_joint_transforms_single_frame(t)
            for joint_name, transform in frame_transforms.items():
                if joint_name not in all_transforms:
                    all_transforms[joint_name] = []
                all_transforms[joint_name].append(transform)

        # Stack transforms along time dimension
        return {
            name: torch.stack(transforms) for name, transforms in all_transforms.items()
        }

    def _create_joint_transforms_single_frame(self, frame_idx):
        """Create joint transforms for a single frame"""
        transforms = {}

        def process_joint(joint_name, parent_transform=None):
            current_joint_pos = self.joints[joint_name]

            if joint_name in self.root_joints:
                root_params = getattr(self, f"root_transform_{joint_name}")[frame_idx]
                local_transform = self.create_root_transform(root_params, current_joint_pos)
            else:
                parent_name = None
                for p, children in self.hierarchy.items():
                    if joint_name in children:
                        parent_name = p
                        break
                parent_pos = self.joints[parent_name]

                local_transform = self.compute_transform_from_parent(
                    current_joint_pos,
                    parent_pos,
                    self.joint_rotations[joint_name][frame_idx],
                    self.joint_translations[joint_name][frame_idx],  # Frame-specific translation
                )

            if parent_transform is not None:
                transforms[joint_name] = parent_transform @ local_transform
            else:
                transforms[joint_name] = local_transform
                # print(frame_idx, joint_name, local_transform)

            if joint_name in self.hierarchy:
                for child in self.hierarchy[joint_name]:
                    process_joint(child, transforms[joint_name])

        for root in self.root_joints:
            process_joint(root)

        return transforms

    def apply_lbs(self, joint_transforms):
        """
        Apply linear blend skinning using pre-computed matrices.
        Processes frames one by one to save memory.

        Args:
            joint_transforms: Dictionary mapping joint names to their transforms
                            Shape is (T, 4, 4) for sequence

        Returns:
            Sequence of deformed meshes
        """
        # Get sequence length from transforms
        T = next(iter(joint_transforms.values())).shape[0]
        deformed_verts_list = []

        for t in range(T):
            # Extract transforms for current frame
            transforms_t = torch.stack(
                [joint_transforms[name][t] for name in self.joint_to_idx.keys()]
            )  # Shape: (J, 4, 4)

            # Apply LBS for current frame
            transformed_verts = torch.einsum(
                "vj,jmn,vn->vm", self.weights, transforms_t, self.verts_h
            )  # Shape: (V, 4)

            new_verts = transformed_verts[:, :3]  # Shape: (V, 3)

            # Handle unweighted vertices
            new_verts[self.unweighted_mask] = self.verts_h[self.unweighted_mask, :3]

            deformed_verts_list.append(new_verts)

        # Stack all frames and add batch dimension
        all_deformed_verts = torch.stack(deformed_verts_list)  # Shape: (T, V, 3)

        # Create a list of meshes, one for each frame
        mesh_list = []
        for t in range(T):
            # Create a new mesh for each frame with the same properties as the original
            # new_mesh = self.mesh.clone()
            # # Update vertices for this frame
            # new_mesh.verts_list()[0] = all_deformed_verts[t]
            # mesh_list.append(new_mesh)
            new_mesh = Meshes(
                verts=[all_deformed_verts[t]],
                faces=[self.mesh.faces_list()[0]],
                textures=self.mesh.textures if hasattr(self.mesh, "textures") else None,
            )
            mesh_list.append(new_mesh)

        # Return list of meshes
        return mesh_list

    def forward(self, start_idx=None, end_idx=None):
        """
        Forward pass: compute transforms and apply skinning for a range of frames.

        Args:
            start_idx: Optional starting frame index (inclusive). If None, start from 0.
            end_idx: Optional ending frame index (exclusive). If None, end at sequence_length.

        Returns:
            List of deformed meshes from start_idx to end_idx (exclusive)
        """
        joint_transforms = self.create_joint_transforms(start_idx, end_idx)
        deformed_meshes = self.apply_lbs(joint_transforms)
        return deformed_meshes

    def get_joint_positions(self, start_idx=None, end_idx=None):
        """
        Get joint positions for a range of frames.

        Args:
            start_idx: Optional starting frame index (inclusive). If None, start from 0.
            end_idx: Optional ending frame index (exclusive). If None, end at sequence_length.

        Returns:
            Dictionary mapping joint names to their positions of shape (T, 3)
        """
        joint_transforms = self.create_joint_transforms(start_idx, end_idx)
        positions = {}

        for joint_name, transform_sequence in joint_transforms.items():
            rest_pos = self.joints[joint_name]
            rest_pos_h = torch.cat([rest_pos, torch.ones(1, device=self.device)])
            transformed_pos = transform_sequence @ rest_pos_h
            positions[joint_name] = transformed_pos[..., :3]

        return positions
    
    def get_root_positions(self, start_idx=None, end_idx=None):
        """
        Get root positions for a range of frames.

        Args:
            start_idx: Optional starting frame index (inclusive). If None, start from 0.
            end_idx: Optional ending frame index (exclusive). If None, end at sequence_length.

        Returns:
            Dictionary mapping root joint names to their positions of shape (T, 3)
            Each position is extracted from the final transformation matrix
        """
        # Handle default range
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = self.sequence_length

        # Validate indices
        assert (
            0 <= start_idx < end_idx <= self.sequence_length
        ), f"Invalid frame range: {start_idx} to {end_idx}, sequence length is {self.sequence_length}"

        # Get joint transforms for the specified range
        joint_transforms = self.create_joint_transforms(start_idx, end_idx)
        positions = {}

        # Extract positions from the transformation matrices for root joints
        for root in self.root_joints:
            transform_sequence = joint_transforms[root]  # Shape: (T, 4, 4)
            rest_pos = self.joints[root]
            rest_pos_h = torch.cat([rest_pos, torch.ones(1, device=self.device)])
            
            # Apply transforms to rest position
            transformed_pos = transform_sequence @ rest_pos_h
            positions[root] = transformed_pos[..., :3]

        return positions

    def get_joint_rotations(self, frame_idx=None):
        """
        Get the rotations of all joints in axis-angle representation.

        Args:
            frame_idx: Optional frame index. If None, return all frames.

        Returns:
            Dictionary mapping joint names to their axis-angle representations
            If frame_idx is None: angles have shape (T, 3)
            If frame_idx is specified: angles have shape (3,)
        """
        rotations = {}
        for joint_name, rot6d in self.joint_rotations.items():
            if frame_idx is not None:
                rotations[joint_name] = matrix_to_axis_angle(
                    rotation_6d_to_matrix(rot6d[frame_idx].unsqueeze(0)), fast=False
                ).squeeze(0)
            else:
                rotations[joint_name] = matrix_to_axis_angle(
                    rotation_6d_to_matrix(rot6d), fast=False
                )
        return rotations
    
    def get_joint_translations(self, frame_idx=None):
        """
        Get the additional translations of all non-root joints.
        
        Args:
            frame_idx: Optional frame index. If None, return all frames.
            
        Returns:
            Dictionary mapping joint names to their translation vectors
            If frame_idx is None: translations have shape (T, 3)
            If frame_idx is specified: translations have shape (3,)
        """
        translations = {}
        for joint_name, trans in self.joint_translations.items():
            if frame_idx is not None:
                translations[joint_name] = trans[frame_idx].clone()
            else:
                translations[joint_name] = trans.clone()
        return translations
    
    def get_root_rotations(self, frame_idx=None):
        """
        Get the rotations of root joints in axis-angle representation.

        Args:
            frame_idx: Optional frame index. If None, return all frames.

        Returns:
            Dictionary mapping root joint names to their axis-angle representations
            If frame_idx is None: angles have shape (T, 3)
            If frame_idx is specified: angles have shape (3,)
        """
        rotations = {}
        for root in self.root_joints:
            root_params = getattr(self, f"root_transform_{root}")
            
            if frame_idx is not None:
                # Extract rotation part (last 6 elements) for a specific frame
                rot6d = root_params[frame_idx, 3:]
                rotations[root] = matrix_to_axis_angle(
                    rotation_6d_to_matrix(rot6d.unsqueeze(0)), fast=False
                ).squeeze(0)
            else:
                # Extract rotation part (last 6 elements) for all frames
                rot6d = root_params[:, 3:]
                rotations[root] = matrix_to_axis_angle(
                    rotation_6d_to_matrix(rot6d), fast=False
                )
                
        return rotations
    
    def get_joint_transforms(self, frame_idx=None):
        """
        Get the transformation matrices of all joints.
        
        Args:
            frame_idx: Optional frame index. If None, return transforms for all frames.
            
        Returns:
            Dictionary mapping joint names to their 4x4 transformation matrices.
            If frame_idx is None: transforms have shape (T, 4, 4)
            If frame_idx is specified: transforms have shape (4, 4)
        """
        if frame_idx is not None:
            # For a specific frame, call the single frame version
            joint_transforms = self._create_joint_transforms_single_frame(frame_idx)
            return joint_transforms
        else:
            # For all frames, use the sequence version
            joint_transforms = self.create_joint_transforms()
            return joint_transforms
