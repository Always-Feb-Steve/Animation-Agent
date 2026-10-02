import torch
import os

def parse_rig_file(filepath):
    """
    Parse a rigging file containing joint positions, skin weights, and joint hierarchy.
    
    Args:
        filepath (str): Path to the rigging file
        
    Returns:
        tuple: (joints, skin_weights, hierarchy)
            - joints: dict of joint name to position tensor
            - skin_weights: dict of vertex index to list of (joint_name, weight) tuples
            - hierarchy: dict of parent joint to list of child joints
    """
    joints = {}
    skin_weights = {}
    hierarchy = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("joints"):
            # Parse joint positions
            _, name, x, y, z = line.split()
            joints[name] = torch.tensor([float(x), float(y), float(z)])
            
        elif line.startswith("skin"):
            # Parse skin weights
            parts = line.split()
            idx = int(parts[1])
            weights = [(parts[i], float(parts[i+1])) for i in range(2, len(parts), 2)]
            skin_weights[idx] = weights
            
        elif line.startswith("hier"):
            # Parse hierarchical relationships
            _, parent, child = line.split()
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(child)

    return joints, skin_weights, hierarchy