import bpy
import os
import sys


def export_obj(obj_path):
    bpy.ops.export_scene.obj(filepath=obj_path, use_selection=False)


def get_armature_and_mesh():
    armature = None
    mesh = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
        elif obj.type == 'MESH':
            mesh = obj
    return armature, mesh


def get_bone_world_position(armature_obj, bone):
    """Return the world space position of the bone head."""
    bone_matrix = armature_obj.matrix_world @ bone.matrix_local
    return bone_matrix.to_translation()


def write_rig_txt(rig_path, armature, mesh):
    arm_data = armature.data
    vg_map = {vg.index: vg.name for vg in mesh.vertex_groups}

    with open(rig_path, 'w') as f:
        # Write joints and root
        for bone in arm_data.bones:
            # pos = get_bone_world_position(armature, bone)
            pos = bone.matrix_local.to_translation()
            # f.write(
            #     f'joints {bone.name} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}\n')
            f.write(
                f'joints {bone.name} {pos.x:.8f} {pos.z:.8f} {-pos.y:.8f}\n')

        f.write(f'root {arm_data.bones[0].name}\n')

        # Write skin weights
        mesh_data = mesh.data
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = mesh.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()

        for i, v in enumerate(eval_mesh.vertices):
            weights = []
            for g in v.groups:
                name = vg_map.get(g.group)
                if name:
                    weights.append(name)
                    weights.append(f"{g.weight:.4f}")
            if weights:
                f.write("skin " + str(i) + " " + " ".join(weights) + "\n")

        eval_obj.to_mesh_clear()

        # Write hierarchy
        for bone in arm_data.bones:
            for child in bone.children:
                f.write(f'hier {bone.name} {child.name}\n')


def main():
    # Parse Blender command line arguments
    # Arguments after -- are passed to the script
    filepath = None
    
    if "--" in sys.argv:
        args =(sys.argv[sys.argv.index("--") + 1:])
        for i, arg in enumerate(args):
            if arg == "--filepath" and i + 1 < len(args):
                filepath = args[i + 1]
                break
    
    if not filepath:
        print("Error: --filepath argument is required")
        print("Usage: blender --background --python blender_read_fbx.py -- --filepath /path/to/file.fbx")
        return

    base_dir = os.path.dirname(filepath)
    file_name = os.path.basename(filepath).split(".")[0]
    model_id = file_name
    fbx_path = os.path.join(base_dir, f"{model_id}_skin.fbx")
    obj_path = os.path.join(base_dir, f"{model_id}_ori.obj")
    rig_path = os.path.join(base_dir, f"{model_id}_ori_rig.txt")
    
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Identify the armature and mesh
    armature, mesh = get_armature_and_mesh()
    if not armature or not mesh:
        raise RuntimeError(
            "Could not find both armature and mesh after import.")

    # Export geometry as .obj
    export_obj(obj_path)

    # Export rig to .txt
    write_rig_txt(rig_path, armature, mesh)

    print(f"Exported OBJ to {obj_path}")
    print(f"Exported RIG to {rig_path}")


if __name__ == "__main__":
    main()
