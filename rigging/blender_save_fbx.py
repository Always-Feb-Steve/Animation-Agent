import bpy
import os


def load_rig_info(info_path):
    joint_pos = {}
    joint_hier = {}
    joint_skin = []
    root_name = None

    with open(info_path, 'r') as f:
        for line in f:
            word = line.strip().split()
            if not word:
                continue
            if word[0] == 'joints':
                joint_pos[word[1]] = [
                    float(word[2]),
                    float(word[3]),
                    float(word[4])
                ]
            elif word[0] == 'root':
                root_name = word[1]
            elif word[0] == 'hier':
                if word[1] not in joint_hier:
                    joint_hier[word[1]] = [word[2]]
                else:
                    joint_hier[word[1]].append(word[2])
            elif word[0] == 'skin':
                joint_skin.append(word[1:])

    return joint_pos, joint_hier, joint_skin, root_name


def build_armature(joint_pos, joint_hier, root_name):
    # Create a new armature object in the scene
    bpy.ops.object.add(type='ARMATURE')
    armature = bpy.context.object
    armature.parent = None
    # armature.scale = (0.01, 0.01, 0.01)

    # Make sure you're in edit mode to start adding bones
    bpy.ops.object.mode_set(mode='EDIT')

    bones = {}

    def create_bone(name, parent_name=None):
        if name.endswith('_end') or name in bones:
            return  # Skip _end bones or already created bones

        # Create the bone
        bone = armature.data.edit_bones.new(name)
        # bone.head = joint_pos[name]
        # Convert to Blender's coordinate system
        pos = joint_pos[name]
        bone.head = (pos[0], -pos[2], pos[1])

        # Set the tail based on default offset
        bone.tail = [
            joint_pos[name][0], joint_pos[name][1] + 0.1, joint_pos[name][2]
        ]

        # If there's a parent, set it, but skip setting for the root
        if parent_name and parent_name in bones:
            bone.parent = bones[parent_name]

        bones[name] = bone

        # Recursively create children
        if name in joint_hier:
            for child in joint_hier[name]:
                create_bone(child, name)

    # Create root bone with no parent
    create_bone(root_name)

    # Go back to object mode once you're done with the bone setup
    bpy.ops.object.mode_set(mode='OBJECT')

    return armature


def assign_weights(mesh_obj, armature_obj, joint_skin):
    for skin in joint_skin:
        vtx_id = int(skin[0])
        for j in range(1, len(skin), 2):
            joint_name = skin[j]
            weight = float(skin[j + 1])

            if joint_name not in mesh_obj.vertex_groups:
                mesh_obj.vertex_groups.new(name=joint_name)

            vg = mesh_obj.vertex_groups[joint_name]
            vg.add([vtx_id], weight, 'REPLACE')

    modifier = mesh_obj.modifiers.new(name="ArmatureMod", type='ARMATURE')
    modifier.object = armature_obj


def import_obj(obj_path):
    bpy.ops.import_scene.obj(filepath=obj_path,
                             use_split_objects=False,
                             use_split_groups=False,
                             use_groups_as_vgroups=False,
                             axis_forward='-Z',
                             axis_up='Y')
    objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if objs:
        return objs[0]
    else:
        raise RuntimeError("No mesh found in imported OBJ.")


def export_fbx(filepath):
    bpy.ops.export_scene.fbx(filepath=filepath,
                             use_armature_deform_only=True,
                             add_leaf_bones=False)


def main():
    model_id = "dragon"
    base_dir = "../../asset/dragon_texture_obj/"

    obj_path = os.path.join(base_dir, f"{model_id}_ori.obj")
    rig_path = os.path.join(base_dir, f"{model_id}_ori_rig.txt")
    out_path = os.path.join(base_dir, f"{model_id}_ori.fbx")

    bpy.ops.wm.read_factory_settings(use_empty=True)

    mesh_obj = import_obj(obj_path)
    joint_pos, joint_hier, joint_skin, root_name = load_rig_info(rig_path)
    armature_obj = build_armature(joint_pos, joint_hier, root_name)

    bpy.context.view_layer.objects.active = mesh_obj
    assign_weights(mesh_obj, armature_obj, joint_skin)

    export_fbx(out_path)
    print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()
