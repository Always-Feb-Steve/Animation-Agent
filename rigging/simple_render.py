'''
    Tested with Blender 3.0.0
    usage: blender --background --python simple_render.py
'''
import bpy
import os
import math


def apply_smooth_shading(obj):
    """
    Apply smooth shading to the given object.
    """
    # Ensure the object is a mesh
    if obj.type == 'MESH':
        # Select the object and make it active
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()  # Apply smooth shading
    else:
        print(f"Smooth shading not applied: {obj.name} is not a mesh.")


def setup_scene():
    # Clear the scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Add a plane as the ground
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"

    # Add a material to the ground
    mat_ground = bpy.data.materials.new(name="GroundMaterial")
    mat_ground.use_nodes = True
    mat_ground.node_tree.nodes["Principled BSDF"].inputs[
        "Base Color"].default_value = (0.5, 0.5, 0.5, 1)  # Gray
    ground.data.materials.append(mat_ground)

    # print(f"Imported human object: {human.name}")
    # print(f"Location: {human.location}, Scale: {human.scale}")
    # print(f"Number of materials: {len(human.data.materials)}")
    # print("Scene objects:")
    # for obj in bpy.context.scene.objects:
    #     print(f"Object: {obj.name}, Type: {obj.type}")
    # print(f"Vertices: {len(human.data.vertices)}")
    # print(f"Edges: {len(human.data.edges)}")
    # print(f"Faces: {len(human.data.polygons)}")


def add_character(character_obj_path,
                  character_tex_path,
                  position=(0.0, 0.0, 0.0),
                  scale=1.0,
                  rotation_deg=None,
                  rotation_quat=None,
                  rotation_axis_angle=None):
    # Import the character .obj file
    if os.path.exists(character_obj_path):
        bpy.ops.import_scene.obj(filepath=character_obj_path)
        character = bpy.context.selected_objects[0]  # The imported character model
        character.name = "Character"

        # Ensure the character is centered and scaled properly
        character.location = (position[0], position[1], position[2])
        character.scale = (scale, scale, scale)

        apply_smooth_shading(character)

        # Create a material and apply the character texture
        mat_character = bpy.data.materials.new(name="CharacterMaterial")
        mat_character.use_nodes = True
        nodes = mat_character.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        tex_image = nodes.new("ShaderNodeTexImage")

        if os.path.exists(character_tex_path):
            tex_image.image = bpy.data.images.load(character_tex_path)
            mat_character.node_tree.links.new(bsdf.inputs['Base Color'],
                                            tex_image.outputs['Color'])

            # Ensure the character has a valid UV map for texture mapping
            if len(character.data.uv_layers) == 0:
                print("Character has no UV map. Adding a UV map...")
                bpy.context.view_layer.objects.active = character
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.uv.smart_project()  # Generate UVs automatically
                bpy.ops.object.mode_set(mode='OBJECT')
            else:
                print("Character already has a UV map.")
        else:
            print(
                f"Warning: Character texture not found at: {character_tex_path}")

        # Assign the material to the character object
        if not character.data.materials:
            character.data.materials.append(mat_character)
        else:
            character.data.materials[0] = mat_character

        # # Handle rotation parameters (run regardless of material presence)
        # if rotation_deg is not None:
        #     # rotation_deg expected as degrees tuple (x_deg, y_deg, z_deg)
        #     character.rotation_mode = 'XYZ'
        #     character.rotation_euler = tuple(math.radians(a) for a in rotation_deg)
        #     print(f"Applied Euler rotation (deg) {rotation_deg} to {character.name}")
        # elif rotation_quat is not None:
        #     # rotation_quat expected as quaternion tuple (w, x, y, z)
        #     character.rotation_mode = 'QUATERNION'
        #     character.rotation_quaternion = rotation_quat
        #     print(f"Applied quaternion rotation {rotation_quat} to {character.name}")
        # elif rotation_axis_angle is not None:
        #     # rotation_axis_angle expected as (ax, ay, az, angle_deg)
        #     ax, ay, az, angle_deg = rotation_axis_angle
        #     # normalize axis
        #     norm = math.sqrt(ax * ax + ay * ay + az * az)
        #     if norm == 0:
        #         print(f"Invalid rotation axis {rotation_axis_angle[:3]}: zero vector, skipping rotation")
        #     else:
        #         nx, ny, nz = ax / norm, ay / norm, az / norm
        #         a = math.radians(angle_deg) / 2.0
        #         w = math.cos(a)
        #         s = math.sin(a)
        #         q = (nx * s, ny * s, nz * s, w)
        #         character.rotation_mode = 'QUATERNION'
        #         character.rotation_quaternion = q
        #         print(f"Applied axis-angle rotation axis=({nx:.3f},{ny:.3f},{nz:.3f}) angle={angle_deg}deg -> quat={q} to {character.name}")
    else:
        raise FileNotFoundError(
            f"The character .obj file was not found at: {character_obj_path}")


def _setup_lighting():
    # Ensure a world is set up in the scene
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new(name="World")

    # Enable environment lighting
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    background = world_nodes["Background"]
    background.inputs["Strength"].default_value = 2.0  # Adjust brightness
    background.inputs["Color"].default_value = (1, 1, 1, 1
                                                )  # White environment light

    # Add a sun light to ensure additional lighting
    bpy.ops.object.light_add(type='SUN', location=(0, -5, 5))
    sun = bpy.context.object
    sun.data.energy = 2.0  # Increase sunlight brightness


def setup_lighting(hdri_path=None):
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new(name="World")

    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links

    background = world_nodes.get("Background")
    if background:
        world_nodes.remove(background)

    env_tex = world_nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(
        hdri_path) if hdri_path and os.path.exists(hdri_path) else None

    background = world_nodes.new(type="ShaderNodeBackground")
    world_links.new(env_tex.outputs["Color"], background.inputs["Color"])

    world_output = world_nodes.get("World Output") or world_nodes.new(
        type="ShaderNodeOutputWorld")
    world_links.new(background.outputs["Background"],
                    world_output.inputs["Surface"])

    background.inputs["Strength"].default_value = 5.0


def setup_camera():
    # Add a camera
    bpy.ops.object.camera_add(location=(0, -5, 1.2), rotation=(1.5, 0, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam


def setup_render(output_path):
    # Set render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'  # Use GPU if available
    bpy.context.scene.render.threads_mode = 'FIXED'
    bpy.context.scene.render.threads = 16
    bpy.context.scene.cycles.samples = 512

    # Set resolution
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    # Set output file path
    bpy.context.scene.render.filepath = output_path


def main():
    # Path to files
    name = "trex"
    character_obj_path = f"asset/{name}_texture_obj/{name}_texture.obj"  # Character .obj file path
    character_tex_path = f"asset/{name}_texture_obj/{name}_texture.jpg"  # Character texture path
    output_path = f"render.png"  # Output file path

    setup_scene()
    add_character(character_obj_path,
                  character_tex_path,
                  position=(0, 0, 0.3),
                  scale=1.0,
                  rotation_axis_angle=(0, 0, 1, 0))
    # setup_lighting("studio_small_09_4k.hdr")
    _setup_lighting()
    setup_camera()
    setup_render(output_path)

    # Render the scene
    bpy.ops.render.render(write_still=True)
    print(f"Render complete. Image saved to {output_path}")


if __name__ == "__main__":
    main()
