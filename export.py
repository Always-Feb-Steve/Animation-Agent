"""
export.py -- write an animation as glTF 2.0 binary (.glb), optionally FBX.

The .glb holds the textured mesh, the skeleton, the skin weights and one
baked rotation track per joint (plus the root's translation), so it imports
into Blender, Unreal and Unity (glTFast) as a regular skinned animation.

Why the mapping is exact
------------------------
RiggingModel poses joint j with  S_j = S_parent @ T(p_j) R_j T(-p_j),  where
p_j is the rest position and R_j the joint's rotation. A glTF node with local
translation p_j - p_parent and rotation R_j has global G_j = G_parent @
T(p_j - p_parent) R_j, and glTF skins with G_j @ IBM_j. With IBM_j = T(-p_j),
induction gives G_j @ IBM_j = S_j for every joint, so the exported animation
deforms the mesh exactly as the renderer did. The root node additionally
carries the root translation t: translation p_root + t, rotation R_root.
"""

import os
import json
import shutil
import struct
import subprocess

import cv2
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, rotation_6d_to_matrix

import agent_fb_07 as fb07


FLOAT, UINT, USHORT = 5126, 5125, 5123
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
BLENDER_DEFAULT = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"


class _Bin:
    """Accumulates the GLB binary chunk and its bufferViews/accessors."""

    def __init__(self):
        self.data = bytearray()
        self.views, self.accessors = [], []

    def add(self, arr, comp, typ, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        while len(self.data) % 4:
            self.data += b"\0"
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": arr.nbytes}
        if target:
            view["target"] = target
        self.data += arr.tobytes()
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1, "componentType": comp,
               "count": int(arr.shape[0]), "type": typ}
        if minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc["min"] = flat.min(0).astype(float).tolist()
            acc["max"] = flat.max(0).astype(float).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def add_blob(self, blob):
        while len(self.data) % 4:
            self.data += b"\0"
        self.views.append({"buffer": 0, "byteOffset": len(self.data), "byteLength": len(blob)})
        self.data += blob
        return len(self.views) - 1


def _quat_xyzw(rot6d):
    q = matrix_to_quaternion(rotation_6d_to_matrix(rot6d))       # w, x, y, z
    return torch.cat([q[..., 1:], q[..., :1]], -1)


def export_glb(model, keys, interpolate, fps, path, name="character"):
    """Bake `keys` (as rendered: interpolate(keys, t) at `fps`) into `path`."""
    mesh = model.mesh
    V = mesh.verts_packed().detach().cpu().numpy().astype(np.float32)
    F = mesh.faces_packed().cpu().numpy()
    N = mesh.verts_normals_packed().detach().cpu().numpy().astype(np.float32)
    tex = mesh.textures

    # glTF has one UV per vertex; OBJ has one per face corner. Split vertices
    # that carry several UVs.
    has_uv = hasattr(tex, "verts_uvs_padded")
    if has_uv:
        uv = tex.verts_uvs_padded()[0].cpu().numpy()
        fuv = tex.faces_uvs_padded()[0].cpu().numpy()
        pairs = np.stack([F.reshape(-1), fuv.reshape(-1)], 1)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        src = uniq[:, 0]
        tc = uv[uniq[:, 1]].astype(np.float32).copy()
        tc[:, 1] = 1.0 - tc[:, 1]                    # OBJ origin bottom-left -> glTF top-left
        idx = inv.reshape(-1).astype(np.uint32)
    else:
        src = np.arange(V.shape[0])
        idx = F.reshape(-1).astype(np.uint32)

    # Skeleton, in a fixed order with parents before children.
    c2p = fb07._build_child_to_parent(model.hierarchy)
    root = sorted(model.root_joints)[0]
    order, stack = [], [root]
    while stack:
        j = stack.pop()
        order.append(j)
        stack.extend(reversed(model.hierarchy.get(j, [])))
    jid = {j: i for i, j in enumerate(order)}
    P = {j: model.joints[j].detach().cpu().numpy().astype(np.float64) for j in order}

    # Skin: top-4 influences per vertex, renormalised.
    Vn = V.shape[0]
    J4 = np.zeros((Vn, 4), np.uint16)
    W4 = np.zeros((Vn, 4), np.float32)
    for v in range(Vn):
        ws = sorted(model.skin_weights.get(v, []), key=lambda x: -x[1])[:4]
        ws = [(j, w) for j, w in ws if j in jid and w > 0]
        if not ws:
            ws = [(root, 1.0)]
        tot = sum(w for _, w in ws)
        for k, (j, w) in enumerate(ws):
            J4[v, k] = jid[j]
            W4[v, k] = w / tot

    # Bake the animation exactly as render_sequence samples it.
    n = int(round(keys[-1][0] * fps)) + 1
    times = (np.arange(n) / fps).astype(np.float32)
    rest = fb07.save_joint_state(model)
    rots = {j: [] for j in order}
    root_t = []
    with torch.no_grad():
        for i in range(n):
            fb07.restore_joint_state(model, interpolate(keys, i / fps))
            rp = fb07.root_param(model).data
            rots[root].append(_quat_xyzw(rp[3:]).cpu().numpy())
            root_t.append(rp[:3].cpu().numpy() + P[root])
            for j in order[1:]:
                rots[j].append(_quat_xyzw(model.joint_rotations[j].data).cpu().numpy())
            for j in order[1:]:
                if float(model.joint_translations[j].data.abs().max()) > 1e-8:
                    raise ValueError(f"joint_translations[{j}] is non-zero; not exportable")
    fb07.restore_joint_state(model, rest)

    b = _Bin()
    attrs = {"POSITION": b.add(V[src], FLOAT, "VEC3", ARRAY_BUFFER, minmax=True),
             "NORMAL": b.add(N[src], FLOAT, "VEC3", ARRAY_BUFFER),
             "JOINTS_0": b.add(J4[src], USHORT, "VEC4", ARRAY_BUFFER),
             "WEIGHTS_0": b.add(W4[src], FLOAT, "VEC4", ARRAY_BUFFER)}
    if has_uv:
        attrs["TEXCOORD_0"] = b.add(tc, FLOAT, "VEC2", ARRAY_BUFFER)
    ind = b.add(idx, UINT, "SCALAR", ELEMENT_ARRAY_BUFFER)

    ibm = np.stack([np.eye(4, dtype=np.float32) for _ in order])
    for j, i in jid.items():
        ibm[i, :3, 3] = -P[j]
    ibm_acc = b.add(ibm.transpose(0, 2, 1).reshape(-1, 16), FLOAT, "MAT4")   # column-major

    material = {"name": "material",
                "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}}
    textures, images, samplers = [], [], []
    if has_uv:
        img = (tex.maps_padded()[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
        images.append({"bufferView": b.add_blob(jpg.tobytes()), "mimeType": "image/jpeg"})
        samplers.append({"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497})
        textures.append({"source": 0, "sampler": 0})
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}

    t_acc = b.add(times, FLOAT, "SCALAR", minmax=True)
    samplers_a, channels = [], []
    for j in order:
        samplers_a.append({"input": t_acc, "interpolation": "LINEAR",
                           "output": b.add(np.array(rots[j], np.float32), FLOAT, "VEC4")})
        channels.append({"sampler": len(samplers_a) - 1,
                         "target": {"node": 1 + jid[j], "path": "rotation"}})
    samplers_a.append({"input": t_acc, "interpolation": "LINEAR",
                       "output": b.add(np.array(root_t, np.float32), FLOAT, "VEC3")})
    channels.append({"sampler": len(samplers_a) - 1,
                     "target": {"node": 1 + jid[root], "path": "translation"}})

    nodes = [{"name": name, "mesh": 0, "skin": 0}]
    for j in order:
        p = c2p.get(j)
        t = P[j] - (P[p] if p is not None else 0.0)
        node = {"name": j, "translation": t.astype(float).tolist()}
        kids = model.hierarchy.get(j, [])
        if kids:
            node["children"] = [1 + jid[k] for k in kids]
        nodes.append(node)

    gltf = {
        "asset": {"version": "2.0", "generator": "Animation-Agent export.py"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1 + jid[root]]}],
        "nodes": nodes,
        "meshes": [{"name": name, "primitives": [{"attributes": attrs, "indices": ind,
                                                  "material": 0}]}],
        "materials": [material],
        "skins": [{"joints": [1 + jid[j] for j in order], "skeleton": 1 + jid[root],
                   "inverseBindMatrices": ibm_acc}],
        "animations": [{"name": "agent", "samplers": samplers_a, "channels": channels}],
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(b.data)}],
    }
    if images:
        gltf.update(images=images, textures=textures, samplers=samplers)

    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    bin_ = bytes(b.data) + b"\0" * (-len(b.data) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(bin_), 0x004E4942) + bin_)
    return path


FBX_SCRIPT = r'''
import bpy, sys
src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
# The glTF importer adds an unparented "Icosphere" as the bones' display
# shape; it is not part of the character, so keep it out of the FBX.
for o in list(bpy.context.scene.objects):
    if o.type == "MESH" and o.parent is None:
        bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.export_scene.fbx(filepath=dst, path_mode="COPY", embed_textures=True,
                         bake_anim=True, add_leaf_bones=False)
'''


def find_blender():
    for c in (os.environ.get("BLENDER"), shutil.which("blender"), BLENDER_DEFAULT):
        if c and os.path.exists(c):
            return c
    return None


def glb_to_fbx(glb, fbx, blender=None, timeout=300):
    """Convert with Blender if it is installed; returns the path or None."""
    blender = blender or find_blender()
    if not blender:
        return None
    script = os.path.splitext(fbx)[0] + "_tofbx.py"
    with open(script, "w", encoding="utf-8") as f:
        f.write(FBX_SCRIPT)
    try:
        subprocess.run([blender, "--background", "--factory-startup", "--python", script,
                        "--", os.path.abspath(glb), os.path.abspath(fbx)],
                       check=True, timeout=timeout, capture_output=True)
    except (subprocess.SubprocessError, OSError):
        return None
    finally:
        os.remove(script)
    return fbx if os.path.exists(fbx) else None
