"""
director.py -- mode A of animate.py: Claude choreographs the keyframes itself,
no reference video.

Why there is a calibration step
-------------------------------
A joint's rotation is an axis-angle about the REST-pose world axes, carried
along by its parent. Asking Claude "rotate Spine02 about z by -40 degrees"
invites exactly the sign/axis confusion agent_fb_07 spent seven versions
fighting (forward bow vs backward lean). So Claude never reasons about axes:

1. Facing: four rest renders (azimuth 0/90/180/270), Claude picks the one
   that shows the front. That fixes the body frame (forward, up, left).
2. Motion table: every adjustable joint is rotated +30 degrees about x, y, z
   in turn, and the displacement of its farthest descendant is measured in the
   body frame. "Spine02 x+: tip moves forward, down" is a measurement, not a
   guess; Claude only has to read the table.
3. Plan: Claude writes keyframes as JSON (degrees per axis, root offsets in
   body heights).
4. Review: the keyframes are rendered front + side and shown back to Claude,
   which approves or returns a corrected plan.
"""

import os
import json
import math
import re

import cv2
import numpy as np
import torch
from pytorch3d.renderer import (TexturesVertex, FoVPerspectiveCameras,
                                look_at_view_transform)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

from rendering.renderer import Renderer
from rendering.camera import get_camera
import agent_fb_07 as fb07
import measure


UP = np.array([0.0, 1.0, 0.0])    # rig files are written Y-up by blender_read_fbx.py
PROBE_DEG = 30.0
VIEW_SIZE = 320


# ==============================================================================
# RENDERING HELPERS
# ==============================================================================

def make_camera(dist, elev, azim, device, at=(0.0, 0.0, 0.0)):
    """get_camera with a look-at point; same lens (fov 60) as get_camera."""
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim,
                                  at=(tuple(float(a) for a in at),))
    return FoVPerspectiveCameras(device=device, R=R, T=T, znear=1, zfar=100, fov=60)


def _render(model, renderer, lights, azim, elev, dist, device, at=None):
    camera = (get_camera(dist=dist, elev=elev, azim=azim, device=device) if at is None
              else make_camera(dist, elev, azim, device, at))
    with torch.no_grad():
        img = renderer.render(model(), camera, lights,
                              background_color=torch.tensor([1.0, 1.0, 1.0]))
    rgb = (img[0, :, :, :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), camera


def _caption(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def _label_joints(img, camera, model, names):
    """Write joint names at their projected positions, so Claude can tie the
    names in the table to body parts."""
    pos = model.get_joint_positions()
    pts = torch.stack([pos[n] for n in names])[None]
    size = img.shape[0]
    with torch.no_grad():
        scr = camera.transform_points_screen(pts, image_size=((size, size),))[0, :, :2]
    out = img.copy()
    for n, (x, y) in zip(names, scr.cpu().numpy()):
        x, y = int(x), int(y)
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(out, n, (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (160, 0, 0), 1, cv2.LINE_AA)
    return out


def _b64_png(img):
    import base64
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _img_block(img):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": _b64_png(img)}}


# ==============================================================================
# 1. BODY FRAME
# ==============================================================================

def detect_front_azimuth(model, lights, client, claude_model, dist, device, log):
    renderer = Renderer(image_size=VIEW_SIZE, device=device)
    letters = "ABCD"
    azims = [0, 90, 180, 270]
    tiles = [_caption(_render(model, renderer, lights, a, 10, dist, device)[0], l)
             for a, l in zip(azims, letters)]
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    response = client.messages.create(
        model=claude_model, max_tokens=4096,
        messages=[{"role": "user", "content": [
            _img_block(grid),
            {"type": "text", "text":
                "The same 3D character seen from four sides (A, B, C, D). Which "
                "view looks at the character's FRONT -- the side its face, eyes "
                "or chest point toward (for an animal: its head/snout end seen "
                "head-on; for an object: the side that reads as its front)? "
                "Answer with one letter only."},
        ]}],
    )
    raw = fb07._answer_text(response).strip()
    m = re.search(r"\b([ABCD])\b", raw.splitlines()[-1] if raw else "") or \
        re.search(r"\b([ABCD])\b", raw)
    letter = m.group(1) if m else "A"
    azim = azims[letters.index(letter)]
    log(f"front view: {letter} (azim {azim})  raw={raw[:80]!r}")
    return azim, grid


def body_frame(front_azim):
    a = math.radians(front_azim)
    # PyTorch3D's look_at camera at azimuth a sits on (sin a, ., cos a); the
    # character faces that camera, so that is its forward.
    forward = np.array([math.sin(a), 0.0, math.cos(a)])
    left = np.cross(UP, forward)
    return {"forward": forward, "up": UP, "left": left}


def check_left_right(model, frame, log):
    """Sanity check against L_/R_ joint names when the rig has them."""
    pos = {k: v.detach().cpu().numpy() for k, v in model.joints.items()}
    score = 0.0
    for n, p in pos.items():
        if n.startswith("L_") and "R_" + n[2:] in pos:
            score += float(np.dot(p - pos["R_" + n[2:]], frame["left"]))
    if score:
        log(f"L/R name check: {'agrees' if score > 0 else 'DISAGREES'} with detected facing")
    return score >= 0


# ==============================================================================
# 2. MOTION TABLE
# ==============================================================================

def _set_rot(model, joint, deg_xyz):
    p = model.joint_rotations[joint]
    v = torch.tensor([math.radians(d) for d in deg_xyz], dtype=p.dtype, device=p.device)
    p.data.copy_(matrix_to_rotation_6d(axis_angle_to_matrix(v)))


def _descendants(model, joint):
    out, stack = [], list(model.hierarchy.get(joint, []))
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(model.hierarchy.get(c, []))
    return out


def _end_label(model, joint, desc):
    """Human-readable name for 'the part this joint moves': the farthest
    descendant, or the main branches when it forks (spine -> head + arms)."""
    kids = model.hierarchy.get(joint, [])
    if len(kids) <= 1:
        j0 = model.joints[joint]
        return max(desc, key=lambda c: float((model.joints[c] - j0).norm()))
    ends = [c for c in desc if not model.hierarchy.get(c)]
    return "+".join(sorted(ends)[:4]) + ("+..." if len(ends) > 4 else "")


def _describe(vec, frame):
    comps = {"forward": float(vec @ frame["forward"]),
             "up": float(vec @ frame["up"]),
             "left": float(vec @ frame["left"])}
    opposite = {"forward": "backward", "up": "down", "left": "right"}
    words = []
    for k, v in sorted(comps.items(), key=lambda kv: -abs(kv[1])):
        if abs(v) >= 0.2:
            words.append(k if v > 0 else opposite[k])
    return ", ".join(words) if words else "twists in place (spins around the bone)"


def motion_table(model, frame, adjustable):
    """{joint: {"tip": label, "x": "forward, down", ...}} measured, not guessed.

    The probe point is the centroid of ALL descendant joints, not the farthest
    one: for a fork like Spine02 the farthest descendant is a hanging hand, and
    its motion says the opposite of what the torso does.
    """
    rest = fb07.save_joint_state(model)
    c2p = fb07._build_child_to_parent(model.hierarchy)

    def centroid(pos, desc):
        return torch.stack([pos[c] for c in desc]).mean(0).cpu().numpy()

    table = {}
    with torch.no_grad():
        for j in sorted(adjustable):
            desc = _descendants(model, j)
            if not desc:
                continue
            pos = model.get_joint_positions()
            p0 = centroid(pos, desc)
            length = float(np.linalg.norm(p0 - pos[j].cpu().numpy()))
            if length < 1e-6:
                continue
            row = {"parent": c2p.get(j), "tip": _end_label(model, j, desc)}
            for i, ax in enumerate("xyz"):
                deg = [0.0, 0.0, 0.0]
                deg[i] = PROBE_DEG
                _set_rot(model, j, deg)
                d = (centroid(model.get_joint_positions(), desc) - p0) / length
                row[ax] = _describe(d, frame)
                fb07.restore_joint_state(model, rest)
            table[j] = row
    return table


def table_text(table, semantics=None):
    sem = semantics or {}

    def name(j):
        return f"{j} [{sem[j]}]" if j in sem else str(j)

    lines = []
    for j, r in table.items():
        what = f"is the {sem[j]}" if j in sem else f"moves the part ending at {r['tip']}"
        lines.append(f"{name(j)} (parent {name(r['parent'])}; {what}): "
                     f"x+ -> {r['x']} | y+ -> {r['y']} | z+ -> {r['z']}")
    return "\n".join(lines)


# ==============================================================================
# 2b. BONE NAMES  (UniRig rigs say bone_12, not left_wing)
# ==============================================================================

GENERIC_NAME = re.compile(r"^bone_?\d+$", re.I)
NAME_TILE = 192


def has_generic_names(names):
    names = list(names)
    return sum(bool(GENERIC_NAME.match(n)) for n in names) >= 0.5 * max(len(names), 1)


def _bone_regions(model, adjustable):
    """Per adjustable joint: (own vertices, vertices of everything it moves).
    A vertex belongs to the adjustable joint its dominant skin weight resolves
    to -- the same rule agent_fb_07 uses for its error attribution."""
    c2p = fb07._build_child_to_parent(model.hierarchy)
    V = model.mesh.verts_packed().shape[0]
    owner = np.full(V, None, dtype=object)
    for v, ws in model.skin_weights.items():
        if ws:
            owner[v] = fb07.nearest_adjustable_ancestor(max(ws, key=lambda x: x[1])[0],
                                                        adjustable, c2p)
    regions = {}
    for j in adjustable:
        moved = set(_descendants(model, j)) | {j}
        own = np.array([o == j for o in owner])
        sub = np.array([o in moved for o in owner])
        regions[j] = (own, sub)
    return regions


def _render_colored(model, colors, renderer, lights, azim, dist, device):
    with torch.no_grad():
        m = model()
    mesh = Meshes(verts=m.verts_list(), faces=m.faces_list(),
                  textures=TexturesVertex(verts_features=[torch.tensor(
                      colors, dtype=torch.float32, device=m.device)]))
    camera = get_camera(dist=dist, elev=10, azim=azim, device=device)
    with torch.no_grad():
        img = renderer.render(mesh, camera, lights,
                              background_color=torch.tensor([1.0, 1.0, 1.0]))
    rgb = (img[0, :, :, :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


NAME_SYSTEM = """You label the bones of an automatically rigged 3D character. \
Each tile shows what ONE bone moves: RED = the bone's own region, ORANGE = \
the parts further down the chain that move along with it, GREY = unaffected. \
Each tile has two views: LEFT half = the character seen from its FRONT (so its \
own left side appears on the RIGHT of the image), RIGHT half = seen from its \
own LEFT side (it faces left on screen).

Give every bone a short anatomical name, at most 5 words, from the \
character's OWN point of view: e.g. "left wing, outer half", "right front \
upper leg", "tail tip", "neck base", "spine (chest)", "jaw". Use the parent \
list to keep chains consistent (thigh -> shin -> foot). If a bone moves \
nothing visible, call it "unused".

Answer with ONE ```json block mapping bone name -> label, covering every bone."""


def name_bones(model, lights, client, claude_model, front_azim, dist, device,
               adjustable, out_dir, log):
    regions = _bone_regions(model, adjustable)
    c2p = fb07._build_child_to_parent(model.hierarchy)
    renderer = Renderer(image_size=NAME_TILE, device=device)
    V = model.mesh.verts_packed().shape[0]
    tiles = []
    order = sorted(adjustable, key=lambda n: (len(n), n))
    for j in order:
        own, sub = regions[j]
        col = np.full((V, 3), 0.82)
        col[sub] = (1.0, 0.65, 0.2)
        col[own] = (0.9, 0.1, 0.1)
        views = [_render_colored(model, col, renderer, lights, a, dist, device)
                 for a in (front_azim, front_azim + 90)]
        tiles.append(_caption(np.hstack(views), j))
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(f"{out_dir}/bone_regions.png", grid)

    parents = "\n".join(f"{j}: parent {c2p.get(j)}" for j in order)
    response = client.messages.create(
        model=claude_model, max_tokens=16000, system=NAME_SYSTEM,
        messages=[{"role": "user", "content": [
            _img_block(grid),
            {"type": "text", "text": f"Bones and their parents:\n{parents}"}]}])
    reply = fb07._answer_text(response)
    with open(f"{out_dir}/bone_names.md", "w", encoding="utf-8") as f:
        f.write(reply)
    m = re.search(r"```json\s*(\{.*?\})\s*```", reply, re.S)
    try:
        names = json.loads(m.group(1) if m else reply[reply.find("{"): reply.rfind("}") + 1])
    except ValueError:
        log("bone naming: unparsable reply, continuing with raw names")
        return {}
    names = {k: str(v) for k, v in names.items() if k in adjustable}
    log(f"bone naming: {len(names)}/{len(adjustable)} bones labeled")
    return names


# ==============================================================================
# 3. PLAN -> STATES
# ==============================================================================

PLAN_SYSTEM = """You are a character animator. You pose a rigged 3D character by \
rotating its joints, and a program interpolates smoothly between your keyframes \
and renders a video.

HOW ROTATIONS WORK
- Each joint rotates the body part below it (all its descendants) about the \
joint itself.
- Angles are degrees about the x, y, z axes, ABSOLUTE relative to the rest \
pose (not cumulative between keyframes). Omitted joints stay at rest.
- You do NOT need to reason about axes. Use the MOTION TABLE: it was MEASURED \
by rotating each joint +30 degrees about each axis in the rest pose and \
recording where the end of that part moved, in the character's own frame \
(forward = the way it faces, left = its own left). A negative angle moves the \
opposite way. Combining two axes on one joint combines the motions.
- "twists in place" axes spin the part around itself; use them sparingly.
- A child inherits its parent's rotation: if the spine bends forward 40, the \
arms go with it; to keep the hands hanging down, counter-rotate the arms.

ROOT
"root": {"forward": f, "up": u, "left": l, "turn": deg} moves the whole \
character. Offsets are in body heights (0.1 = a tenth of its height); turn is \
degrees about the vertical, positive = turns toward its own left. With \
"grounded": true the program keeps the lowest point of the body on the floor \
(plus "up"), so crouching never floats and a jump is just "up" > 0.

GOOD ANIMATION
- Keyframe 0 at t=0 is the rest pose (empty joints) unless told otherwise.
- Use 3-10 keyframes, 0.3-1.0 s apart. Add anticipation before big moves and \
settle at the end.
- Be BOLD. This is stylised animation: a pose must be readable at a glance by \
someone who was not told the command. Timid 10-20 degree moves read as \
nothing. A stretch, a wing flap or fold, a big wave usually needs 45-120 \
degrees, spread over a CHAIN of joints (e.g. 3 spine joints x 30 rather than \
one joint x 90). Only avoid poses that are physically impossible (parts \
passing through the body).
- Move several joints together (spine + neck + limbs) so it does not look \
robotic.

WORKFLOW
You can SEE and MEASURE your work before committing to it:
- preview_pose(joints, root, grounded, label): renders ONE pose. Get the hard \
poses right first: anything beyond ~60 degrees, or combining two axes on one \
joint, is hard to predict from the table (it was measured at 30 degrees) -- \
check it instead of guessing.
- solve_ik(effector, target, chain, base): when you know WHERE a hand/paw/ \
head/tail tip should be ("right hand 20% of body height above the head"), \
let the solver find the angles instead of guessing them. It returns the full \
joints dict and a preview.
- mirror_pose(joints, mode): "copy" makes a pose symmetric (left arm -> both \
arms), "swap" mirrors it (sway right -> sway left). Left/right are paired for \
you, also on bone_N rigs.
- preview_plan(plan): renders every keyframe of a full plan.
- save_pose(name, joints, root, note): store a pose that worked (especially \
one that took several tries) in this character's pose library for future \
commands.
- submit_plan(plan, check): your final answer. Only submit a plan whose \
preview_plan passed the strict check below.

Every preview also reports MEASURED problems: body parts inside each other \
("R_Forearm is inside Spine02") and a centre of mass outside the feet. Treat \
an intersection as a FAIL unless the parts merely touch on purpose.

Every image shows two views per pose: TOP / LEFT = the camera the final \
video uses, BOTTOM / RIGHT = side view from the character's own left (it \
faces LEFT on screen). The camera follows the character, so travelling does \
not leave the frame.

STRICT CHECK -- you are the director, not the animator defending the work:
1. Split the command into its separate actions.
2. Each action must be clearly visible with convincing amplitude: a viewer \
who was not told the command would name it from the images. "Slightly" or \
"hinted" is a FAIL.
3. No wrong directions (a flipped sign is the most common mistake), no parts \
passing through the body or head, no floating or sinking.

PLAN FORMAT (for preview_plan and submit_plan)
{"duration": seconds, "grounded": true,
 "keyframes": [{"t": 0.0, "joints": {}, "root": {}},
               {"t": 0.8, "joints": {"JointName": {"x": 0, "y": 0, "z": 0}},
                "root": {"forward": 0, "up": 0, "left": 0, "turn": 0}}]}
Use only joint names from the motion table."""


_POSE_SCHEMA = {
    "type": "object",
    "properties": {
        "joints": {"type": "object",
                   "description": "joint name -> {x, y, z} degrees, absolute vs rest"},
        "root": {"type": "object", "description": "{forward, up, left, turn}"},
        "grounded": {"type": "boolean"},
        "label": {"type": "string", "description": "what this pose is meant to show"},
    },
    "required": ["joints"],
}

_TARGET = {
    "type": "object",
    "description": "where the effector should go: offsets in body heights from "
                   "`relative_to` (a joint name; default: the effector's own position "
                   "in the base pose), in the character's frame",
    "properties": {"relative_to": {"type": "string"}, "forward": {"type": "number"},
                   "up": {"type": "number"}, "left": {"type": "number"}},
}

TOOLS = [
    {"name": "solve_ik",
     "description": "Find angles that put `effector` (any joint, e.g. a hand) at a "
                    "target point, by rotating up to `chain` joints above it (default 3). "
                    "Starts from `base` (a pose: joints/root/grounded) and changes only "
                    "the chain. Returns the full joints dict, the residual and a preview.",
     "input_schema": {"type": "object",
                      "properties": {"effector": {"type": "string"}, "target": _TARGET,
                                     "chain": {"type": "integer"},
                                     "base": _POSE_SCHEMA},
                      "required": ["effector", "target"]}},
    {"name": "mirror_pose",
     "description": "Mirror a pose across the character's left/right plane. mode 'copy' "
                    "keeps the input and adds the other side; 'swap' mirrors everything "
                    "(also root left/turn). Returns the joints dict and a preview.",
     "input_schema": {"type": "object",
                      "properties": {"joints": {"type": "object"},
                                     "root": {"type": "object"},
                                     "mode": {"type": "string", "enum": ["copy", "swap"]}},
                      "required": ["joints", "mode"]}},
    {"name": "save_pose",
     "description": "Save a verified pose to this character's pose library.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "joints": {"type": "object"},
                                     "root": {"type": "object"},
                                     "note": {"type": "string"}},
                      "required": ["name", "joints"]}},
    {"name": "preview_pose",
     "description": "Render ONE pose: the video camera next to a side view.",
     "input_schema": _POSE_SCHEMA},
    {"name": "preview_plan",
     "description": "Render every keyframe of a plan: one column per keyframe, "
                    "video camera on top, side view below.",
     "input_schema": {"type": "object", "properties": {"plan": {"type": "object"}},
                      "required": ["plan"]}},
    {"name": "submit_plan",
     "description": "Submit the final plan, only after its preview passed the strict check.",
     "input_schema": {"type": "object",
                      "properties": {"plan": {"type": "object"},
                                     "check": {"type": "string",
                                               "description": "one line per action: "
                                                              "PASS/FAIL and where it shows"}},
                      "required": ["plan", "check"]}},
]


def parse_plan(text):
    """A plan from a tool input (dict, or a JSON string) or from free text."""
    if isinstance(text, dict):
        plan = text
    else:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
        raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
        plan = json.loads(raw)
    if not plan.get("keyframes"):
        raise ValueError("plan has no keyframes")
    plan["keyframes"] = sorted(plan["keyframes"], key=lambda k: float(k["t"]))
    return plan


def unknown_joints(plan, adjustable):
    return sorted({j for kf in plan["keyframes"] for j in (kf.get("joints") or {})
                   if j not in adjustable})


def _min_up(model):
    with torch.no_grad():
        v = model().verts_packed().cpu().numpy()
    return float((v @ UP).min())


def plan_to_keys(model, plan, frame, adjustable, rest_state, height, log=print):
    """Plan JSON -> [(t, (rots, root))] in the format animate.interpolate_state
    takes. Unknown joint names are dropped with a warning, not fatal."""
    keys = []
    rest_min = None
    root_p = fb07.root_param(model)
    for kf in plan["keyframes"]:
        fb07.restore_joint_state(model, rest_state)
        if rest_min is None:
            rest_min = _min_up(model)
        for j, ang in (kf.get("joints") or {}).items():
            if j not in adjustable:
                log(f"  ignoring unknown joint {j!r}")
                continue
            _set_rot(model, j, [float(ang.get(a, 0.0)) for a in "xyz"])
        r = kf.get("root") or {}
        offset = (float(r.get("forward", 0)) * frame["forward"]
                  + float(r.get("left", 0)) * frame["left"]) * height
        yaw = math.radians(float(r.get("turn", 0)))
        rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(
            torch.tensor(UP * yaw, dtype=root_p.dtype, device=root_p.device)))
        root_p.data[3:] = rot6d
        root_p.data[:3] = rest_state[1][:3] + torch.tensor(offset, dtype=root_p.dtype,
                                                            device=root_p.device)
        up = float(r.get("up", 0)) * height
        if plan.get("grounded", True):
            up += rest_min - _min_up(model)
        root_p.data[:3] += torch.tensor(UP * up, dtype=root_p.dtype, device=root_p.device)
        keys.append((float(kf["t"]), fb07.save_joint_state(model)))
    fb07.restore_joint_state(model, rest_state)
    if keys[0][0] > 0:
        keys.insert(0, (0.0, rest_state))
    return keys


def film_azimuth(model, frame, front_azim):
    """Where the final video's camera sits. Long-bodied characters (quadrupeds,
    fish) shot near head-on hide the body behind the head -- the cat's stretch
    was invisible from 30 degrees off the front -- so they get a more side-on
    angle."""
    with torch.no_grad():
        v = model().verts_packed().cpu().numpy()
    length = float(np.ptp(v @ frame["forward"]))
    width = float(np.ptp(v @ frame["left"]))
    return front_azim - (60 if length > 1.3 * width else 30)


def _pose_shot(model):
    """(center, dist) framing the CURRENT pose; 2.2 x half-diagonal is the
    factor that frames ultraman's rest pose at 1.25."""
    with torch.no_grad():
        v = model().verts_packed()
    lo, hi = v.min(0).values, v.max(0).values
    return ((lo + hi) / 2).cpu().numpy(), 2.2 * float((hi - lo).norm()) / 2


def shots_for_keys(model, keys, rest_state):
    """One centre per key, one shared distance: the camera follows the
    character and is sized to its biggest POSE, not to how far it travels.
    Framing the union of all keyframes made the dolphin a speck."""
    centers, dist = [], 0.0
    for _, state in keys:
        fb07.restore_joint_state(model, state)
        c, d = _pose_shot(model)
        centers.append(c)
        dist = max(dist, d)
    fb07.restore_joint_state(model, rest_state)
    return centers, dist


def _two_views(model, renderer, lights, film_azim, front_azim, center, dist, device):
    film, _ = _render(model, renderer, lights, film_azim, 10, dist, device, center)
    # Camera at front+90 sits on the character's left, so on screen it faces left.
    side, _ = _render(model, renderer, lights, front_azim + 90, 10, dist, device, center)
    return film, side


def contact_sheet(model, keys, lights, film_azim, front_azim, rest_state, device):
    centers, dist = shots_for_keys(model, keys, rest_state)
    renderer = Renderer(image_size=VIEW_SIZE, device=device)
    cols = []
    for (t, state), c in zip(keys, centers):
        fb07.restore_joint_state(model, state)
        film, side = _two_views(model, renderer, lights, film_azim, front_azim, c, dist, device)
        cols.append(np.vstack([_caption(film, f"t={t:.2f}s"), side]))
    fb07.restore_joint_state(model, rest_state)
    return np.hstack(cols)


def pose_sheet(model, state, lights, film_azim, front_azim, rest_state, device, label=""):
    fb07.restore_joint_state(model, state)
    center, dist = _pose_shot(model)
    rest_dist = shots_for_keys(model, [(0.0, rest_state)], rest_state)[1]
    fb07.restore_joint_state(model, state)
    renderer = Renderer(image_size=VIEW_SIZE, device=device)
    film, side = _two_views(model, renderer, lights, film_azim, front_azim, center,
                            max(dist, rest_dist), device)
    fb07.restore_joint_state(model, rest_state)
    return np.hstack([_caption(film, label[:40]), side])


def tracking_cameras(model, keys, fps, interpolate, film_azim, elev, device,
                     rest_state, smooth_s=0.5):
    """Per-frame cameras for the final video: centre follows the character's
    bounding box (moving average over `smooth_s` so it glides, not jitters),
    distance fixed at the largest pose."""
    n = int(round(keys[-1][0] * fps)) + 1
    centers, dist = [], 0.0
    for i in range(n):
        fb07.restore_joint_state(model, interpolate(keys, i / fps))
        c, d = _pose_shot(model)
        centers.append(c)
        dist = max(dist, d)
    fb07.restore_joint_state(model, rest_state)
    C = np.array(centers)
    w = max(1, int(smooth_s * fps)) | 1
    pad = np.pad(C, ((w // 2, w // 2), (0, 0)), mode="edge")
    smooth = np.stack([pad[i:i + w].mean(0) for i in range(n)])
    return [make_camera(dist, elev, film_azim, device, at=c) for c in smooth]


# ==============================================================================
# DRIVER
# ==============================================================================

REFLECT = """The plan is accepted. Before you go, two things for next time:
1. lessons: what did you learn in this session that would help animate OTHER \
characters -- an axis or angle pitfall, a body-type trick, a tool habit that \
paid off? 0-2 short strings (max 25 words each); none if nothing goes beyond \
the LESSONS you were given.
2. poses: up to 2 keyframes of your submitted plan worth keeping in THIS \
character's pose library -- distinctive or hard-won poses you would want to \
reuse (not the rest pose).
Reply with ONLY a JSON object:
{"lessons": ["..."], "poses": [{"t": <keyframe time>, "name": "short_snake_case", \
"note": "what it shows"}]}"""


def direct(model, lights, client, claude_model, command, out_dir, dist, device,
           max_steps=14, log=print, memory=None, lessons=None, previous=None):
    """Returns (keys, film_azim). Planning is a tool-use loop: Claude previews
    single poses and whole plans, and submits when its own strict check
    passes. The earlier plan -> review -> rewrite loop stalled: the reviewer
    diagnosed the panda correctly three times, but every rewrite was a blind
    guess at 150-degree compound rotations the 30-degree table cannot predict."""
    rest_state = fb07.save_joint_state(model)
    adjustable = fb07.get_adjustable_joints(model)

    front_azim = memory.get("front_azim") if memory else None
    if front_azim is None:
        front_azim, grid = detect_front_azimuth(model, lights, client, claude_model,
                                                dist, device, log)
        cv2.imwrite(f"{out_dir}/facing_views.png", grid)
        if memory:
            memory.set("front_azim", front_azim)
    else:
        log(f"front view: azim {front_azim} (memory)")
    frame = body_frame(front_azim)
    check_left_right(model, frame, log)

    film_azim = film_azimuth(model, frame, front_azim)
    log(f"film camera azim {film_azim}")

    semantics = {}
    if has_generic_names(adjustable):
        semantics = memory.get("bone_names") if memory else None
        if semantics:
            log(f"bone naming: {len(semantics)} names (memory)")
        else:
            semantics = name_bones(model, lights, client, claude_model, front_azim, dist,
                                   device, adjustable, out_dir, log)
            if memory and semantics:
                memory.set("bone_names", semantics)

    table = motion_table(model, frame, adjustable)
    text = table_text(table, semantics)
    with open(f"{out_dir}/motion_table.txt", "w", encoding="utf-8") as f:
        f.write(text)

    with torch.no_grad():
        v = model().verts_packed().cpu().numpy() @ UP
    height = float(v.max() - v.min())
    checks = measure.BodyChecks(model, adjustable, height, semantics)
    mirror = measure.Mirror(model, frame, adjustable, height)

    renderer = Renderer(image_size=512, device=device)
    names = sorted(table)
    front_img, cam_f = _render(model, renderer, lights, front_azim + 20, 10, dist, device)
    side_img, cam_s = _render(model, renderer, lights, front_azim + 90, 10, dist, device)
    labeled = np.hstack([_label_joints(front_img, cam_f, model, names),
                         _label_joints(side_img, cam_s, model, names)])
    cv2.imwrite(f"{out_dir}/rest_labeled.png", labeled)

    off = round(film_azim - front_azim)
    side = "right" if off < 0 else "left"
    camera_note = (
        f"CAMERA: the final video is filmed from {abs(off)} degrees to the "
        f"character's {side} of straight ahead (a front-{side} three-quarter view). "
        f"'The viewer', 'the audience' and 'the camera' mean that direction: to "
        f"face it squarely use root turn {off}; to lean or reach toward it, move "
        f"toward the front-{side}.")

    system = PLAN_SYSTEM + "\n\nMOTION TABLE\n" + text
    context = [camera_note]
    if lessons and lessons.text():
        context.append(lessons.text())
    if memory and memory.poses_text():
        context.append(memory.poses_text())
    if previous:
        context.append(
            "PREVIOUS ANIMATION of this character -- the user's last command was "
            f"\"{previous['command']}\" and this plan was rendered:\n"
            f"{json.dumps(previous['plan'])}\n"
            "If the new COMMAND modifies that animation (\"deeper\", \"slower\", "
            "\"add a spin at the end\", \"now with the other arm\"), start from this "
            "plan: preview_plan it, change only what is asked, keep the rest. If the "
            "COMMAND is a different action, ignore it.")
    messages = [{"role": "user", "content": [
        _img_block(labeled),
        {"type": "text", "text": "The character at rest with its joints labeled "
                                 "(left: front view, right: side view, facing left).\n\n"
                                 + "\n\n".join(context)
                                 + f"\n\nCOMMAND: {command}\n\n"
                                 f"You have {max_steps} tool turns."},
    ]}]

    os.makedirs(f"{out_dir}/previews", exist_ok=True)
    agent_log = open(f"{out_dir}/agent_log.md", "w", encoding="utf-8")
    state = {"n": 0, "last_plan": None, "final": None}

    def problems(state_, grounded):
        """Measured issues of one posed state, as short lines."""
        fb07.restore_joint_state(model, state_)
        out = checks.collisions()
        if grounded:
            b = checks.balance(frame)
            if b:
                out.append(b)
        fb07.restore_joint_state(model, rest_state)
        return out

    def pose_result(pose, label, tag, kind):
        """Pose + root + grounded -> preview image and measured problems."""
        one = {"grounded": pose.get("grounded", True),
               "keyframes": [{"t": 0.0, "joints": pose.get("joints") or {},
                              "root": pose.get("root") or {}}]}
        bad = unknown_joints(one, adjustable)
        keys = plan_to_keys(model, one, frame, adjustable, rest_state, height, log)
        img = pose_sheet(model, keys[0][1], lights, film_azim, front_azim,
                         rest_state, device, label)
        cv2.imwrite(f"{out_dir}/previews/{tag}_{kind}.png", img)
        probs = problems(keys[0][1], one["grounded"])
        note = (f"pose '{label}' (left: video camera, right: side view). Measured: "
                + ("; ".join(probs) if probs else "no intersections, balanced"))
        if bad:
            note += f". UNKNOWN joints ignored: {bad}"
        return img, note

    def run_tool(name, inp):
        state["n"] += 1
        tag = f"{state['n']:02d}"
        if name == "preview_pose":
            img, note = pose_result(inp, inp.get("label", ""), tag, "pose")
            return [_img_block(img), {"type": "text", "text": note}]

        if name == "solve_ik":
            eff = inp["effector"]
            if eff not in model.joints:
                raise ValueError(f"unknown effector {eff!r}")
            base = inp.get("base") or {}
            one = {"grounded": base.get("grounded", True),
                   "keyframes": [{"t": 0.0, "joints": base.get("joints") or {},
                                  "root": base.get("root") or {}}]}
            keys = plan_to_keys(model, one, frame, adjustable, rest_state, height, log)
            fb07.restore_joint_state(model, keys[0][1])
            tg = inp.get("target") or {}
            pos = {k: v.detach().cpu().numpy() for k, v in model.get_joint_positions().items()}
            ref = tg.get("relative_to") or eff
            if ref not in pos:
                raise ValueError(f"unknown relative_to joint {ref!r}")
            target = pos[ref] + height * (float(tg.get("forward", 0)) * frame["forward"]
                                          + float(tg.get("up", 0)) * UP
                                          + float(tg.get("left", 0)) * frame["left"])
            angles, res = measure.solve_ik(model, eff, target, int(inp.get("chain", 3)),
                                           adjustable)
            fb07.restore_joint_state(model, rest_state)
            joints = dict(base.get("joints") or {})
            joints.update(angles)
            pose = dict(base, joints=joints)
            img, note = pose_result(pose, f"IK {eff}", tag, "ik")
            return [_img_block(img), {"type": "text", "text":
                    f"residual {res / height * 100:.1f}% of body height. "
                    f"joints: {json.dumps(joints)}. {note}"}]

        if name == "mirror_pose":
            mode = inp.get("mode", "copy")
            joints = mirror.apply(inp.get("joints") or {}, mode)
            root = dict(inp.get("root") or {})
            if mode == "swap":
                for k in ("left", "turn"):
                    if k in root:
                        root[k] = -float(root[k])
            img, note = pose_result({"joints": joints, "root": root}, f"mirror {mode}",
                                    tag, "mirror")
            return [_img_block(img), {"type": "text", "text":
                    f"joints: {json.dumps(joints)}; root: {json.dumps(root)}. {note}"}]

        if name == "save_pose":
            if memory is None:
                return [{"type": "text", "text": "No memory for this run; not saved."}]
            bad = [j for j in (inp.get("joints") or {}) if j not in adjustable]
            if bad:
                raise ValueError(f"unknown joints {bad}")
            memory.save_pose(inp["name"], inp["joints"], inp.get("root"), inp.get("note", ""))
            log(f"  saved pose '{inp['name']}'")
            return [{"type": "text", "text": f"Saved '{inp['name']}'."}]

        plan = parse_plan(inp["plan"])
        bad = unknown_joints(plan, adjustable)
        if name == "preview_plan":
            keys = plan_to_keys(model, plan, frame, adjustable, rest_state, height, log)
            img = contact_sheet(model, keys, lights, film_azim, front_azim, rest_state, device)
            cv2.imwrite(f"{out_dir}/previews/{tag}_plan.png", img)
            state["last_plan"] = plan
            grounded = plan.get("grounded", True)
            measured = [f"t={t:.2f}s: " + "; ".join(p) for t, st in keys
                        for p in [problems(st, grounded)] if p]
            note = (f"{len(plan['keyframes'])} keyframes, {plan.get('duration')}s "
                    f"(top: video camera, bottom: side view). Measured: "
                    + (" | ".join(measured) if measured else "no intersections, balanced"))
            if bad:
                note += f". UNKNOWN joints ignored: {bad}"
            return [_img_block(img), {"type": "text", "text": note}]

        if name == "submit_plan":
            if bad:
                return [{"type": "text", "text": f"Rejected: unknown joints {bad}. "
                                                 "Use names from the motion table."}]
            state["final"] = plan
            agent_log.write(f"### submitted, self-check\n\n{inp.get('check', '')}\n\n")
            return [{"type": "text", "text": "Plan accepted."}]
        raise ValueError(f"unknown tool {name}")

    def learn(results):
        """One extra turn: Claude writes down lessons that transfer to other
        characters and picks keyframes for this character's pose library.
        Picking happens here rather than through save_pose because, offered
        as an optional tool, it was never called."""
        try:
            messages.append({"role": "user", "content": results +
                             [{"type": "text", "text": REFLECT}]})
            response = client.messages.create(model=claude_model, max_tokens=4000,
                                              system=system, tools=TOOLS,
                                              tool_choice={"type": "none"},
                                              messages=messages)
            said = fb07._answer_text(response)
            reply = json.loads(said[said.find("{"): said.rfind("}") + 1])
        except Exception as e:                      # memory is a bonus, never fatal
            log(f"memory: skipped ({type(e).__name__})")
            return
        items = [str(i) for i in (reply.get("lessons") or [])][:2]
        if items and lessons is not None:
            lessons.add(items, memory.name if memory else "?")
            log("lessons: " + " | ".join(items))
        if memory is None:
            return
        for p in (reply.get("poses") or [])[:2]:
            try:
                kf = min(state["final"]["keyframes"],
                         key=lambda k: abs(float(k["t"]) - float(p["t"])))
                if not kf.get("joints"):
                    continue
                memory.save_pose(str(p["name"]), kf["joints"], kf.get("root"),
                                 str(p.get("note", "")))
                log(f"pose library: saved '{p['name']}'")
            except (KeyError, TypeError, ValueError):
                continue

    for step in range(max_steps):
        response = client.messages.create(model=claude_model, max_tokens=16000,
                                          system=system, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": response.content})
        said = fb07._answer_text(response)
        if said:
            agent_log.write(f"## step {step}\n\n{said}\n\n")
        uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not uses:
            messages.append({"role": "user", "content":
                             "Use the tools: preview, then submit_plan."})
            continue
        results = []
        for u in uses:
            try:
                content = run_tool(u.name, u.input)
            except Exception as e:                  # bad input is feedback, not a crash
                content = [{"type": "text", "text": f"error: {type(e).__name__}: {e}"}]
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": content})
            label = u.input.get("label", "") if isinstance(u.input, dict) else ""
            log(f"step {step}: {u.name} {label}".rstrip())
            agent_log.write(f"- tool `{u.name}` {label}\n")
            for c in content:
                if c.get("type") == "text":
                    agent_log.write(f"  -> {c['text'][:800]}\n")
        if state["final"] is not None:
            if lessons is not None or memory is not None:
                learn(results)
            break
        left = max_steps - step - 1
        if left <= 2:
            results.append({"type": "text", "text": f"{left} tool turn(s) left: "
                            "call submit_plan with your best plan now."})
        messages.append({"role": "user", "content": results})
    agent_log.close()

    plan = state["final"] or state["last_plan"]
    if plan is None:
        raise RuntimeError("Claude never previewed or submitted a plan.")
    log("plan: " + ("submitted" if state["final"] else "budget spent, using last preview")
        + f" after {state['n']} tool calls")
    with open(f"{out_dir}/plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    keys = plan_to_keys(model, plan, frame, adjustable, rest_state, height, log)
    duration = float(plan.get("duration") or keys[-1][0])
    if duration > keys[-1][0]:
        keys.append((duration, keys[-1][1]))       # hold the last pose
    return keys, film_azim
