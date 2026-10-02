"""
Agent FB-07 = FB-05's global alignment + FB-04-UTM's direction rescue
            + FB-06's axis rescue, plus three fixes for defects all of them share.

Why FB-04-UTM could reach `Spine02 z-` and why that matters
------------------------------------------------------------
On the ultraman bow the silhouette score is structurally blind to which way the
torso bends: folding forward and leaning back both foreshorten the projected
height, so both lower the XOR. At iteration 1 the score ranked `Spine02 z+`
first (13997 from a base of 17189) and the correct forward bow `Spine02 z-`
well below it (16486). FB-04-UTM still got there, because a
`no_wrong_direction` verdict inserts the FLIPPED direction at the FRONT of the
review queue, where it is judged on its own merits and accepted on Claude's
word alone -- a worse-SCORING move beating a better-scoring one. That is the
only mechanism in the whole design that can break the front/back ambiguity, so
it is preserved verbatim.

FB-06 (yours) generalised it: `no_wrong_axis` says the right body part moved
but about the wrong axis, and re-queues every candidate on the other two axes.
FLIP fixes the sign, this fixes the axis. Both are kept.

What FB-05 contributed
-----------------------
Stage 0 aligns the mesh to the reference by the FEET anchor before any joint
search: 17189 -> 10771 (-37.3%) on ultraman, zero joint moves, zero API calls.
Without it ~6174 false pixels are unreachable by any rotation, and the score is
actively misleading -- leaning back scores well partly because it drags the
torso toward where the reference character stands after its sidestep.

Three defects present in ALL of the above, fixed here
------------------------------------------------------
1. Identical questions, re-asked.  When an iteration accepts nothing, the pose
   is unchanged, so the next iteration enumerates the same candidates and asks
   the same questions. FB-04-UTM iterations 3/4/5 and FB-05 iterations 2/3/4
   are verbatim repeats -- ~15 wasted calls per stall, and `patience` expires
   without one new question being asked. FB-07 memoises verdicts by
   (pose signature, joint, direction, step). It also widens the ranking window
   (3 -> 6 -> 9 ...) on each stalled iteration instead of re-offering the same
   three joints.

2. The `improving` filter hides the move that matters.  Candidates were kept
   only if they scored below the current base, so a move that is semantically
   right but temporarily RAISES the XOR never reached review. That is exactly
   the bow: with the arms riding backward on the spine, bending deeper adds
   false pixels, so "bow further" was filtered out before Claude could approve
   it -- the mechanism behind "the bow is not deep enough". FB-07 admits
   candidates up to `--tolerance` above base, ranked after the improving ones
   so they are reached only once everything cheaper has been rejected.
   Guards: a regressive move must be Claude-verified (never auto-accepted),
   consecutive regressions are capped by `--max_regress`, and the best-state
   snapshot still restores the global optimum at the end.

3. Parts dragged by forward kinematics never get a turn.  L_Clavicle and
   R_Clavicle are children of Spine02, so a 74-degree spine rotation swings the
   arms backward rigidly; a real bow counter-rotates the shoulders. Ranking by
   assigned error does not surface them -- in FB-04-UTM the arms fell out of
   the top 3 at iteration 3 and never returned. FB-07 force-includes the direct
   adjustable descendants of whichever joint was just accepted.

Usage:
  conda activate VideoArticulation
  set KMP_DUPLICATE_LIB_OK=TRUE
  set ANTHROPIC_API_KEY=<key>
  python agent_fb_07.py
"""

import os
import json
import base64
import hashlib
import argparse
import glob as _glob
import random
import colorsys

import cv2
import numpy as np
import torch
from pytorch3d.renderer import (
    PointLights,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

import anthropic

from rigging.model import RiggingModel
from utils.io import load_mesh
from utils.rig_parser import parse_rig_file
from rendering.renderer import Renderer
from rendering.camera import get_camera


STEP_LEVELS = [0.8, 0.5, 0.1, 0.05, 0.02]
ROOT_STEP_LEVELS = [0.08, 0.04, 0.01, 0.005, 0.002]   # world units

VERIFY_MIN_STEP = 0.1
ROOT_VERIFY_MIN_STEP = 0.01

ROOT_KEY = "__root__"

DIRECTIONS = [
    ("x+", (1, 0, 0)), ("x-", (-1, 0, 0)),
    ("y+", (0, 1, 0)), ("y-", (0, -1, 0)),
    ("z+", (0, 0, 1)), ("z-", (0, 0, -1)),
]
AXIS_OF = dict(DIRECTIONS)
FLIP = {"x+": "x-", "x-": "x+", "y+": "y-", "y-": "y+", "z+": "z-", "z-": "z+"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# CLAUDE VERIFICATION
# ==============================================================================

VERIFY_SYSTEM = (
    "You are a 3D pose alignment verifier. You will see three images: "
    "the pose BEFORE a joint rotation, the pose AFTER the rotation, and the "
    "REFERENCE target pose. One body part visibly changed between before and "
    "after — it can be any motion: head turning or tilting up/down, a leg "
    "swinging, the spine bending, the tail moving, etc. "
    "Judge the motion and answer with exactly ONE token:\n"
    "yes — natural articulation, moved toward the reference, and did NOT "
    "overshoot past the reference position.\n"
    "too_far — the RIGHT body part moved TOWARD the reference, but the "
    "rotation was too large: it went PAST the pose in the reference "
    "(e.g. reference bows 60 degrees, the move bent to 90). A smaller "
    "rotation in the SAME direction would be correct.\n"
    "no_wrong_direction — the RIGHT body part moved AWAY from the reference, "
    "opposite to where it should go (e.g. reference bows forward, the move "
    "leans backward). Flipping the rotation would help.\n"
    "no_wrong_axis — the RIGHT body part moved, but around the wrong axis "
    "(e.g. it needs to bend forward but instead twisted sideways). Neither "
    "this rotation nor its opposite helps; a different axis would.\n"
    "no_wrong_joint — anything else: wrong body part, unnatural distortion, "
    "stretching or dragging, or motion unrelated to the reference.\n"
    "Judge ONLY the body part that changed between BEFORE and AFTER. Other "
    "parts may already be misplaced; that is not this move's fault and must "
    "not make you reject it.\n"
    "Output a single token only: yes, too_far, no_wrong_direction, "
    "no_wrong_axis, or no_wrong_joint."
)

ROOT_VERIFY_SYSTEM = (
    "You are a 3D pose alignment verifier. You will see three images: BEFORE, "
    "AFTER, and the REFERENCE. Between before and after the whole character "
    "was TRANSLATED as a rigid body — its pose did not change at all, only "
    "where it stands in the frame. "
    "Judge only the placement and answer with exactly ONE token:\n"
    "yes — now stands closer to where it stands in the reference, no overshoot.\n"
    "too_far — moved the right way but past the reference's position.\n"
    "no_wrong_direction — moved away from the reference's position.\n"
    "no_wrong_joint — no visible change, or the change is not a translation.\n"
    "Ignore the pose entirely; judge position only. Output a single token."
)


PAIR_SYSTEM = (
    "You are a 3D pose alignment judge. You will see three images: OPTION A, "
    "OPTION B, and the REFERENCE. A and B are the same character posed by "
    "rotating ONE joint the same amount in OPPOSITE directions — for example "
    "the torso folded forward in one and leaned backward in the other.\n"
    "Say which option's CHANGED PART is closer to the corresponding part of "
    "the reference pose. Ignore every other part of the body: they are "
    "identical in A and B, and any of them may already be misplaced.\n"
    "Answer with exactly one token: A, B, or neither.\n"
    "Use 'neither' only when the changed part is equally far off in both — "
    "not merely because both look imperfect."
)


def ask_claude_compare_pair(a_path, b_path, reference_path,
                            client, claude_model):
    """Which of two opposite rotations is closer to the reference: A, B, or neither.

    Asked instead of judging one move in isolation against a five-label
    taxonomy. The taxonomy is where the search kept dying: `Spine02 z+` drew
    `no_wrong_direction` under FB-04-UTM (which fires FLIP and reaches the bow)
    but `no_wrong_joint` under FB-07 (which fires nothing and ends the line) --
    same joint, same magnitude, opposite outcomes, decided by a label. A
    forced choice between two rendered alternatives removes the taxonomy from
    the critical path, and it is the exact question the silhouette cannot
    answer: front or back.
    """
    response = client.messages.create(
        model=claude_model,
        max_tokens=1024,
        system=PAIR_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "OPTION A:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": _img_to_b64(a_path)}},
                {"type": "text", "text": "OPTION B:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": _img_to_b64(b_path)}},
                {"type": "text", "text": "REFERENCE:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": _img_to_b64(reference_path)}},
                {"type": "text", "text": "Which option's changed part is closer "
                                         "to the reference? Answer A, B, or neither."},
            ],
        }],
    )
    raw = _answer_text(response)
    if not raw:
        return "neither", "(no text block)"
    tail = raw.splitlines()[-1].strip().lower().strip(".*_ ")
    for probe in (tail, raw.lower()):
        if probe.startswith("a") and not probe.startswith("and"):
            return "A", raw
        if probe.startswith("b"):
            return "B", raw
        if "neither" in probe:
            return "neither", raw
    return "neither", raw


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _answer_text(response) -> str:
    """Concatenate the text blocks of a reply, skipping non-text ones.

    `response.content[0].text` is wrong: the first block can be a
    ThinkingBlock, which carries `.thinking` and no `.text`. That is not an
    edge case here -- the harder the judgement, the likelier the model thinks
    first, so the LARGE Spine02 moves (the ones that decide whether the figure
    bows at all) were exactly the ones that raised AttributeError and got
    filed as api_error, three iterations running, while the small limb moves
    answered in plain text and went through.
    """
    return "\n".join(b.text for b in response.content
                     if getattr(b, "type", None) == "text").strip()


def ask_claude_verify_move(before_path, after_path, reference_path,
                           client, claude_model, is_root=False):
    """Returns (verdict, raw_text).

    Verdicts: yes / too_far / no_wrong_direction / no_wrong_axis /
              no_wrong_joint / unparsed.
    `unparsed` exists so a reply the parser does not recognise can never be
    filed as a rejection -- FB-05's first run read 23 transport failures as 23
    rejections and reported "converged".
    """
    question = ("Did the character's POSITION move toward its position in the "
                "reference? Answer with one token: yes, too_far, "
                "no_wrong_direction, or no_wrong_joint."
                if is_root else
                "Answer with one token: yes, too_far, no_wrong_direction, "
                "no_wrong_axis, or no_wrong_joint.")
    response = client.messages.create(
        model=claude_model,
        # Must leave room for a thinking block AND the one-word answer. At
        # max_tokens=16 a thinking model can spend the whole budget before
        # emitting any text, and the verdict comes back empty.
        max_tokens=1024,
        system=ROOT_VERIFY_SYSTEM if is_root else VERIFY_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "BEFORE:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": _img_to_b64(before_path)}},
                {"type": "text", "text": "AFTER:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": _img_to_b64(after_path)}},
                {"type": "text", "text": "REFERENCE:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": _img_to_b64(reference_path)}},
                {"type": "text", "text": question},
            ],
        }],
    )
    raw = _answer_text(response)
    if not raw:
        return "unparsed", f"(no text block; blocks={[b.type for b in response.content]})"
    # A thinking model may reason first and answer last, so try the final line
    # before falling back to the whole reply.
    tail = raw.splitlines()[-1].strip().lower()
    for probe in (tail, raw.lower()):
        if probe.startswith("yes"):
            return "yes", raw
        if "too_far" in probe:
            return "too_far", raw
        if "wrong_direction" in probe:
            return "no_wrong_direction", raw
        if "wrong_axis" in probe:
            return "no_wrong_axis", raw
        if "wrong_joint" in probe:
            return "no_wrong_joint", raw
    return "unparsed", raw


# ==============================================================================
# REFERENCE FRAME
# ==============================================================================

def extract_frame_at_second(video_path, output_img_path, second):
    if os.path.exists(output_img_path):
        print("Reference frame already exists. Skipping extraction.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame at {second}s from {video_path}")
    cv2.imwrite(output_img_path, frame)
    cap.release()
    print(f"Extracted reference frame at {second}s -> {output_img_path}")


def reference_silhouette(reference_path, size):
    img = cv2.imread(reference_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read reference image: {reference_path}")
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    bg = int(np.median(corners))
    if bg > 127:
        _, mask = cv2.threshold(gray, bg - 40, 255, cv2.THRESH_BINARY_INV)
    else:
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return mask > 0


# ==============================================================================
# JOINTS
# ==============================================================================

def _build_child_to_parent(hierarchy):
    return {child: parent for parent, kids in hierarchy.items() for child in kids}


def get_adjustable_joints(model):
    joints_with_children = set(model.hierarchy.keys())
    return {j for j in model.joint_rotations
            if "_dup" not in j.lower() and "twist" not in j.lower()
            and j in joints_with_children}


def nearest_adjustable_ancestor(joint, adjustable, c2p):
    cur = joint
    while cur is not None:
        if cur in adjustable:
            return cur
        cur = c2p.get(cur)
    return None


def adjustable_descendants(model, joint, adjustable):
    """Nearest adjustable children, hopping over filtered bones.

    Spine02's children are NeckTwist01, L_Clavicle, R_Clavicle. The Twist bone
    is filtered out of the adjustable set, so the search has to descend through
    it or the neck chain stays unreachable.
    """
    out, stack = set(), list(model.hierarchy.get(joint, []))
    while stack:
        c = stack.pop()
        if c in adjustable:
            out.add(c)
        else:
            stack.extend(model.hierarchy.get(c, []))
    return out


# ==============================================================================
# RASTERIZATION / ERROR ATTRIBUTION
# ==============================================================================

def compute_face_joint_ids(model, joint_to_id, c2p, adjustable):
    V = model.mesh.verts_packed().shape[0]
    vert_ids = np.full(V, -1, dtype=np.int32)
    for v_idx, weights in model.skin_weights.items():
        if not weights:
            continue
        dominant = max(weights, key=lambda x: x[1])[0]
        resolved = nearest_adjustable_ancestor(dominant, adjustable, c2p)
        if resolved is not None:
            vert_ids[v_idx] = joint_to_id[resolved]
    faces = model.mesh.faces_packed().cpu().numpy()
    a, b, c = vert_ids[faces[:, 0]], vert_ids[faces[:, 1]], vert_ids[faces[:, 2]]
    face_ids = a.copy()
    face_ids[b == c] = b[b == c]
    return face_ids


def rasterize_id_map(model, rasterizer, face_joint_ids):
    with torch.no_grad():
        fragments = rasterizer(model())
    pix_to_face = fragments.pix_to_face[0, :, :, 0].cpu().numpy()
    fg = pix_to_face >= 0
    id_img = np.full(pix_to_face.shape, -1, dtype=np.int32)
    id_img[fg] = face_joint_ids[pix_to_face[fg]]
    return id_img, fg


def build_assignment_map(id_img, fg_mask):
    if not fg_mask.any():
        raise RuntimeError("Rasterized silhouette is empty — camera/mesh mismatch.")
    src = np.where(fg_mask, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(fg_mask)
    lut = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
    lut[labels[ys, xs]] = id_img[ys, xs]
    return lut[labels]


def silhouette_of_current(model, rasterizer):
    with torch.no_grad():
        fragments = rasterizer(model())
    return (fragments.pix_to_face[0, :, :, 0] >= 0).cpu().numpy()


def rank_joints(assign_map, false_map, num_joints):
    ids = assign_map[false_map]
    ids = ids[ids >= 0]
    counts = np.bincount(ids, minlength=num_joints)
    return [(int(j), int(counts[j])) for j in np.argsort(-counts)]


# ==============================================================================
# STATE
# ==============================================================================

def root_param(model):
    roots = sorted(model.root_joints)
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one root joint, got {roots}")
    return getattr(model, f"root_transform_{roots[0]}")


def apply_root_translation(model, delta_xyz):
    p = root_param(model)
    p.data[:3] += torch.tensor(delta_xyz, dtype=p.dtype, device=p.device)


def save_joint_state(model):
    """Rotations AND the root transform. Both must round-trip, or the candidate
    sweep leaks trial translations into the accepted pose."""
    return ({k: v.data.clone() for k, v in model.joint_rotations.items()},
            root_param(model).data.clone())


def restore_joint_state(model, state):
    rots, root = state
    for k, v in rots.items():
        model.joint_rotations[k].data.copy_(v)
    root_param(model).data.copy_(root)


def state_signature(model):
    """Fingerprint of the full pose, used to memoise verdicts.

    Two iterations sharing a signature would enumerate identical candidates and
    ask identical questions -- the repeat that burned ~15 calls per stall in
    every predecessor.
    """
    parts = [model.joint_rotations[k].data for k in sorted(model.joint_rotations)]
    parts.append(root_param(model).data)
    v = torch.cat([p.flatten() for p in parts]).detach().cpu().numpy()
    return hashlib.blake2b(np.round(v, 5).tobytes(), digest_size=12).hexdigest()


def apply_axis_delta(model, joint, delta_xyz):
    p = model.joint_rotations[joint]
    delta_t = torch.tensor(delta_xyz, dtype=p.dtype, device=p.device)
    R_delta = axis_angle_to_matrix(delta_t)
    p.data.copy_(matrix_to_rotation_6d(R_delta @ rotation_6d_to_matrix(p.data)))


def apply_move(model, jname, delta_xyz):
    """Single dispatch point, so the candidate sweep and the accept path can
    never disagree about what a move means."""
    if jname == ROOT_KEY:
        apply_root_translation(model, delta_xyz)
    else:
        apply_axis_delta(model, jname, delta_xyz)


# ==============================================================================
# STAGE 0: GLOBAL ALIGNMENT  (from FB-05)
# ==============================================================================

def largest_component(mask):
    """Threshold speckle at the frame border would drag the feet anchor far
    off; keep only the character."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def feet_anchor(mask):
    """Centroid of the bottom 8% of the silhouette bbox.

    Pose-invariant where it counts: the feet stay planted through a bow, so
    this measures where the character STANDS, not how it is bent. The global
    XOR optimum lacks that property -- measured, it wants dy=-20 where the true
    foot offset is +9.4, buying the difference with a fake bend.
    """
    m = largest_component(mask)
    ys, xs = np.nonzero(m)
    y0, y1 = ys.min(), ys.max()
    band = m.copy()
    band[:int(y1 - 0.08 * (y1 - y0)), :] = False
    fys, fxs = np.nonzero(band)
    return np.array([fxs.mean(), fys.mean()])


def calibrate_pixel_jacobian(model, rasterizer, eps=0.05):
    """J[:, i] = centroid pixel shift per unit world translation on axis i.

    Measured with three probes rather than derived: PyTorch3D uses a row-vector
    convention (X_cam = X_world @ R + T) and `world_to_image_space` negates the
    projection, so a hand-derived mapping is easy to sign-flip and hard to
    notice.
    """
    def centroid():
        ys, xs = np.nonzero(silhouette_of_current(model, rasterizer))
        return np.array([xs.mean(), ys.mean()])

    base = centroid()
    J = np.zeros((2, 3))
    for i in range(3):
        d = [0.0, 0.0, 0.0]
        d[i] = eps
        apply_root_translation(model, d)
        J[:, i] = (centroid() - base) / eps
        apply_root_translation(model, [-x for x in d])
    return J


def align_root_to_reference(model, rasterizer, ref_mask, J,
                            iters=4, tol=0.5, log=None):
    """Move the mesh -- never the reference image -- so the feet land where the
    reference's feet are. Closed form, no search, no API calls."""
    target = feet_anchor(ref_mask)
    applied = np.zeros(3)
    for k in range(iters):
        d_pix = target - feet_anchor(silhouette_of_current(model, rasterizer))
        err = float(np.linalg.norm(d_pix))
        msg = (f"  stage0 iter {k}: feet offset = "
               f"({d_pix[0]:+.2f}, {d_pix[1]:+.2f}) px  |err|={err:.2f}")
        print(msg)
        if log:
            log.write(msg + "\n")
        if err < tol:
            break
        # lstsq's minimum-norm solution is orthogonal to J's null space -- the
        # viewing ray -- so the correction stays in the camera plane.
        t, *_ = np.linalg.lstsq(J, d_pix, rcond=None)
        apply_root_translation(model, tuple(t))
        applied += t
    return applied


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def save_assignment_vis(assign_map, num_joints, out_path):
    h, w = assign_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for jid in range(num_joints):
        r, g, b = colorsys.hsv_to_rgb(jid / max(num_joints, 1), 0.9, 0.95)
        vis[assign_map == jid] = (int(b * 255), int(g * 255), int(r * 255))
    cv2.imwrite(out_path, vis)


def save_xor_vis(fg_mask, ref_mask, out_path):
    h, w = fg_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[fg_mask & ~ref_mask] = (0, 0, 255)
    vis[ref_mask & ~fg_mask] = (0, 255, 0)
    cv2.imwrite(out_path, vis)


def render_rgb_to_png(model, renderer, camera, lights, out_path):
    with torch.no_grad():
        images = renderer.render(model(), camera, lights,
                                 background_color=torch.tensor([1.0, 1.0, 1.0]))
    rgb = (images[0, :, :, :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# ==============================================================================
# MAIN LOOP
# ==============================================================================

def run_agent(model, renderer, camera, lights, reference_path, output_dir,
              image_size, claude_client, claude_model, max_iter, patience,
              max_verify, root_in_pool, tolerance, max_regress):

    for old in _glob.glob(os.path.join(output_dir, "iter_*.png")):
        os.remove(old)

    log = open(os.path.join(output_dir, "agent_log.md"), "w", encoding="utf-8")
    log.write(f"# Agent FB-07 Log\n\n"
              f"steps={STEP_LEVELS} | root_steps={ROOT_STEP_LEVELS} | "
              f"root_in_pool={root_in_pool} | tolerance={tolerance} | "
              f"max_regress={max_regress} | max_iter={max_iter} | "
              f"patience={patience} | max_verify={max_verify}\n\n")

    adjustable = get_adjustable_joints(model)
    joint_names = sorted(adjustable)
    joint_to_id = {n: i for i, n in enumerate(joint_names)}
    num_joints = len(joint_names)
    c2p = _build_child_to_parent(model.hierarchy)
    print(f"Adjustable joints ({num_joints}): {joint_names}")

    face_joint_ids = compute_face_joint_ids(model, joint_to_id, c2p, adjustable)
    rasterizer = MeshRasterizer(
        cameras=camera,
        raster_settings=RasterizationSettings(image_size=image_size,
                                              blur_radius=0.0,
                                              faces_per_pixel=1))
    ref_mask = reference_silhouette(reference_path, image_size)

    # ---- Stage 0 -------------------------------------------------------------
    render_rgb_to_png(model, renderer, camera, lights,
                      os.path.join(output_dir, "initial_pose_prealign.png"))
    fg_pre = silhouette_of_current(model, rasterizer)
    score_pre = int((fg_pre ^ ref_mask).sum())
    save_xor_vis(fg_pre, ref_mask, os.path.join(output_dir, "xor_prealign.png"))

    print(f"\n{'='*60}\nStage 0: global alignment\n{'='*60}")
    log.write(f"## Stage 0: global alignment\n\nXOR before: **{score_pre}**\n\n```\n")
    J = calibrate_pixel_jacobian(model, rasterizer)
    px = np.linalg.norm(J, axis=0)
    print(f"  Jacobian px/world-unit: x={px[0]:.1f} y={px[1]:.1f} z={px[2]:.1f}")
    log.write(f"Jacobian px per world unit: x={px[0]:.1f} y={px[1]:.1f} z={px[2]:.1f}\n")
    log.write(f"root steps in px: {[round(s*float(px.max()),1) for s in ROOT_STEP_LEVELS]}\n")
    applied = align_root_to_reference(model, rasterizer, ref_mask, J, log=log)
    log.write("```\n\n")

    fg0 = silhouette_of_current(model, rasterizer)
    total_initial = int((fg0 ^ ref_mask).sum())
    gain = score_pre - total_initial
    print(f"  applied ({applied[0]:+.4f}, {applied[1]:+.4f}, {applied[2]:+.4f}) "
          f"world units")
    print(f"  XOR {score_pre} -> {total_initial} "
          f"(-{gain}, -{100.0*gain/max(score_pre,1):.1f}%)")
    log.write(f"root translation: ({applied[0]:+.4f}, {applied[1]:+.4f}, "
              f"{applied[2]:+.4f})\n\n**XOR {score_pre} -> {total_initial} "
              f"(-{gain}, -{100.0*gain/max(score_pre,1):.1f}%), "
              f"zero joint moves, zero API calls**\n\n---\n\n")

    render_rgb_to_png(model, renderer, camera, lights,
                      os.path.join(output_dir, "initial_pose.png"))
    save_xor_vis(fg0, ref_mask, os.path.join(output_dir, "xor_initial.png"))
    id0, _ = rasterize_id_map(model, rasterizer, face_joint_ids)
    save_assignment_vis(build_assignment_map(id0, fg0), num_joints,
                        os.path.join(output_dir, "assignment_map_initial.png"))

    best_state = save_joint_state(model)
    best_score = total_initial

    cumulative = {}
    accepted_counter = 0
    no_improve_streak = 0
    regress_streak = 0
    api_calls = api_errors = memo_hits = 0
    verdict_memo = {}        # (sig, joint, dir, step) -> verdict
    forced_joints = set()    # FK dependents of the last accepted move

    for it in range(1, max_iter + 1):
        print(f"\n{'='*60}\nIteration {it}/{max_iter}\n{'='*60}")
        log.write(f"## Iteration {it}\n\n")

        sig = state_signature(model)
        id_img, fg_mask = rasterize_id_map(model, rasterizer, face_joint_ids)
        assign_map = build_assignment_map(id_img, fg_mask)
        false_map = fg_mask ^ ref_mask
        base_score = int(false_map.sum())
        ranking = rank_joints(assign_map, false_map, num_joints)

        window = 3 + 3 * no_improve_streak
        picked = [joint_names[j] for j, c in ranking[:window] if c > 0]
        for f in sorted(forced_joints):
            if f not in picked:
                picked.append(f)

        if not picked:
            print("No error assigned to any joint. Done.")
            log.write("No error assigned to any joint. Done.\n\n")
            break

        print(f"base={base_score} window={window} joints={picked}"
              + (f" (forced: {sorted(forced_joints)})" if forced_joints else ""))
        log.write(f"base={base_score}, window={window}, joints={picked}, "
                  f"forced={sorted(forced_joints)}\n\n")

        # ---- enumerate ------------------------------------------------------
        candidates = []
        state = save_joint_state(model)
        for jname in picked:
            for dname, axis in DIRECTIONS:
                for s in STEP_LEVELS:
                    apply_axis_delta(model, jname, tuple(a * s for a in axis))
                    fg_c = silhouette_of_current(model, rasterizer)
                    candidates.append((int((fg_c ^ ref_mask).sum()),
                                       jname, dname, s))
                    restore_joint_state(model, state)
        if root_in_pool:
            # Root is its own class, never a ranked joint: it moves every pixel,
            # so on the assignment map it would outrank every limb and starve
            # the arms of candidate slots -- the failure that let FB-04's arms
            # ride the spine backward uncorrected.
            for dname, axis in DIRECTIONS:
                for s in ROOT_STEP_LEVELS:
                    apply_root_translation(model, tuple(a * s for a in axis))
                    fg_c = silhouette_of_current(model, rasterizer)
                    candidates.append((int((fg_c ^ ref_mask).sum()),
                                       ROOT_KEY, dname, s))
                    restore_joint_state(model, state)

        candidates.sort(key=lambda x: x[0])
        log.write(f"best 5 candidates: {candidates[:5]}\n\n")

        improving = [c for c in candidates if c[0] < base_score]
        ceiling = base_score * (1.0 + tolerance)
        tolerated = [c for c in candidates
                     if base_score <= c[0] <= ceiling
                     and c[3] >= (ROOT_VERIFY_MIN_STEP if c[1] == ROOT_KEY
                                  else VERIFY_MIN_STEP)]

        if not improving and not tolerated:
            print("No candidate within tolerance. Converged.")
            log.write("No candidate within tolerance. Converged.\n\n")
            break

        # ---- review queue: one slot per (joint, direction), best step -------
        before_snap = os.path.join(output_dir, "_before.png")
        render_rgb_to_png(model, renderer, camera, lights, before_snap)

        best_per_dir = {}
        for score, jname, dname, s in improving + tolerated:
            key = (jname, dname)
            if key not in best_per_dir or score < best_per_dir[key][0]:
                best_per_dir[key] = (score, jname, dname, s)
        queue = sorted(best_per_dir.values())

        accepted = None
        verdicts = []
        reviewed = set()
        reviews_used = 0

        def measured(j, d, step):
            """True silhouette score of (j, d, step) from the current state."""
            m = tuple(a * step for a in AXIS_OF[d])
            apply_move(model, j, m)
            v = int((silhouette_of_current(model, rasterizer) ^ ref_mask).sum())
            restore_joint_state(model, state)
            return v

        while queue and reviews_used < max_verify:
            score, jname, dname, s = queue.pop(0)
            if (jname, dname, s) in reviewed:
                continue
            reviewed.add((jname, dname, s))

            is_root = jname == ROOT_KEY
            floor = ROOT_VERIFY_MIN_STEP if is_root else VERIFY_MIN_STEP
            steps_for = ROOT_STEP_LEVELS if is_root else STEP_LEVELS
            delta = tuple(a * s for a in AXIS_OF[dname])
            apply_move(model, jname, delta)

            # Unverified moves must strictly improve. A regressive move is only
            # ever taken on Claude's say-so.
            if s < floor:
                if score < base_score:
                    verdicts.append((jname, dname, s, score, "auto(small step)", ""))
                    accepted = (score, jname, dname, s, delta)
                    break
                restore_joint_state(model, state)
                continue

            memo_key = (sig, jname, dname, s)
            if memo_key in verdict_memo:
                verdict, raw = verdict_memo[memo_key], "(memo)"
                memo_hits += 1
            else:
                after_snap = os.path.join(output_dir, "_after.png")
                render_rgb_to_png(model, renderer, camera, lights, after_snap)
                try:
                    verdict, raw = ask_claude_verify_move(
                        before_snap, after_snap, reference_path,
                        claude_client, claude_model, is_root=is_root)
                except Exception as e:
                    # An unreachable API is not a judgement about the pose.
                    verdict, raw = "api_error", f"{type(e).__name__}: {e}"
                    print(f"  [API ERROR] {raw}")
                    api_errors += 1
                api_calls += 1
                reviews_used += 1
                if verdict != "api_error":
                    verdict_memo[memo_key] = verdict

            tag = "REGRESS" if score >= base_score else ""
            verdicts.append((jname, dname, s, score, verdict, raw[:60], tag))

            if verdict == "yes":
                if score >= base_score and regress_streak >= max_regress:
                    print(f"  regression budget spent ({regress_streak}), skipping")
                    restore_joint_state(model, state)
                    continue
                accepted = (score, jname, dname, s, delta)
                break

            restore_joint_state(model, state)

            # FLIP (FB-04-UTM) -- the rule that reached Spine02 z-. The flipped
            # direction goes to the FRONT of the queue and is judged on its own
            # merits, so a worse-SCORING but semantically right move can win.
            if verdict == "no_wrong_direction":
                fd = FLIP[dname]
                if (jname, fd, s) not in reviewed:
                    flipped = best_per_dir.get((jname, fd))
                    if flipped is None:
                        flipped = (measured(jname, fd, s), jname, fd, s)
                    queue.insert(0, flipped)

            # AXIS (FB-06) -- right body part, wrong axis. FLIP fixes the sign;
            # this fixes the axis. Both other axes go to the front, cheapest
            # first, each still individually reviewed.
            elif verdict == "no_wrong_axis":
                alts = []
                for alt_d, _ in DIRECTIONS:
                    if alt_d[0] == dname[0]:
                        continue
                    if (jname, alt_d, s) in reviewed:
                        continue
                    cand = best_per_dir.get((jname, alt_d))
                    if cand is None:
                        cand = (measured(jname, alt_d, s), jname, alt_d, s)
                    alts.append(cand)
                for cand in sorted(alts, reverse=True):
                    queue.insert(0, cand)

            elif verdict == "too_far":
                i = steps_for.index(s)
                if i + 1 < len(steps_for):
                    s2 = steps_for[i + 1]
                    if (jname, dname, s2) not in reviewed:
                        queue.insert(0, (measured(jname, dname, s2),
                                         jname, dname, s2))

        print(f"verdicts: {verdicts}")
        log.write(f"verdicts: {verdicts}\n\n")

        if api_calls >= 3 and api_errors == api_calls:
            msg = (f"ABORT: all {api_calls} verifier calls failed "
                   f"(model={claude_model!r}). The search is meaningless "
                   f"without the verifier -- the silhouette alone cannot tell "
                   f"a forward bow from a backward lean.")
            print(f"\n{msg}")
            log.write(f"**{msg}**\n\n")
            break

        if accepted is None:
            no_improve_streak += 1
            forced_joints = set()
            print(f"Nothing accepted. streak={no_improve_streak} "
                  f"(window widens to {3 + 3*no_improve_streak})")
            log.write(f"Nothing accepted. streak={no_improve_streak}\n\n")
            if no_improve_streak >= patience:
                print("Converged: patience exhausted.")
                log.write("Converged: patience exhausted.\n\n")
                break
            continue

        # ---- accept ---------------------------------------------------------
        score, jname, dname, s, delta = accepted
        no_improve_streak = 0
        regress_streak = regress_streak + 1 if score >= base_score else 0
        cumulative.setdefault(jname, [0.0, 0.0, 0.0])
        for i in range(3):
            cumulative[jname][i] += delta[i]
        accepted_counter += 1

        # Whatever FK just dragged along gets a candidate slot next round.
        forced_joints = (set() if jname == ROOT_KEY
                         else adjustable_descendants(model, jname, adjustable))

        if score < best_score:
            best_score = score
            best_state = save_joint_state(model)

        note = f" [REGRESS {regress_streak}/{max_regress}]" if score >= base_score else ""
        print(f"ACCEPTED: {jname} {dname} step={s} -> {score} "
              f"(best {best_score}){note}")
        log.write(f"**ACCEPTED:** {jname} {dname} step={s} -> {score} "
                  f"(best {best_score}){note}; next-round forced: "
                  f"{sorted(forced_joints)}\n\n---\n\n")
        render_rgb_to_png(model, renderer, camera, lights,
                          os.path.join(output_dir,
                                       f"iter_{it:03d}_{jname}_{dname}_{s}.png"))

    # ---- finish --------------------------------------------------------------
    now = int((silhouette_of_current(model, rasterizer) ^ ref_mask).sum())
    if best_score < now:
        restore_joint_state(model, best_state)
        print(f"Rolled back to best state: {now} -> {best_score}")
        log.write(f"Rolled back to best state: {now} -> {best_score}\n\n")

    fg_final = silhouette_of_current(model, rasterizer)
    total_final = int((fg_final ^ ref_mask).sum())
    save_xor_vis(fg_final, ref_mask, os.path.join(output_dir, "xor_final.png"))
    render_rgb_to_png(model, renderer, camera, lights,
                      os.path.join(output_dir, "final_pose.png"))
    with open(os.path.join(output_dir, "cumulative_deltas.json"), "w") as f:
        json.dump(cumulative, f, indent=2)

    summary = (
        f"XOR {score_pre} -> {total_initial} (stage 0: -{score_pre-total_initial}) "
        f"-> {total_final} (joints: -{total_initial-total_final}); "
        f"total -{100.0*(score_pre-total_final)/max(score_pre,1):.1f}%, "
        f"{accepted_counter} accepted moves, "
        f"{api_calls-api_errors}/{api_calls} verifier calls ok, "
        f"{memo_hits} memo hits saved"
    )
    print(f"\n{summary}")
    log.write(f"---\n\n**{summary}**\n")
    log.close()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    set_seed(42)
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_path",  default="asset/ultraman_texture_obj/ultraman_texture.obj")
    p.add_argument("--rig_path",   default="asset/ultraman_texture_obj/rig/ultraman_ori_rig.txt")
    p.add_argument("--video_path", default="asset/ultraman_texture_obj/videos/bow_seq.mp4")
    p.add_argument("--output_dir", default="agent_output_fb07")
    p.add_argument("--api_key",    default=os.environ.get("ANTHROPIC_API_KEY", ""))
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--second",     type=float, default=2.0)
    p.add_argument("--max_iter",   type=int, default=30)
    p.add_argument("--patience",   type=int, default=3)
    p.add_argument("--max_verify", type=int, default=5)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--cam_dist",   type=float, default=1.25)
    p.add_argument("--cam_elev",   type=float, default=10)
    p.add_argument("--cam_azim",   type=float, default=75)
    p.add_argument("--claude_model", default="claude-sonnet-5")
    p.add_argument("--root_in_pool", action="store_true",
                   help="keep root translation live after stage 0")
    p.add_argument("--tolerance", type=float, default=0.06,
                   help="fraction above the current score a candidate may sit "
                        "and still be reviewed. 0 reproduces FB-04/FB-05, "
                        "which could never approve a move that gets worse "
                        "before it gets better -- e.g. bowing deeper while the "
                        "arms are still swung back.")
    p.add_argument("--max_regress", type=int, default=2,
                   help="consecutive score-worsening accepts allowed")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY or pass --api_key.")

    os.makedirs(args.output_dir, exist_ok=True)
    ref_frame_path = os.path.join(args.output_dir, "reference.png")
    extract_frame_at_second(args.video_path, ref_frame_path, second=args.second)

    device = args.device
    mesh = load_mesh(args.mesh_path, load_materials=True, device=device)
    joints, skin_weights, hierarchy = parse_rig_file(args.rig_path)
    model = RiggingModel(mesh, joints, skin_weights, hierarchy).to(device)
    model.eval()

    camera = get_camera(dist=args.cam_dist, elev=args.cam_elev,
                        azim=args.cam_azim, device=device)
    lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]],
                         ambient_color=((0.95, 0.95, 0.95),),
                         diffuse_color=((0.05, 0.05, 0.05),),
                         specular_color=((0.0, 0.0, 0.0),))
    renderer = Renderer(image_size=args.image_size, device=device)

    client = anthropic.Anthropic(api_key=args.api_key, timeout=30.0)

    # Smoke-test the verifier before spending compute: without it the search is
    # not just weaker, it is aimed at the wrong bend direction.
    print(f"Checking verifier model {args.claude_model!r} ...", end=" ")
    try:
        client.messages.create(model=args.claude_model, max_tokens=4,
                               messages=[{"role": "user", "content": "Reply ok."}])
        print("reachable.")
    except UnicodeEncodeError:
        raise SystemExit(
            "\nThe API key contains non-ASCII characters, so it cannot go in an "
            "HTTP header — almost always a placeholder that was never replaced "
            "(e.g. the literal text 你的key). Set the real key:\n"
            '  setx ANTHROPIC_API_KEY "sk-ant-..."\n'
            "then reopen the terminal.")
    except anthropic.AuthenticationError as e:
        raise SystemExit(f"\nAPI key rejected: {e}")
    except anthropic.APIConnectionError as e:
        raise SystemExit(
            f"\nCannot reach the API: {e}\n"
            "Transport problem, not a model problem. Behind a proxy, this shell "
            "needs it too:\n"
            "  set HTTP_PROXY=http://127.0.0.1:10808\n"
            "  set HTTPS_PROXY=http://127.0.0.1:10808\n"
            "Verify with: echo %HTTPS_PROXY%")
    except Exception as e:
        raise SystemExit(f"\nmodel {args.claude_model!r} unusable: "
                         f"{type(e).__name__}: {e}\nTry --claude_model "
                         f"claude-opus-4-8 or claude-haiku-4-5-20251001.")

    run_agent(model=model, renderer=renderer, camera=camera, lights=lights,
              reference_path=ref_frame_path, output_dir=args.output_dir,
              image_size=args.image_size, claude_client=client,
              claude_model=args.claude_model, max_iter=args.max_iter,
              patience=args.patience, max_verify=args.max_verify,
              root_in_pool=args.root_in_pool, tolerance=args.tolerance,
              max_regress=args.max_regress)
