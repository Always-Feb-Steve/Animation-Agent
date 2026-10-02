"""
measure.py -- numeric checks and solvers the animator can call instead of
eyeballing renders.

  BodyChecks.collisions()  body parts that intersect (arm inside head, ...)
  BodyChecks.balance()     centre of mass outside the feet's support area
  solve_ik()               joint angles that put a joint at a target point
  Mirror                   left/right copy or swap of a pose

All of them work on any rig: body parts are the adjustable joints' skin
regions, and left/right pairs are found geometrically, not by name.
"""

import math

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_rotation_6d, rotation_6d_to_matrix)

import agent_fb_07 as fb07


UP = np.array([0.0, 1.0, 0.0])


def _hops(c2p, a, b):
    """Tree distance between two joints."""
    def chain(j):
        out = []
        while j is not None:
            out.append(j)
            j = c2p.get(j)
        return out
    ca, cb = chain(a), chain(b)
    common = set(ca) & set(cb)
    if not common:
        return 99
    return min(ca.index(c) + cb.index(c) for c in common)


class BodyChecks:
    """Collision and balance checks for one character. Built once at rest,
    because 'touching at rest' (armpits, thighs) is not a collision."""

    def __init__(self, model, adjustable, height, names=None):
        self.model = model
        self.height = height
        self.names = names or {}
        c2p = fb07._build_child_to_parent(model.hierarchy)
        V = model.mesh.verts_packed().shape[0]
        owner = np.full(V, -1)
        self.parts = sorted(adjustable)
        pid = {j: i for i, j in enumerate(self.parts)}
        for v, ws in model.skin_weights.items():
            if ws:
                o = fb07.nearest_adjustable_ancestor(max(ws, key=lambda x: x[1])[0],
                                                     adjustable, c2p)
                if o is not None:
                    owner[v] = pid[o]
        self.owner = owner
        # Leaf bones skinned into a part (panda's Head has no adjustable joint
        # of its own and resolves to Spine02); named in reports so "Spine02 is
        # inside L_Forearm" reads as the head it really is.
        self.extra = {}
        for j in model.joints:
            if j in adjustable or "twist" in j.lower() or "_dup" in j.lower():
                continue
            o = fb07.nearest_adjustable_ancestor(j, adjustable, c2p)
            if o is not None and not model.hierarchy.get(j):
                self.extra.setdefault(o, []).append(j)
        n = len(self.parts)
        # Parts within 2 hops share a joint region (upper arm / shoulder /
        # chest); their surfaces meet by construction.
        self.near = np.array([[_hops(c2p, self.parts[i], self.parts[k]) <= 2
                               for k in range(n)] for i in range(n)])
        self.rest_pairs = self._inside_pairs()

    def _label(self, j):
        if j in self.names:
            return f"{j} ({self.names[j]})"
        extra = sorted(self.extra.get(j, []))[:2]
        return f"{j} [+{', '.join(extra)}]" if extra else j

    def _inside_pairs(self):
        """{(a, b): count} of vertices of part a lying behind part b's surface."""
        with torch.no_grad():
            m = self.model()
        v = m.verts_packed().cpu().numpy()
        nrm = m.verts_normals_packed().cpu().numpy()
        tree = cKDTree(v)
        r = 0.06 * self.height
        dist, idx = tree.query(v, k=8, distance_upper_bound=r)
        pairs = {}
        for a in range(v.shape[0]):
            pa = self.owner[a]
            if pa < 0:
                continue
            for d, b in zip(dist[a], idx[a]):
                if not np.isfinite(d):
                    break
                pb = self.owner[b]
                if pb < 0 or pb == pa or self.near[pa, pb]:
                    continue
                # a is inside b's part if it sits behind b's outward normal.
                if (v[a] - v[b]) @ nrm[b] < -0.004 * self.height:
                    key = (pa, pb)
                    pairs[key] = pairs.get(key, 0) + 1
                    break
        return pairs

    def collisions(self, min_verts=6):
        """Lines like 'R_Hand (right paw) is inside Head (head): 23 vertices'."""
        found = []
        for (a, b), c in self._inside_pairs().items():
            if c >= min_verts and c > 3 * self.rest_pairs.get((a, b), 0):
                found.append((c, a, b))
        found.sort(reverse=True)
        return [f"{self._label(self.parts[a])} is inside {self._label(self.parts[b])}: "
                f"{c} vertices" for c, a, b in found[:4]]

    def balance(self, frame):
        """None if balanced, else a sentence on where the mass hangs out."""
        with torch.no_grad():
            v = self.model().verts_packed().cpu().numpy()
        up = v @ UP
        foot = up < up.min() + 0.02 * self.height
        if foot.sum() < 3:
            return None
        f, l = frame["forward"], frame["left"]
        pts = np.stack([v @ f, v @ l], 1).astype(np.float32)
        hull = cv2.convexHull(pts[foot])
        com = pts.mean(0)
        d = cv2.pointPolygonTest(hull, (float(com[0]), float(com[1])), True)
        if d >= -0.05 * self.height:
            return None
        c = hull.reshape(-1, 2).mean(0)
        off = com - c
        side = ("forward" if off[0] > 0 else "backward") if abs(off[0]) >= abs(off[1]) \
            else ("left" if off[1] > 0 else "right")
        return (f"centre of mass is {-d / self.height * 100:.0f}% of body height outside "
                f"the support area ({side}) -- the pose would tip over unless it is "
                f"mid-motion or the character flies/swims")


# ==============================================================================
# IK
# ==============================================================================

def _aa_deg(rot6d):
    aa = matrix_to_axis_angle(rotation_6d_to_matrix(rot6d))
    return {a: round(float(x) * 180.0 / math.pi, 1) for a, x in zip("xyz", aa)}


def solve_ik(model, effector, target, chain, adjustable, iters=200, lr=0.08, tol=0.003):
    """Rotate up to `chain` adjustable ancestors of `effector` so the joint
    lands on `target` (world xyz). Starts from the model's CURRENT pose and
    stays close to it; stops once within `tol` world units.
    Returns ({joint: {x, y, z} degrees}, residual)."""
    c2p = fb07._build_child_to_parent(model.hierarchy)
    joints, j = [], c2p.get(effector)
    while j is not None and len(joints) < chain:
        if j in adjustable and j in model.joint_rotations:
            joints.append(j)
        j = c2p.get(j)
    if not joints:
        raise ValueError(f"{effector} has no adjustable ancestors")
    init = {k: model.joint_rotations[k].data.clone() for k in joints}
    tgt = torch.tensor(target, dtype=torch.float32, device=init[joints[0]].device)
    params = [model.joint_rotations[k] for k in joints]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    with torch.enable_grad():
        for _ in range(iters):
            opt.zero_grad()
            pos = model.get_joint_positions()[effector]
            err = ((pos - tgt) ** 2).sum()
            if float(err.detach()) < tol * tol:
                break
            loss = err + 1e-3 * sum(((model.joint_rotations[k] - init[k]) ** 2).sum()
                                     for k in joints)
            loss.backward()
            opt.step()
    for p in params:
        p.requires_grad_(False)
        # re-orthonormalise: Adam steps leave the 6D vectors unnormalised
        p.data.copy_(matrix_to_rotation_6d(rotation_6d_to_matrix(p.data)))
    with torch.no_grad():
        res = float((model.get_joint_positions()[effector] - tgt).norm())
    return {k: _aa_deg(model.joint_rotations[k].data) for k in joints}, res


# ==============================================================================
# MIRROR
# ==============================================================================

_SIDE_PATTERNS = [("L_", "R_"), ("Left", "Right"), ("left", "right"),
                  (".L", ".R"), ("_L", "_R"), ("_l", "_r")]


def _name_twin(j, names):
    for a, b in _SIDE_PATTERNS:
        for x, y in ((a, b), (b, a)):
            if x in j:
                t = j.replace(x, y, 1)
                if t != j and t in names:
                    return t
    return None


class Mirror:
    """Left/right pairs: by name when the rig has side names (L_/R_, Left/
    Right, .L/.R), otherwise by reflecting rest joint positions across the
    sagittal plane through the root (mutual nearest neighbours), so bone_N
    rigs work too. Joints on the plane (spine, neck, tail) pair with
    themselves. Reflecting through the joints' centroid paired panda's Hip
    with its Pelvis and left the arms unpaired."""

    def __init__(self, model, frame, adjustable, height):
        n = frame["left"] / np.linalg.norm(frame["left"])
        self.M = np.eye(3) - 2 * np.outer(n, n)
        names = sorted(adjustable)
        P = np.stack([model.joints[j].detach().cpu().numpy() for j in names])
        c = model.joints[sorted(model.root_joints)[0]].detach().cpu().numpy()
        side = (P - c) @ n
        refl = (P - c) @ self.M.T + c
        tree = cKDTree(P)
        d, idx = tree.query(refl)
        nameset = set(names)
        self.pair = {}
        for i, j in enumerate(names):
            twin = _name_twin(j, nameset)
            if twin:
                self.pair[j] = twin
            elif abs(side[i]) < 0.025 * height:
                self.pair[j] = j
            elif d[i] < 0.08 * height and tree.query(refl[idx[i]])[1] == i:
                self.pair[j] = names[idx[i]]

    def rot(self, xyz):
        # Axis-angle is a pseudovector: under reflection M it maps to -M v.
        v = -self.M @ np.array([float(xyz.get(a, 0.0)) for a in "xyz"])
        return {a: round(float(x), 1) for a, x in zip("xyz", v)}

    def apply(self, joints, mode="copy"):
        """copy: keep the input and add the mirrored side (symmetric pose).
        swap: mirror the whole pose (lean right -> lean left)."""
        out = dict(joints) if mode == "copy" else {}
        for j, ang in joints.items():
            p = self.pair.get(j)
            if p is None:
                continue
            if mode == "copy" and p == j:
                continue                     # centre joint: keep as given
            out[p] = self.rot(ang)
        return out
