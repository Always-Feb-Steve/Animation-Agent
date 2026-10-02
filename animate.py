"""
animate.py -- one text command in, one animated video of the mesh out.

Two modes:
  --mode direct (default)  Claude choreographs keyframes itself; only an
                           Anthropic key is needed. See director.py.
  --mode video             a reference video (Kling, or --video_path) is
                           generated and the pose is fitted to it, below.

Video-mode pipeline
-------------------
1. Render the rigged mesh in its rest pose from the fitting camera.
2. Claude turns the command into an image-to-video prompt; Kling (via fal)
   animates the rest render. Because the render IS the first frame, the
   reference video shares our camera exactly -- that is what lets the
   silhouette fitting of agent_fb_07 work on it unchanged.
3. Every `--key_every` seconds a frame is pulled from the reference video and
   agent_fb_07's search fits the pose to it, warm-started from the previous
   keyframe (frame 0 is the rest pose by construction and is not fitted).
4. Joint rotations are slerped between keyframes, root translation is lerped,
   and every frame is rendered to animation.mp4 (plus comparison.mp4: the
   reference beside the result).

Usage:
  conda activate VideoArticulation
  set KMP_DUPLICATE_LIB_OK=TRUE
  set ANTHROPIC_API_KEY=<key>
  set FAL_KEY=<key>
  python animate.py --command "bow politely, then stand back up"
  python animate.py --mode video --command "..."      # needs FAL_KEY too

  # skip video generation, reuse an existing reference video
  python animate.py --mode video --command "bow" --video_path asset/ultraman_texture_obj/videos/bow_seq.mp4
  # skip fitting, re-render from a previous run's poses
  python animate.py --from_poses runs/xxx/poses.pt
"""

import os
import json
import time
import argparse

import cv2
import numpy as np
import torch
import requests
import anthropic
from pytorch3d.renderer import PointLights
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

from rigging.model import RiggingModel
from utils.io import load_mesh
from utils.rig_parser import parse_rig_file
from rendering.renderer import Renderer
from rendering.camera import get_camera
import agent_fb_07 as fb07
import director


KLING_ENDPOINT = "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"

PROMPT_SYSTEM = (
    "You write prompts for an image-to-video model. The input image is a 3D "
    "character rendered on a plain white background. The video will be used "
    "as a motion reference: a program will track the character's silhouette "
    "frame by frame and copy the pose onto the 3D model, so the video must be "
    "easy to track.\n"
    "Rules for the prompt you write:\n"
    "- Static camera: no pan, zoom, cut, or camera shake.\n"
    "- The background stays plain white for the whole video; no shadows, "
    "props, or other characters appear.\n"
    "- The character keeps exactly the same design, proportions, and colors; "
    "it does not morph, grow, or change clothes.\n"
    "- The whole body stays inside the frame at all times.\n"
    "- The motion is the one the user asked for, performed clearly and with "
    "readable, slightly exaggerated body poses, completed within the video's "
    "duration.\n"
    "- Describe body parts concretely (which arm, which way the torso bends).\n"
    "Return only the prompt, in English, under 80 words."
)


# ==============================================================================
# SETUP
# ==============================================================================

def build_scene(args):
    device = args.device
    mesh = load_mesh(args.mesh_path, load_materials=True, device=device)
    joints, skin_weights, hierarchy = parse_rig_file(args.rig_path)
    model = RiggingModel(mesh, joints, skin_weights, hierarchy).to(device)
    model.eval()
    if args.cam_dist is None:
        # Fit the bounding box: the assets are not normalised (cat ~2.0 across,
        # ultraman ~1.0 tall). 2.2 x half-diagonal reproduces ultraman's 1.25.
        with torch.no_grad():
            v = model().verts_packed()
        args.cam_dist = round(2.2 * float((v.max(0).values - v.min(0).values).norm()) / 2, 3)
    camera = get_camera(dist=args.cam_dist, elev=args.cam_elev,
                        azim=75 if args.cam_azim is None else args.cam_azim,
                        device=device)
    # Same lights as agent_fb_07: flat, near-ambient, so the silhouette is clean.
    lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]],
                         ambient_color=((0.95, 0.95, 0.95),),
                         diffuse_color=((0.05, 0.05, 0.05),),
                         specular_color=((0.0, 0.0, 0.0),))
    renderer = Renderer(image_size=args.image_size, device=device)
    return model, renderer, camera, lights


# ==============================================================================
# STEP 2: COMMAND -> REFERENCE VIDEO
# ==============================================================================

def write_video_prompt(command, rest_png, client, claude_model, duration):
    response = client.messages.create(
        model=claude_model,
        max_tokens=4096,
        system=PROMPT_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": fb07._img_to_b64(rest_png)}},
                {"type": "text", "text": f"Video length: {duration} seconds.\n"
                                         f"User's command: {command}"},
            ],
        }],
    )
    prompt = fb07._answer_text(response)
    if not prompt:
        raise RuntimeError("Claude returned no prompt text.")
    return prompt


def generate_reference_video(prompt, rest_png, out_path, duration):
    import fal_client  # only needed on this path

    image_url = fal_client.upload_file(rest_png)
    print(f"Kling: generating {duration}s video ...")
    result = fal_client.subscribe(
        KLING_ENDPOINT,
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "duration": str(duration),
            "negative_prompt": "camera movement, zoom, cut, blur, distortion, "
                               "morphing, extra limbs, background change, low quality",
            "cfg_scale": 0.5,
        },
        with_logs=False,
    )
    url = result["video"]["url"]
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"Reference video -> {out_path}")


# ==============================================================================
# STEP 3: KEYFRAME FITTING
# ==============================================================================

def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def read_frames(path, size):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (size, size)))
    cap.release()
    return frames


def fit_keyframes(model, renderer, camera, lights, ref_frames, ref_fps,
                  out_dir, args, client):
    duration = (len(ref_frames) - 1) / ref_fps
    times = list(np.arange(0.0, duration + 1e-6, args.key_every))
    if duration - times[-1] > 0.25 * args.key_every:
        times.append(duration)

    rest = fb07.save_joint_state(model)
    keys = [(0.0, rest)]
    for t in times[1:]:
        kdir = os.path.join(out_dir, f"key_{t:05.2f}s")
        os.makedirs(kdir, exist_ok=True)
        ref_png = os.path.join(kdir, "reference.png")
        cv2.imwrite(ref_png, ref_frames[min(round(t * ref_fps), len(ref_frames) - 1)])
        print(f"\n{'#'*60}\nKeyframe t={t:.2f}s (warm start from previous)\n{'#'*60}")
        fb07.run_agent(model=model, renderer=renderer, camera=camera, lights=lights,
                       reference_path=ref_png, output_dir=kdir,
                       image_size=args.image_size, claude_client=client,
                       claude_model=args.claude_model, max_iter=args.key_max_iter,
                       patience=args.patience, max_verify=args.max_verify,
                       root_in_pool=False, tolerance=args.tolerance,
                       max_regress=args.max_regress)
        keys.append((float(t), fb07.save_joint_state(model)))
    return keys


# ==============================================================================
# STEP 4: INTERPOLATE + RENDER
# ==============================================================================

def _slerp(q0, q1, w):
    if torch.dot(q0, q1) < 0:          # take the short way round
        q1 = -q1
    d = torch.clamp(torch.dot(q0, q1), -1.0, 1.0)
    if d > 0.9995:
        q = q0 + w * (q1 - q0)
        return q / q.norm()
    theta = torch.acos(d)
    return (torch.sin((1 - w) * theta) * q0 + torch.sin(w * theta) * q1) / torch.sin(theta)


def _slerp_6d(a, b, w):
    qa = matrix_to_quaternion(rotation_6d_to_matrix(a))
    qb = matrix_to_quaternion(rotation_6d_to_matrix(b))
    return matrix_to_rotation_6d(quaternion_to_matrix(_slerp(qa, qb, w)))


def interpolate_state(keys, t):
    """Pose at time t: slerp joint rotations, lerp root translation, slerp root
    rotation, between the two keyframes bracketing t."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, s0), (t1, s1) in zip(keys, keys[1:]):
        if t <= t1:
            w = (t - t0) / max(t1 - t0, 1e-9)
            # smoothstep: ease in/out at every key, so motion does not jerk there
            w = w * w * (3 - 2 * w)
            rots = {k: _slerp_6d(s0[0][k], s1[0][k], w) for k in s0[0]}
            r0, r1 = s0[1], s1[1]
            root = torch.cat([r0[:3] + w * (r1[:3] - r0[:3]),
                              _slerp_6d(r0[3:], r1[3:], w)])
            return rots, root
    return keys[-1][1]


def render_sequence(model, renderer, camera, lights, keys, out_dir, fps,
                    ref_frames=None, ref_fps=None, cameras=None):
    duration = keys[-1][0]
    n = int(round(duration * fps)) + 1
    size = None
    anim = comp = None
    tmp = os.path.join(out_dir, "_frame.png")
    for i in range(n):
        t = i / fps
        fb07.restore_joint_state(model, interpolate_state(keys, t))
        fb07.render_rgb_to_png(model, renderer, cameras[i] if cameras else camera,
                               lights, tmp)
        frame = cv2.imread(tmp)
        if anim is None:
            size = frame.shape[1], frame.shape[0]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            anim = cv2.VideoWriter(os.path.join(out_dir, "animation.mp4"), fourcc, fps, size)
            if ref_frames:
                comp = cv2.VideoWriter(os.path.join(out_dir, "comparison.mp4"), fourcc,
                                       fps, (size[0] * 2, size[1]))
        anim.write(frame)
        if comp is not None:
            ref = ref_frames[min(round(t * ref_fps), len(ref_frames) - 1)]
            comp.write(np.hstack([cv2.resize(ref, size), frame]))
        print(f"\rrendering {i + 1}/{n}", end="")
    print()
    os.remove(tmp)
    anim.release()
    if comp is not None:
        comp.release()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def check_claude(client, claude_model):
    try:
        client.messages.create(model=claude_model, max_tokens=4,
                               messages=[{"role": "user", "content": "Reply ok."}])
    except anthropic.AuthenticationError as e:
        raise SystemExit(f"API key rejected: {e}")
    except Exception as e:
        raise SystemExit(f"Claude model {claude_model!r} unusable: "
                         f"{type(e).__name__}: {e}")


def run_direct(args, model, renderer, camera, lights, client, out_dir, log=print):
    """Direct mode, command -> animation.mp4. Leaves the model at rest so a
    caller (agent.py) can reuse it for the next command."""
    with open(os.path.join(out_dir, "command.txt"), "w", encoding="utf-8") as f:
        f.write(args.command + "\n")
    rest = fb07.save_joint_state(model)
    try:
        keys, film_azim = director.direct(model, lights, client, args.claude_model,
                                          args.command, out_dir, args.cam_dist,
                                          args.device, max_steps=args.max_steps, log=log)
        torch.save({"keys": [(t, s[0], s[1]) for t, s in keys], "video_path": ""},
                   os.path.join(out_dir, "poses.pt"))
        cameras = None
        if args.cam_azim is None:
            cameras = director.tracking_cameras(model, keys, args.fps, interpolate_state,
                                                film_azim, args.cam_elev, args.device, rest)
        render_sequence(model, renderer, camera, lights, keys, out_dir, args.fps,
                        cameras=cameras)
    finally:
        fb07.restore_joint_state(model, rest)
    return os.path.abspath(os.path.join(out_dir, "animation.mp4"))


def main():
    fb07.set_seed(42)
    p = argparse.ArgumentParser()
    p.add_argument("--command", default="",
                   help="what the character should do, in English, e.g. \"wave hello, then bow\"")
    p.add_argument("--mode", default="direct", choices=["direct", "video"])
    p.add_argument("--max_steps", type=int, default=14,
                   help="direct mode: tool turns Claude gets to preview and submit")
    p.add_argument("--mesh_path", default="asset/ultraman_texture_obj/ultraman_texture.obj")
    p.add_argument("--rig_path", default="asset/ultraman_texture_obj/rig/ultraman_ori_rig.txt")
    p.add_argument("--video_path", default="",
                   help="existing reference video; skips Claude prompt + Kling")
    p.add_argument("--from_poses", default="",
                   help="poses.pt of a previous run; skips generation and fitting")
    p.add_argument("--output_dir", default="")
    p.add_argument("--duration", type=int, default=5, choices=[5, 10],
                   help="Kling clip length in seconds")
    p.add_argument("--key_every", type=float, default=1.0,
                   help="seconds between fitted keyframes")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cam_dist", type=float, default=None,
                   help="default: fitted to the mesh's bounding box")
    p.add_argument("--cam_elev", type=float, default=10)
    p.add_argument("--cam_azim", type=float, default=None,
                   help="default: direct mode films 30 deg off the detected front; "
                        "video mode uses 75 (agent_fb_07's camera)")
    p.add_argument("--claude_model", default="claude-opus-5-5")
    p.add_argument("--key_max_iter", type=int, default=15,
                   help="agent iterations per keyframe (warm start needs fewer)")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max_verify", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=0.06)
    p.add_argument("--max_regress", type=int, default=2)
    args = p.parse_args()
    if args.command and not args.command.isascii():
        raise SystemExit('--command must be in English, e.g. "wave hello, then bow".')

    out_dir = args.output_dir or os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    model, renderer, camera, lights = build_scene(args)
    ref_frames = ref_fps = None

    if args.from_poses:
        saved = torch.load(args.from_poses)
        keys = [(t, (rots, root)) for t, rots, root in saved["keys"]]
        if saved.get("video_path") and os.path.exists(saved["video_path"]):
            ref_fps, _ = video_info(saved["video_path"])
            ref_frames = read_frames(saved["video_path"], args.image_size)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY.")
        client = anthropic.Anthropic(api_key=api_key, timeout=60.0)
        check_claude(client, args.claude_model)

        rest_png = os.path.join(out_dir, "rest.png")
        fb07.render_rgb_to_png(model, renderer, camera, lights, rest_png)

        if args.mode == "direct":
            if not args.command:
                raise SystemExit("Pass --command.")
            video = run_direct(args, model, renderer, camera, lights, client, out_dir)
            print(f"\nDone: {video}")
            return

        video_path = args.video_path
        if not video_path:
            if not args.command:
                raise SystemExit("Pass --command (or --video_path).")
            if not os.environ.get("FAL_KEY"):
                raise SystemExit("Set FAL_KEY for Kling video generation.")
            prompt = write_video_prompt(args.command, rest_png, client,
                                        args.claude_model, args.duration)
            print(f"Video prompt: {prompt}")
            with open(os.path.join(out_dir, "video_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(f"command: {args.command}\n\nprompt: {prompt}\n")
            video_path = os.path.join(out_dir, "reference.mp4")
            generate_reference_video(prompt, rest_png, video_path, args.duration)

        ref_fps, _ = video_info(video_path)
        ref_frames = read_frames(video_path, args.image_size)
        keys = fit_keyframes(model, renderer, camera, lights, ref_frames, ref_fps,
                             out_dir, args, client)
        torch.save({"keys": [(t, s[0], s[1]) for t, s in keys],
                    "video_path": os.path.abspath(video_path)},
                   os.path.join(out_dir, "poses.pt"))

    render_sequence(model, renderer, camera, lights, keys, out_dir, args.fps,
                    ref_frames, ref_fps)
    print(f"\nDone: {os.path.join(out_dir, 'animation.mp4')}")


if __name__ == "__main__":
    main()
