"""
agent.py -- interactive Animation Agent.

Pick a rigged character, type what it should do (in English), get a video.
Repeat as often as you like; the character stays loaded between commands.

  conda activate VideoArticulation        (or call that env's python.exe)
  set KMP_DUPLICATE_LIB_OK=TRUE
  set ANTHROPIC_API_KEY=<key>             (asked for at startup if missing)
  python agent.py

At the command prompt:
  <text>    animate the current character, e.g. "wave hello, then bow"
  :c        choose another character
  :o        open the last video
  :q        quit
"""

import os
import sys
import time
import glob
import getpass
import argparse

import torch
import anthropic

import animate
import agent_fb_07 as fb07


ASSET_DIR = "asset"
RUNS_DIR = "runs"


def find_characters(asset_dir=ASSET_DIR):
    """{name: (mesh .obj, rig .txt)} for every asset folder that has both."""
    chars = {}
    for d in sorted(glob.glob(os.path.join(asset_dir, "*"))):
        meshes = glob.glob(os.path.join(d, "*_texture.obj"))
        rigs = glob.glob(os.path.join(d, "rig", "*_ori_rig.txt"))
        if meshes and rigs:
            name = os.path.basename(d).replace("_texture_obj", "")
            chars[name] = (meshes[0], rigs[0])
    return chars


def print_characters(chars):
    names = list(chars)
    width = max(len(n) for n in names) + 6
    per_row = max(1, 96 // width)
    for i in range(0, len(names), per_row):
        print("".join(f"{j + 1:>3}. {names[j]:<{width - 5}}"
                      for j in range(i, min(i + per_row, len(names)))))


def ask(prompt):
    """input() that strips a BOM (PowerShell pipes add one) and turns EOF /
    Ctrl+C into a clean quit."""
    try:
        return input(prompt).replace("\ufeff", "").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        raise SystemExit(0)


def choose_character(chars, current=None):
    print_characters(chars)
    names = list(chars)
    while True:
        hint = f" [Enter = keep {current}]" if current else ""
        raw = ask(f"\nCharacter (number or name){hint}: ")
        if not raw and current:
            return current
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        if raw in chars:
            return raw
        matches = [n for n in names if n.startswith(raw.lower())] if raw else []
        if len(matches) == 1:
            return matches[0]
        print("  not found -- type a number from the list, or the name")


def make_args(mesh, rig, base):
    """A fresh Namespace per character: build_scene fills in cam_dist, so a
    value left over from the previous character would mis-frame the next."""
    return argparse.Namespace(**dict(vars(base), mesh_path=mesh, rig_path=rig,
                                     cam_dist=None, command=""))


def get_client(claude_model):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        key = getpass.getpass("ANTHROPIC_API_KEY is not set. Paste it (hidden): ").strip()
        os.environ["ANTHROPIC_API_KEY"] = key
    client = anthropic.Anthropic(api_key=key, timeout=60.0)
    print(f"Checking {claude_model} ...", end=" ", flush=True)
    animate.check_claude(client, claude_model)
    print("ok")
    return client


def open_file(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except OSError as e:
        print(f"  could not open it: {e}")


def main():
    p = argparse.ArgumentParser(description="Interactive Animation Agent")
    p.add_argument("--claude_model", default="claude-opus-5-5")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_steps", type=int, default=14)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--cam_elev", type=float, default=10)
    base = p.parse_args()
    base.cam_azim = None

    fb07.set_seed(42)
    chars = find_characters()
    if not chars:
        raise SystemExit(f"No characters found under {ASSET_DIR}/ "
                         "(need <name>_texture.obj and rig/<name>_ori_rig.txt).")

    print("=" * 60)
    print(" Animation Agent -- type a command, get a video")
    print("=" * 60)
    client = get_client(base.claude_model)

    name = choose_character(chars)
    scene = None
    last_video = None

    while True:
        if scene is None or scene[0] != name:
            print(f"Loading {name} ...")
            args = make_args(*chars[name], base)
            scene = (name, args, *animate.build_scene(args))
        _, args, model, renderer, camera, lights = scene

        raw = ask(f"\n[{name}] command (:c character, :o open last, :q quit) > ")
        if not raw:
            continue
        if raw.lower() in (":q", ":quit", "exit", "quit"):
            break
        if raw.lower() == ":c":
            name = choose_character(chars, current=name)
            continue
        if raw.lower() == ":o":
            if last_video:
                open_file(last_video)
            else:
                print("  no video yet")
            continue
        if not raw.isascii():
            print('  Please type the command in English, e.g. "wave hello, then bow".')
            continue

        args.command = raw
        out_dir = os.path.join(RUNS_DIR, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Animating ... (usually 1-3 minutes; Ctrl+C to cancel)")
        t0 = time.time()
        try:
            last_video = animate.run_direct(args, model, renderer, camera, lights,
                                            client, out_dir,
                                            log=lambda m: print(f"  {m}"))
        except KeyboardInterrupt:
            print("\n  cancelled")
            continue
        except Exception as e:
            print(f"\n  failed: {type(e).__name__}: {e}")
            continue
        print(f"\nDone in {time.time() - t0:.0f}s:\n  {last_video}")
        if ask("Open it now? [Y/n] ").lower() in ("", "y", "yes"):
            open_file(last_video)

    print("Bye.")


if __name__ == "__main__":
    main()
