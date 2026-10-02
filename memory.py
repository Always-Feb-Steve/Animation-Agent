"""
memory.py -- what the agent keeps between runs.

memory/<character>.json   per character
    front_azim, bone_names   perception results, so a known character skips
                             the facing and bone-naming calls
    poses                    a pose library: poses Claude verified and saved
                             with save_pose (e.g. panda "arms_overhead_v",
                             which took five previews to find)
memory/lessons.json        transferable lessons, written by Claude after each
                             run ("big-headed characters: raise arms out to
                             the side, not forward") and shown to every
                             later run on every character

Character entries are keyed by a hash of the mesh and rig files, so editing
either one invalidates the cache instead of silently reusing stale results.
"""

import os
import json
import time
import hashlib


MEMORY_DIR = "memory"
MAX_LESSONS = 40


def _file_hash(*paths):
    h = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


class CharacterMemory:
    def __init__(self, name, mesh_path, rig_path, root=MEMORY_DIR):
        self.name = name
        self.path = os.path.join(root, f"{name}.json")
        self.key = _file_hash(mesh_path, rig_path)
        data = _load(self.path, {})
        if data.get("key") != self.key:
            data = {"key": self.key, "character": name}
        self.data = data

    def get(self, field, default=None):
        return self.data.get(field, default)

    def set(self, field, value):
        self.data[field] = value
        self.save()

    @property
    def poses(self):
        return self.data.setdefault("poses", {})

    def save_pose(self, pose_name, joints, root=None, note=""):
        self.poses[pose_name] = {"joints": joints, "root": root or {}, "note": note,
                                 "saved": time.strftime("%Y-%m-%d")}
        self.save()

    def poses_text(self):
        if not self.poses:
            return ""
        lines = [f'- "{k}": {v.get("note", "")}\n  {json.dumps({"joints": v["joints"], "root": v.get("root") or {}})}'
                 for k, v in self.poses.items()]
        return ("POSE LIBRARY (poses you verified on this character earlier; reuse or "
                "adapt them instead of rediscovering):\n" + "\n".join(lines))

    def save(self):
        _save(self.path, self.data)


class Lessons:
    def __init__(self, root=MEMORY_DIR):
        self.path = os.path.join(root, "lessons.json")
        self.items = _load(self.path, [])

    def add(self, texts, character):
        known = {i["text"].strip().lower() for i in self.items}
        for t in texts:
            t = str(t).strip()
            if t and t.lower() not in known:
                self.items.append({"text": t, "character": character,
                                   "date": time.strftime("%Y-%m-%d")})
                known.add(t.lower())
        self.items = self.items[-MAX_LESSONS:]
        _save(self.path, self.items)

    def text(self, limit=20):
        if not self.items:
            return ""
        return ("LESSONS from earlier work (on this and other characters):\n"
                + "\n".join(f"- {i['text']} [{i['character']}]"
                            for i in self.items[-limit:]))
