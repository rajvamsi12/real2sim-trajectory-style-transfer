#!/usr/bin/env python3
"""
il/fix_uturn_commands.py

One-off patch (2026-10-25): applies build_scenarios_8s.apply_uturn_command to the already built
scenarios_8s.json and scenarios_aug.json, so U-turn egos get "u-turn" until the annotated U-turn ends.
Only route_cmd changes; files are replaced atomically (the running replay has its copy in memory).
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

from build_scenarios_8s import apply_uturn_command  # noqa: E402

IL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Scenariosssss", "IL")


def save(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)


def first_cmd(s):
    return next(v for v in s["vehicles"] if v["role"] == "ego")["route_cmd"][0]


base = json.load(open(os.path.join(IL_DIR, "scenarios_8s.json")))
before = sum(first_cmd(s) == "u-turn" for s in base if s["behaviour"] == "U-turn")
for s in base:
    apply_uturn_command(s)
after = sum(first_cmd(s) == "u-turn" for s in base if s["behaviour"] == "U-turn")
save(base, os.path.join(IL_DIR, "scenarios_8s.json"))
print(f"base: U-turn egos starting with 'u-turn': {before} -> {after} of {sum(s['behaviour'] == 'U-turn' for s in base)}")

aug = json.load(open(os.path.join(IL_DIR, "scenarios_aug.json")))
by_id = {s["id"]: s for s in base}
n = 0
for c in aug:
    if c["behaviour"] != "U-turn":
        continue
    b = by_id[c["base_id"]]
    a = c.get("aug") or {}
    if a.get("type") == "scene_noise":
        fb = lambda f, a=a, c=c, b=b: b["start"] + a["trim_frames"] + a["speed_scale"] * (f - c["start"])
    else:
        fb = None
    apply_uturn_command(c, fb)
    n += first_cmd(c) == "u-turn"
save(aug, os.path.join(IL_DIR, "scenarios_aug.json"))
print(f"augmented: {n} of {sum(c['behaviour'] == 'U-turn' for c in aug)} U-turn copies start with 'u-turn'")
