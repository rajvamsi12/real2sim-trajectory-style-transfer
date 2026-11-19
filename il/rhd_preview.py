"""
il/rhd_preview.py

Preview of the corrected (right-hand traffic) mapping, il/mapping_rhd.py, without touching the current data.
  build   rebuild the 8 s scenarios with the corrected matrices (same offline checks as il/build_scenarios_8s.py:
          in-box, spawnable, car contact with <= 0.7 m shift, kerb/median, neighbours) -> Scenariosssss/IL_rhd/
  video   drone vs CARLA clips (il/render_compare_video.py) of chosen scenarios from Scenariosssss/IL_rhd/

  python il/rhd_preview.py build
  python il/rhd_preview.py video --ids 00012 00100 00300 --out Scenariosssss/IL_rhd/preview_3.mp4
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

import mapping_rhd  # noqa: E402

mapping_rhd.use_rhd()
OUT = os.path.join(os.path.dirname(SCRIPT_DIR), "Scenariosssss", "IL_rhd")

if __name__ == "__main__":
    what, sys.argv = sys.argv[1], [sys.argv[0]] + sys.argv[2:]
    if what == "build":
        import build_scenarios_8s as B
        B.OUT_DIR = OUT
        B.main()
    elif what == "video":
        import render_compare_video as R
        R.IL_DIR = OUT
        R.main()
    else:
        sys.exit("usage: rhd_preview.py build | video ...")
