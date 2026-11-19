"""Folders <behaviour>/<id>_<split>/ with drone, CARLA-from-drone and model+TTC videos and the two trajectory images (H.264)."""
import json, os, shutil, subprocess
IL = "Scenariosssss/IL"
SRC, DR, OUT = f"{IL}/showcase_ttc_raw", f"{IL}/showcase_ttc_drone", f"{IL}/showcase_ttc"
M = "rhd_scratch@cl+ttc_0.7_0.2"
pick = json.load(open(f"{IL}/showcase_ttc_pick.json"))
ff = shutil.which("ffmpeg")
def put(src, dst):
    if not os.path.exists(src):
        print("MISSING", src); return
    if src.endswith(".mp4") and ff:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", dst], check=True)
    else:
        shutil.copy(src, dst)
for b, items in pick.items():
    for sid, split in items:
        d = os.path.join(OUT, b.replace(" ", "_"), f"{sid}_{split}")
        os.makedirs(d, exist_ok=True)
        put(f"{DR}/{sid}_drone.mp4", f"{d}/1_drone_real.mp4")
        put(f"{SRC}/{sid}_expert.mp4", f"{d}/2_carla_from_drone.mp4")
        put(f"{SRC}/{sid}_expert_trajectory.png", f"{d}/2_carla_from_drone_trajectory.png")
        put(f"{SRC}/{sid}_{M}.mp4", f"{d}/3_model_ttc_0.7s_0.2m.mp4")
        put(f"{SRC}/{sid}_{M}_trajectory.png", f"{d}/3_model_ttc_0.7s_0.2m_trajectory.png")
print("ffmpeg:", ff)
print("saved", OUT)
