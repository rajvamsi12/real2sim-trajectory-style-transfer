import glob
import os
import subprocess
import time

SCENARIOS_DIR = "/home/raj/Desktop/carlascripts/Scenariosssss"
PYTHON_BIN = "/home/raj/miniconda3/envs/carla_env/bin/python"
ANNOTATIONS_FILE = "/home/raj/Desktop/carlascripts/default.json"
VIDEO_FILE = "/home/raj/Desktop/carlascripts/DJI_0003-002.MP4"
SCRIPT_PATH = "/home/raj/Desktop/carlascripts/overlay_scenario_bboxes.py"

print("Removing old video files from Scenariosssss...")
old_vids = glob.glob(os.path.join(SCENARIOS_DIR, "**/*_zoom_boxed.mp4"), recursive=True)
for v in old_vids:
    try:
        os.remove(v)
    except Exception:
        pass

summaries = glob.glob(os.path.join(SCENARIOS_DIR, "**/*_summary.txt"), recursive=True)
summaries.sort()

total_summaries = len(summaries)
print(f"=== Starting Batch Render of {total_summaries} Sample Drone Videos (2 per category) ===", flush=True)

rendered_count = 0
failed_count = 0
start_time = time.time()

for idx, summary_path in enumerate(summaries, 1):
    scen_folder = os.path.dirname(summary_path)
    scen_name = os.path.basename(scen_folder)
    cat_name = os.path.basename(os.path.dirname(scen_folder))
    
    video_name = f"{scen_name}_zoom_boxed.mp4"
    video_path = os.path.join(scen_folder, video_name)
    
    print(f"[{idx}/{total_summaries}] Rendering ({cat_name} -> {scen_name})...", flush=True)
    
    cmd = [
        PYTHON_BIN,
        SCRIPT_PATH,
        "--summary", summary_path,
        "--annotations", ANNOTATIONS_FILE,
        "--video", VIDEO_FILE,
        "--zoom",
        "--zoom-padding", "250",
        "--zoom-min-size", "650"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        rendered_count += 1
        size_mb = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0.0
        print(f"  --> SUCCESS: {video_name} ({size_mb:.2f} MB)", flush=True)
    else:
        failed_count += 1
        print(f"  --> FAILED: {res.stderr.strip()[:200]}", flush=True)

elapsed = time.time() - start_time
print(f"\n==========================================")
print(f"Sample Test Batch Render Finished in {elapsed/60.0:.2f} minutes!")
print(f"  Rendered: {rendered_count} / {total_summaries}")
print(f"==========================================")
