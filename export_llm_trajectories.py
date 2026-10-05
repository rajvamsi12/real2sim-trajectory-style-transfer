import json
import math
import os

STABILIZE_EKF_CSV = "/home/raj/Desktop/carlascripts/stabilize_ekf_final2.csv"
INPUT_FILE = STABILIZE_EKF_CSV if os.path.exists(STABILIZE_EKF_CSV) else "/home/raj/Desktop/carlascripts/default.json"
TIDS_FILE = "/home/raj/Desktop/carlascripts/tids.txt"
JSON_OUTPUT = "/home/raj/Desktop/carlascripts/llm_trajectory_prompts.json"

FRAME_REGIONS = {
    1: (0, 177),
    2: (178, 327),
    3: (328, 475),
    4: (476, 590),
    5: (591, 779),
    6: (780, 910),
    7: (911, 1075),
    8: (1076, 1240),
    9: (1241, 1407)
}

def parse_tids_file(tids_path):
    regions_tids = {}
    current_region = None
    with open(tids_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(".") and line[:-1].isdigit():
                current_region = int(line[:-1])
                regions_tids[current_region] = set()
            elif current_region is not None and line.startswith("{") and line.endswith("}"):
                inner = line[1:-1]
                tids = set(int(x.strip()) for x in inner.split(",") if x.strip())
                regions_tids[current_region] = tids
    return regions_tids

region_tids_map = parse_tids_file(TIDS_FILE)

print("Exporting 15-second trajectory JSONs for Zero-Shot LLM Reasoning...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

items = data.get("items", [])
total_frames = len(items)

track_history = {}
frame_tracks = {}

for f_idx, item in enumerate(items):
    frame_tracks[f_idx] = {}
    for ann in item.get("annotations", []):
        tid = ann.get("attributes", {}).get("track_id")
        if tid is None:
            continue
        tid = int(tid)
        if tid < 20: # Exclude static background objects < 20
            continue
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            cx, cy = x + w / 2.0, y + h / 2.0
            frame_tracks[f_idx][tid] = (cx, cy, w, h)
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append((f_idx, cx, cy, w, h))

def calculate_movement(pts):
    if len(pts) < 2:
        return 0.0
    return math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])

moving_tracks = {}
for tid, pts in track_history.items():
    mv = calculate_movement(pts)
    if len(pts) >= 12 and mv >= 20.0:
        moving_tracks[tid] = pts

window_size = 64 # 15 seconds at 4.28 FPS
llm_prompts = []
used_egos = set()

for reg_num in range(1, 10):
    reg_sf, reg_ef = FRAME_REGIONS[reg_num]
    allowed_tids = region_tids_map.get(reg_num, set())
    
    for tid, pts in moving_tracks.items():
        if tid in allowed_tids and tid not in used_egos:
            first_f, last_f = pts[0][0], pts[-1][0]
            if reg_sf <= first_f <= reg_ef or reg_sf <= last_f <= reg_ef:
                sf = max(reg_sf, first_f)
                ef = min(reg_ef, sf + window_size - 1)
                if ef - sf + 1 < 30:
                    sf = max(reg_sf, reg_ef - window_size + 1)
                    ef = min(reg_ef, sf + window_size - 1)
                
                ego_pts = [(cx, cy) for (f, cx, cy, w, h) in pts if sf <= f <= ef]
                if len(ego_pts) < 10:
                    continue
                    
                used_egos.add(tid)
                
                # Extract neighbor trajectories in window
                neighbors_dict = {}
                for f in range(sf, ef + 1):
                    for t, (cx, cy, w, h) in frame_tracks.get(f, {}).items():
                        if t in allowed_tids and t != tid:
                            if t not in neighbors_dict:
                                neighbors_dict[t] = []
                            neighbors_dict[t].append((cx, cy))
                            
                valid_neighbors = {t: p for t, p in neighbors_dict.items() if len(p) >= 5}
                sorted_neighbors = sorted(valid_neighbors.items(), key=lambda x: len(x[1]), reverse=True)[:4]
                neighbor_ids = [n[0] for n in sorted_neighbors]
                
                # Format structured JSON for LLM
                dx = ego_pts[-1][0] - ego_pts[0][0]
                dy = -(ego_pts[-1][1] - ego_pts[0][1]) # Drone Y-inversion
                heading_deg = math.degrees(math.atan2(dy, dx)) if (abs(dx)+abs(dy)) > 1e-4 else 0.0
                
                prompt_entry = {
                    "ego_id": tid,
                    "region": reg_num,
                    "start_frame": sf,
                    "end_frame": ef,
                    "duration_frames": ef - sf + 1,
                    "duration_seconds": 15.0,
                    "ego_kinematics": {
                        "start_pos": [round(ego_pts[0][0], 1), round(ego_pts[0][1], 1)],
                        "end_pos": [round(ego_pts[-1][0], 1), round(ego_pts[-1][1], 1)],
                        "delta_x_px": round(dx, 1),
                        "delta_y_px": round(dy, 1),
                        "net_heading_deg": round(heading_deg, 1)
                    },
                    "neighbor_ids": neighbor_ids,
                    "num_neighbors": len(neighbor_ids)
                }
                llm_prompts.append(prompt_entry)

with open(JSON_OUTPUT, "w", encoding="utf-8") as out:
    json.dump(llm_prompts, out, indent=2)

print(f"Exported {len(llm_prompts)} 15-second trajectory JSON prompts to {JSON_OUTPUT}")
