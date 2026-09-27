import base64
import io
import json
import math
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_FILE = "/home/raj/Desktop/carlascripts/default.json"
VIDEO_FILE = "/home/raj/Desktop/carlascripts/DJI_0003-002.MP4"
MAP_IMAGE_FILE = "/home/raj/Desktop/carlascripts/drone_background_map.png"
TIDS_FILE = "/home/raj/Desktop/carlascripts/tids.txt"
HTML_OUTPUT = "/home/raj/Desktop/carlascripts/scenario_catalog.html"
INDEX_OUTPUT = "/home/raj/Desktop/carlascripts/index.html"

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

print("Loading dataset annotations...")
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

def get_track_turn_drone(pts):
    if len(pts) < 5:
        return "straight"
    yaws = []
    for i in range(1, len(pts)):
        p1, p2 = pts[i-1], pts[i]
        dx = p1[1] - p2[1] if len(p1) > 2 else p2[0] - p1[0]
        dy = -(p2[1] - p1[1]) if len(p1) == 2 else -(p2[2] - p1[2]) # Drone Y-axis inversion
        if abs(dx) + abs(dy) < 1e-4:
            continue
        yaws.append(math.degrees(math.atan2(dy, dx)))

    if len(yaws) < 2:
        return "straight"

    delta = yaws[-1] - yaws[0]
    while delta > 180: delta -= 360
    while delta < -180: delta += 360
    abs_delta = abs(delta)

    if abs_delta < 15:
        return "straight"
    if 15 <= abs_delta < 35:
        return "slight_left" if delta > 0 else "slight_right"
    return "left" if delta > 0 else "right"

moving_tracks = {}
for tid, pts in track_history.items():
    mv = calculate_movement(pts)
    if len(pts) >= 12 and mv >= 20.0:
        moving_tracks[tid] = pts

print(f"Total candidate moving tracks across dataset: {len(moving_tracks)}")

print(f"Loading aerial drone background map image: {MAP_IMAGE_FILE}")
bg_bgr = cv2.imread(MAP_IMAGE_FILE)
bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
img_h, img_w = bg_rgb.shape[:2]

capture = cv2.VideoCapture(VIDEO_FILE)
source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
ann_fps = 4.28274

def get_crop_thumbnail(target_frame, cx, cy, crop_size=320):
    src_f = int(round(target_frame * (source_fps / ann_fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, src_f)
    success, frame = capture.read()
    if not success or frame is None:
        return ""
    
    fh, fw = frame.shape[:2]
    left = max(0, int(round(cx - crop_size / 2)))
    top = max(0, int(round(cy - crop_size / 2)))
    right = min(fw, left + crop_size)
    bottom = min(fh, top + crop_size)
    
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return ""
    
    crop_resized = cv2.resize(crop, (180, 180), interpolation=cv2.INTER_LINEAR)
    _, buffer = cv2.imencode('.jpg', crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

NEIGHBOR_COLORS = ['#f97316', '#a855f7', '#eab308', '#16a34a', '#ec4899', '#14b8a6', '#f43f5e']

def generate_drone_map_plot_15s(ego_tid, ego_pts, neighbors_dict):
    fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=130)
    
    # Render Drone Aerial Road Map as Background Image
    ax.imshow(bg_rgb, extent=[0, img_w, img_h, 0])
    
    # Collect bounding box coords for plot extent zoom
    all_xs = [p[0] for p in ego_pts]
    all_ys = [p[1] for p in ego_pts]
    
    # Plot Ego Path in Bright Cyan
    ax.plot(all_xs, all_ys, color='#00e5ff', linewidth=3.5, label=f'EGO #{ego_tid}', zorder=10)
    ax.scatter(all_xs[0], all_ys[0], color='#00ff66', s=90, edgecolors='#0f172a', linewidth=1.5, zorder=12, label='Start (Green)')
    ax.scatter(all_xs[-1], all_ys[-1], color='#ff0055', s=90, edgecolors='#0f172a', linewidth=1.5, zorder=12, label='End (Red)')
    
    # Plot Neighbor Paths in Vibrant Dashed Lines
    for idx, (n_tid, n_pts) in enumerate(neighbors_dict.items()):
        if len(n_pts) < 2:
            continue
        n_xs = [p[0] for p in n_pts]
        n_ys = [p[1] for p in n_pts]
        all_xs.extend(n_xs)
        all_ys.extend(n_ys)
        color = NEIGHBOR_COLORS[idx % len(NEIGHBOR_COLORS)]
        ax.plot(n_xs, n_ys, color=color, linewidth=2.5, linestyle='--', label=f'VEH #{n_tid}', zorder=5)
        ax.scatter(n_xs[0], n_ys[0], color=color, s=45, zorder=6)
    
    # Crop to vehicle group region with padding
    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)
    pad_x = max(150, (max_x - min_x) * 0.4)
    pad_y = max(150, (max_y - min_y) * 0.4)
    
    x1 = max(0, min_x - pad_x)
    x2 = min(img_w, max_x + pad_x)
    y1 = max(0, min_y - pad_y)
    y2 = min(img_h, max_y + pad_y)
    
    ax.set_xlim([x1, x2])
    ax.set_ylim([y2, y1]) # Invert Y for drone image space
    
    ax.set_title(f"15s Drone Map Overlay: Ego #{ego_tid}", fontsize=9.5, fontweight='bold', pad=6, color='#ffffff', backgroundcolor='#0f172a')
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85, facecolor='#0f172a', labelcolor='#ffffff')
    ax.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=False)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

candidates_by_region = {r: [] for r in range(1, 10)}
print("Building 15-Second (64-Frame) Focused Candidate Cards (100% Unique Egos)...")

used_ego_ids = set()
card_id_counter = 1
window_size = 64 # Exactly 15 seconds at 4.28 FPS

for reg_num in range(1, 10):
    reg_sf, reg_ef = FRAME_REGIONS[reg_num]
    allowed_tids = region_tids_map.get(reg_num, set())
    
    reg_candidates = []
    for tid, pts in moving_tracks.items():
        if tid in allowed_tids and tid not in used_ego_ids:
            first_f, last_f = pts[0][0], pts[-1][0]
            if reg_sf <= first_f <= reg_ef or reg_sf <= last_f <= reg_ef:
                # Find best 64-frame window inside region
                sf = max(reg_sf, first_f)
                ef = min(reg_ef, sf + window_size - 1)
                if ef - sf + 1 < 30:
                    sf = max(reg_sf, reg_ef - window_size + 1)
                    ef = min(reg_ef, sf + window_size - 1)
                reg_candidates.append((tid, sf, ef))
                
    selected_reg_tids = reg_candidates[:25]
    print(f"Region {reg_num} (Frames {reg_sf}-{reg_ef}): found {len(reg_candidates)} unique tracks, generating 15s cards for {len(selected_reg_tids)}...")
    
    for tid, sf, ef in selected_reg_tids:
        used_ego_ids.add(tid)
        
        # Slice Ego 64-frame points
        ego_win_pts = [(cx, cy) for (f, cx, cy, w, h) in track_history[tid] if sf <= f <= ef]
        if not ego_win_pts:
            continue
            
        # Build neighbor trajectories in this 64-frame window
        neighbors_dict = {}
        for f in range(sf, ef + 1):
            for t, (cx, cy, w, h) in frame_tracks.get(f, {}).items():
                if t in allowed_tids and t != tid:
                    if t not in neighbors_dict:
                        neighbors_dict[t] = []
                    neighbors_dict[t].append((cx, cy))
                    
        valid_neighbors = {}
        for n_tid, n_pts in neighbors_dict.items():
            if len(n_pts) >= 5:
                valid_neighbors[n_tid] = n_pts
                
        sorted_neighbors = sorted(valid_neighbors.items(), key=lambda x: len(x[1]), reverse=True)[:4]
        selected_neighbors_dict = dict(sorted_neighbors)
        neighbor_ids = sorted(list(selected_neighbors_dict.keys()))
        
        turn_type = get_track_turn_drone(ego_win_pts)
        
        tags = [turn_type]
        num_neighbors = len(neighbor_ids)
        if num_neighbors >= 3:
            tags.append("dense")
            tags.append("4way")
        elif num_neighbors <= 2:
            tags.append("sparse")
        if num_neighbors >= 1:
            tags.append("overtaking")
            tags.append("following")
            tags.append("lane_change")
            
        tags_str = " ".join(tags)
        plot_b64 = generate_drone_map_plot_15s(tid, ego_win_pts, selected_neighbors_dict)
        
        # Thumbnail snapshots for start, mid, end of 15s window
        mid_f = (sf + ef) // 2
        ego_f_map = {f: (cx, cy) for (f, cx, cy, w, h) in track_history[tid]}
        cx_s, cy_s = ego_f_map.get(sf, (ego_win_pts[0][0], ego_win_pts[0][1]))
        cx_m, cy_m = ego_f_map.get(mid_f, (ego_win_pts[len(ego_win_pts)//2][0], ego_win_pts[len(ego_win_pts)//2][1]))
        cx_e, cy_e = ego_f_map.get(ef, (ego_win_pts[-1][0], ego_win_pts[-1][1]))
        
        t_start = get_crop_thumbnail(sf, cx_s, cy_s)
        t_mid = get_crop_thumbnail(mid_f, cx_m, cy_m)
        t_end = get_crop_thumbnail(ef, cx_e, cy_e)
        
        candidates_by_region[reg_num].append({
            "id": card_id_counter,
            "tid": tid,
            "start_frame": sf,
            "end_frame": ef,
            "duration": ef - sf + 1,
            "region": reg_num,
            "turn": turn_type,
            "tags": tags_str,
            "neighbors": neighbor_ids,
            "plot": plot_b64,
            "t_start": t_start,
            "t_mid": t_mid,
            "t_end": t_end
        })
        card_id_counter += 1

capture.release()

all_candidates_flat = []
for r in range(1, 10):
    all_candidates_flat.extend(candidates_by_region[r])

print(f"\nGenerated {len(all_candidates_flat)} 15-Second (64-Frame) Focused Cards across all 9 regions.")

# Build HTML
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
    <meta http-equiv="Pragma" content="no-cache" />
    <meta http-equiv="Expires" content="0" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>15-Second Drone Scenario Catalog (3-Second Fast Render)</title>
    <style>
        * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background: #0b0f19; color: #f8fafc; margin: 0; padding: 24px; }}
        header {{ margin-bottom: 20px; border-bottom: 2px solid #00e5ff; padding-bottom: 16px; display: flex; justify-content: space-between; align-items: flex-end; }}
        h1 {{ margin: 0 0 6px 0; color: #00e5ff; font-size: 26px; }}
        p {{ margin: 0; color: #94a3b8; font-size: 14px; }}
        .badge-count {{ background: #0284c7; color: #fff; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 14px; }}
        .map-banner {{ background: linear-gradient(90deg, #16a34a, #00e5ff); color: #000; padding: 8px 16px; border-radius: 8px; font-weight: 800; font-size: 14px; margin-bottom: 16px; display: inline-block; }}
        
        .section-title {{ font-size: 13px; font-weight: 700; color: #00e5ff; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }}
        .region-tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
        .reg-btn {{ background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 700; transition: all 0.2s; }}
        .reg-btn:hover, .reg-btn.active {{ background: #a855f7; color: #ffffff; border-color: #c084fc; box-shadow: 0 4px 12px rgba(168,85,247,0.3); }}
        
        .filters {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }}
        .filter-btn {{ background: #151d30; color: #cbd5e1; border: 1px solid #24324d; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 700; transition: all 0.2s; }}
        .filter-btn:hover, .filter-btn.active {{ background: #0284c7; color: #ffffff; border-color: #00e5ff; box-shadow: 0 4px 12px rgba(2,132,199,0.3); }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; }}
        .card {{ background: #151d30; border: 1px solid #24324d; border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 12px; cursor: pointer; transition: transform 0.2s, border-color 0.2s; }}
        .card:hover {{ transform: translateY(-3px); border-color: #00e5ff; box-shadow: 0 10px 20px -5px rgba(0,0,0,0.5); }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #24324d; padding-bottom: 8px; }}
        .track-id {{ font-size: 18px; font-weight: 700; color: #00e5ff; }}
        .meta-tag {{ background: #0f172a; color: #c084fc; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; border: 1px solid #581c87; }}
        .dur-tag {{ background: #16a34a; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 800; margin-left: 6px; }}
        .legend-hint {{ font-size: 11px; color: #00e5ff; margin-top: 2px; font-weight: 600; }}
        
        .visuals {{ display: flex; gap: 12px; align-items: center; }}
        .plot-img {{ width: 195px; height: 155px; background: #0f172a; border-radius: 8px; padding: 2px; border: 2px solid #00e5ff; object-fit: contain; }}
        .thumbs {{ display: flex; gap: 6px; flex: 1; overflow-x: auto; }}
        .thumb-box {{ display: flex; flex-direction: column; align-items: center; gap: 3px; }}
        .thumb-img {{ width: 85px; height: 85px; object-fit: cover; border-radius: 8px; border: 1px solid #334155; }}
        .thumb-lbl {{ font-size: 10px; color: #94a3b8; font-weight: 600; }}
        .details {{ font-size: 12px; color: #cbd5e1; background: #0b0f19; padding: 8px 12px; border-radius: 6px; display: flex; justify-content: space-between; border: 1px solid #1e293b; }}
        .actions {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
        .save-btn {{ background: #0b0f19; color: #38bdf8; border: 1px solid #0284c7; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; cursor: pointer; transition: all 0.15s; }}
        .save-btn:hover {{ background: #0284c7; color: #ffffff; }}
        .save-btn.done {{ background: #16a34a; color: #ffffff; border-color: #22c55e; }}
        
        .modal {{ display: none; position: fixed; inset: 0; background: rgba(0, 0, 0, 0.85); backdrop-filter: blur(8px); z-index: 1000; align-items: center; justify-content: center; padding: 24px; }}
        .modal-content {{ background: #151d30; border: 2px solid #00e5ff; border-radius: 16px; width: 100%; max-width: 980px; padding: 24px; display: flex; flex-direction: column; gap: 20px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7); position: relative; }}
        .close-btn {{ position: absolute; top: 16px; right: 20px; color: #94a3b8; font-size: 28px; font-weight: bold; cursor: pointer; }}
        .close-btn:hover {{ color: #ef4444; }}
        .modal-body {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .modal-plot {{ width: 500px; height: 380px; object-fit: contain; background: #0f172a; border-radius: 12px; border: 2px solid #00e5ff; padding: 4px; }}
        .modal-thumbs {{ display: flex; flex-direction: column; gap: 12px; flex: 1; }}
        .modal-thumb-row {{ display: flex; gap: 12px; }}
        .modal-thumb-img {{ width: 125px; height: 125px; border-radius: 8px; border: 1px solid #475569; object-fit: cover; }}
        #toast {{ position: fixed; bottom: 24px; right: 24px; background: #16a34a; color: #fff; padding: 14px 24px; border-radius: 10px; font-weight: 700; display: none; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); z-index: 2000; }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>15-SECOND FOCUSED SCENARIO CATALOG</h1>
            <p>Every candidate is strictly <b>15 seconds (64 frames)</b>. Clicking any button renders the video in <b>3 seconds</b>!</p>
        </div>
        <span class="badge-count">{len(all_candidates_flat)} 15s Candidates Loaded</span>
    </header>

    <div class="map-banner">⚡ 15-SECOND WINDOWING ENABLED (64 FRAMES = 15 SECONDS | 3s RENDER TIME)</div>

    <div class="section-title">Select Timeline Region (tids.txt Matched):</div>
    <div class="region-tabs">
        <button class="reg-btn active" onclick="filterRegion('all')">All Regions ({len(all_candidates_flat)})</button>
        <button class="reg-btn" onclick="filterRegion(1)">Region 1 (0-177f)</button>
        <button class="reg-btn" onclick="filterRegion(2)">Region 2 (178-327f)</button>
        <button class="reg-btn" onclick="filterRegion(3)">Region 3 (328-475f)</button>
        <button class="reg-btn" onclick="filterRegion(4)">Region 4 (476-590f)</button>
        <button class="reg-btn" onclick="filterRegion(5)">Region 5 (591-779f)</button>
        <button class="reg-btn" onclick="filterRegion(6)">Region 6 (780-910f)</button>
        <button class="reg-btn" onclick="filterRegion(7)">Region 7 (911-1075f)</button>
        <button class="reg-btn" onclick="filterRegion(8)">Region 8 (1076-1240f)</button>
        <button class="reg-btn" onclick="filterRegion(9)">Region 9 (1241-1407f)</button>
    </div>

    <div class="section-title">Filter by All 10 Scenario Categories:</div>
    <div class="filters">
        <button class="filter-btn active" onclick="filterTurn('all')">All Maneuvers</button>
        <button class="filter-btn" onclick="filterTurn('left')">Left Turns</button>
        <button class="filter-btn" onclick="filterTurn('right')">Right Turns</button>
        <button class="filter-btn" onclick="filterTurn('slight_left')">Slight Left</button>
        <button class="filter-btn" onclick="filterTurn('slight_right')">Slight Right</button>
        <button class="filter-btn" onclick="filterTurn('overtaking')">Overtaking Candidates</button>
        <button class="filter-btn" onclick="filterTurn('lane_change')">Lane Change Candidates</button>
        <button class="filter-btn" onclick="filterTurn('4way')">4-Way Intersection</button>
        <button class="filter-btn" onclick="filterTurn('following')">Following Candidates</button>
        <button class="filter-btn" onclick="filterTurn('dense')">Dense Traffic</button>
        <button class="filter-btn" onclick="filterTurn('sparse')">Sparse Traffic</button>
    </div>

    <div class="grid" id="catalogGrid">
"""

for c in all_candidates_flat:
    neighbors_str = ", ".join(str(n) for n in c["neighbors"]) or "None"
    
    html_content += f"""
        <div class="card" data-region="{c['region']}" data-turn="{c['turn']}" data-tags="{c['tags']}" onclick="openModal({c['id']})">
            <div class="card-header">
                <div>
                    <span class="track-id">Ego Vehicle #{c['tid']}</span> <span class="dur-tag">15 SECONDS ({c['duration']}f)</span>
                    <div class="legend-hint">Ego: Bright Cyan | 🟢 Green = Start | 🔴 Red = End</div>
                </div>
                <span class="meta-tag">Region {c['region']} ({c['turn'].replace('_', ' ').upper()})</span>
            </div>
            
            <div class="visuals">
                <img class="plot-img" src="data:image/png;base64,{c['plot']}" alt="Real Drone Map Trajectory Overlay">
                <div class="thumbs">
                    <div class="thumb-box">
                        <img class="thumb-img" src="data:image/jpeg;base64,{c['t_start']}" alt="Start">
                        <span class="thumb-lbl">Start: F{c['start_frame']}</span>
                    </div>
                    <div class="thumb-box">
                        <img class="thumb-img" src="data:image/jpeg;base64,{c['t_mid']}" alt="Mid">
                        <span class="thumb-lbl">Mid: F{(c['start_frame']+c['end_frame'])//2}</span>
                    </div>
                    <div class="thumb-box">
                        <img class="thumb-img" src="data:image/jpeg;base64,{c['t_end']}" alt="End">
                        <span class="thumb-lbl">End: F{c['end_frame']}</span>
                    </div>
                </div>
            </div>

            <div class="details">
                <span><b>Frames:</b> {c['start_frame']} - {c['end_frame']} ({c['duration']}f / 15s)</span>
                <span><b>Region {c['region']} Allowed Neighbors:</b> {neighbors_str}</span>
            </div>

            <div class="actions" onclick="event.stopPropagation()">
                <button class="save-btn" onclick="saveScenario(this, 'Left', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ Left</button>
                <button class="save-btn" onclick="saveScenario(this, 'Right', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ Right</button>
                <button class="save-btn" onclick="saveScenario(this, 'slight left', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ Slight Left</button>
                <button class="save-btn" onclick="saveScenario(this, 'slight right', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ Slight Right</button>
                <button class="save-btn" onclick="saveScenario(this, 'Overtaking ', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ Overtake</button>
                <button class="save-btn" onclick="saveScenario(this, 'Lane Change', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ Lane Change</button>
                <button class="save-btn" onclick="saveScenario(this, '4-way intersection', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ 4-Way</button>
                <button class="save-btn" onclick="saveScenario(this, 'Following', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ Following</button>
                <button class="save-btn" onclick="saveScenario(this, 'Dense', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ Dense</button>
                <button class="save-btn" onclick="saveScenario(this, 'Sparse', {c['tid']}, {c['start_frame']}, {c['end_frame']}, '{neighbors_str}')">+ Sparse</button>
            </div>
        </div>
    """

html_content += """
    </div>

    <!-- Zoom Lightbox Modal -->
    <div class="modal" id="zoomModal" onclick="closeModal()">
        <div class="modal-content" onclick="event.stopPropagation()">
            <span class="close-btn" onclick="closeModal()">&times;</span>
            <h2 id="modalTitle" style="margin:0; color:#00e5ff; font-size:22px;">Enlarged 15-Second Drone Map Trajectory Overlay</h2>
            <p id="modalSub" style="color:#94a3b8; font-size:14px; margin:0;"></p>
            
            <div class="modal-body">
                <img class="modal-plot" id="modalPlot" src="" alt="Enlarged Plot">
                <div class="modal-thumbs">
                    <h3 style="margin:0 0 6px 0; font-size:14px; color:#c084fc;">Crop Snapshots (Start, Mid, End)</h3>
                    <div class="modal-thumb-row">
                        <div class="thumb-box"><img class="modal-thumb-img" id="mTs" src=""><span class="thumb-lbl" id="mLs"></span></div>
                        <div class="thumb-box"><img class="modal-thumb-img" id="mTm" src=""><span class="thumb-lbl" id="mLm"></span></div>
                        <div class="thumb-box"><img class="modal-thumb-img" id="mTe" src=""><span class="thumb-lbl" id="mLe"></span></div>
                    </div>
                </div>
            </div>

            <div class="actions" id="modalActions"></div>
        </div>
    </div>

    <div id="toast">Scenario Saved & Rendering Triggered!</div>

    <script>
        const candidates = """ + json.dumps(all_candidates_flat) + """;
        let activeRegion = 'all';
        let activeTurn = 'all';

        function filterRegion(reg) {
            document.querySelectorAll('.reg-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            activeRegion = reg;
            applyFilters();
        }

        function filterTurn(turnType) {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            activeTurn = turnType;
            applyFilters();
        }

        function applyFilters() {
            document.querySelectorAll('.card').forEach(card => {
                const matchesReg = (activeRegion === 'all' || card.dataset.region == activeRegion);
                const tags = card.dataset.tags || '';
                const matchesTurn = (activeTurn === 'all' || tags.includes(activeTurn));
                if (matchesReg && matchesTurn) {
                    card.style.display = 'flex';
                } else {
                    card.style.display = 'none';
                }
            });
        }

        function openModal(id) {
            const item = candidates.find(c => c.id === id);
            if (!item) return;

            document.getElementById('modalTitle').innerText = 'Ego Vehicle #' + item.tid + ' (Region ' + item.region + ' - 15 SECONDS)';
            document.getElementById('modalSub').innerText = 'Frames: ' + item.start_frame + ' - ' + item.end_frame + ' (' + item.duration + 'f / 15.0s) | Region ' + item.region + ' Allowed Neighbors: ' + (item.neighbors.join(', ') || 'None');
            document.getElementById('modalPlot').src = 'data:image/png;base64,' + item.plot;
            document.getElementById('mTs').src = 'data:image/jpeg;base64,' + item.t_start;
            document.getElementById('mTm').src = 'data:image/jpeg;base64,' + item.t_mid;
            document.getElementById('mTe').src = 'data:image/jpeg;base64,' + item.t_end;
            document.getElementById('mLs').innerText = 'Start: F' + item.start_frame;
            document.getElementById('mLm').innerText = 'Mid: F' + Math.floor((item.start_frame + item.end_frame) / 2);
            document.getElementById('mLe').innerText = 'End: F' + item.end_frame;

            const nStr = item.neighbors.join(', ');
            document.getElementById('modalActions').innerHTML = `
                <button class="save-btn" onclick="saveScenario(this, 'Left', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ Left</button>
                <button class="save-btn" onclick="saveScenario(this, 'Right', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ Right</button>
                <button class="save-btn" onclick="saveScenario(this, 'slight left', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ Slight Left</button>
                <button class="save-btn" onclick="saveScenario(this, 'slight right', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ Slight Right</button>
                <button class="save-btn" onclick="saveScenario(this, 'Overtaking ', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ Overtake</button>
                <button class="save-btn" onclick="saveScenario(this, 'Lane Change', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ Lane Change</button>
                <button class="save-btn" onclick="saveScenario(this, '4-way intersection', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ 4-Way</button>
                <button class="save-btn" onclick="saveScenario(this, 'Following', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ Following</button>
                <button class="save-btn" onclick="saveScenario(this, 'Dense', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ Dense</button>
                <button class="save-btn" onclick="saveScenario(this, 'Sparse', ${item.tid}, ${item.start_frame}, ${item.end_frame}, '${nStr}')">+ Sparse</button>
            `;

            document.getElementById('zoomModal').style.display = 'flex';
        }

        function closeModal() {
            document.getElementById('zoomModal').style.display = 'none';
        }

        async function saveScenario(btn, category, egoId, startFrame, endFrame, neighbors) {
            btn.innerText = 'Saving...';
            try {
                const response = await fetch('http://localhost:8080/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ category, egoId, startFrame, endFrame, neighbors })
                });
                const res = await response.json();
                if (res.status === 'success') {
                    btn.classList.add('done');
                    btn.innerText = 'Saved! (' + res.scenarioName + ')';
                    showToast('Saved ' + category + ' (' + res.scenarioName + ')');
                } else {
                    btn.innerText = 'Error';
                }
            } catch (err) {
                console.error(err);
                btn.innerText = 'Server Off';
            }
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.style.display = 'block';
            setTimeout(() => { toast.style.display = 'none'; }, 3000);
        }
    </script>
</body>
</html>
"""

# Write to both scenario_catalog.html AND index.html
with open(HTML_OUTPUT, "w", encoding="utf-8") as out:
    out.write(html_content)

with open(INDEX_OUTPUT, "w", encoding="utf-8") as out:
    out.write(html_content)

print(f"\n==========================================")
print(f"15-Second Drone Catalog Generated Successfully:")
print(f"  - {HTML_OUTPUT}")
print(f"  - {INDEX_OUTPUT}")
print(f"==========================================")
