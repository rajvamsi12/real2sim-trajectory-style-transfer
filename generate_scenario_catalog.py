import base64
import io
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_FILE = "/home/raj/Desktop/carlascripts/default.json"
VIDEO_FILE = "/home/raj/Desktop/carlascripts/DJI_0003-002.MP4"
TIDS_FILE = "/home/raj/Desktop/carlascripts/tids.txt"
HTML_OUTPUT = "/home/raj/Desktop/carlascripts/scenario_catalog.html"

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

print("Loading annotation dataset...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

items = data.get("items", [])
total_frames = len(items)
print(f"Total frames: {total_frames}")

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
        dx = p2[1] - p1[1]
        dy = -(p2[2] - p1[2]) # Drone Y-axis inversion
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

# Open Video Capture for thumbnail extraction
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
    
    crop_resized = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_LINEAR)
    _, buffer = cv2.imencode('.jpg', crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

# Color palette for neighbor vehicle trajectory lines
NEIGHBOR_COLORS = ['#f97316', '#16a34a', '#a855f7', '#eab308', '#ec4899', '#14b8a6', '#f43f5e']

def generate_multi_vehicle_plot(ego_tid, ego_pts, neighbors_dict):
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=120)
    
    # Plot Ego
    ego_xs = [p[1] for p in ego_pts]
    ego_ys = [p[2] for p in ego_pts]
    ax.plot(ego_xs, ego_ys, color='#0284c7', linewidth=3.0, label=f'EGO {ego_tid}', zorder=10)
    ax.scatter(ego_xs[0], ego_ys[0], color='#22c55e', s=70, zorder=12, label='Start')
    ax.scatter(ego_xs[-1], ego_ys[-1], color='#ef4444', s=70, zorder=12, label='End')
    
    # Plot Neighbors
    for idx, (n_tid, n_pts) in enumerate(neighbors_dict.items()):
        if not n_pts:
            continue
        n_xs = [p[0] for p in n_pts]
        n_ys = [p[1] for p in n_pts]
        color = NEIGHBOR_COLORS[idx % len(NEIGHBOR_COLORS)]
        ax.plot(n_xs, n_ys, color=color, linewidth=2.0, linestyle='--', label=f'VEH {n_tid}', zorder=5)
        ax.scatter(n_xs[0], n_ys[0], color=color, s=35, zorder=6)
    
    ax.invert_yaxis() # Invert Y for drone pixel space
    ax.set_title(f"Ego #{ego_tid} + Neighbors (X,Y) Path", fontsize=10, fontweight='bold', pad=4)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.7)
    ax.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

candidates = []
print("Building multi-vehicle catalog cards across all dataset regions...")

# Select candidate moving tracks across all 9 regions
all_candidate_tids = sorted(list(moving_tracks.keys()))
print(f"Processing {len(all_candidate_tids)} candidate tracks...")

for idx, tid in enumerate(all_candidate_tids, 1):
    pts = moving_tracks[tid]
    sf, ef = pts[0][0], pts[-1][0]
    
    # Match Region
    reg_num = next((r for r, (rs, re) in FRAME_REGIONS.items() if rs <= sf <= re), 5)
    allowed_tids = region_tids_map.get(reg_num, set())

    # Build neighbor trajectory dictionary
    neighbors_dict = {}
    for f in range(sf, ef + 1):
        for t, (cx, cy, w, h) in frame_tracks.get(f, {}).items():
            if t in allowed_tids and t != tid:
                if t not in neighbors_dict:
                    neighbors_dict[t] = []
                neighbors_dict[t].append((cx, cy))
                
    # Keep top 4 active neighbors
    sorted_neighbors = sorted(neighbors_dict.items(), key=lambda x: len(x[1]), reverse=True)[:4]
    selected_neighbors_dict = dict(sorted_neighbors)
    neighbor_ids = sorted(list(selected_neighbors_dict.keys()))
    
    turn_type = get_track_turn_drone(pts)
    plot_b64 = generate_multi_vehicle_plot(tid, pts, selected_neighbors_dict)
    
    # Extract 3 thumbnails
    mid_idx = len(pts) // 2
    t_start = get_crop_thumbnail(pts[0][0], pts[0][1], pts[0][2])
    t_mid = get_crop_thumbnail(pts[mid_idx][0], pts[mid_idx][1], pts[mid_idx][2])
    t_end = get_crop_thumbnail(pts[-1][0], pts[-1][1], pts[-1][2])
    
    candidates.append({
        "id": idx,
        "tid": tid,
        "start_frame": sf,
        "end_frame": ef,
        "region": reg_num,
        "turn": turn_type,
        "neighbors": neighbor_ids,
        "plot": plot_b64,
        "t_start": t_start,
        "t_mid": t_mid,
        "t_end": t_end
    })
    
    if idx % 20 == 0 or idx == len(all_candidate_tids):
        print(f"  Processed {idx}/{len(all_candidate_tids)} candidates...")

capture.release()

# HTML Template
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Vehicle Trajectory Catalog & Zoom Inspector</title>
    <style>
        * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background: #0b0f19; color: #f8fafc; margin: 0; padding: 24px; }}
        header {{ margin-bottom: 24px; border-bottom: 1px solid #1e293b; padding-bottom: 16px; display: flex; justify-content: space-between; align-items: flex-end; }}
        h1 {{ margin: 0 0 6px 0; color: #38bdf8; font-size: 26px; }}
        p {{ margin: 0; color: #94a3b8; font-size: 14px; }}
        .badge-count {{ background: #0284c7; color: #fff; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 13px; }}
        .filters {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }}
        .filter-btn {{ background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600; transition: all 0.2s; }}
        .filter-btn:hover, .filter-btn.active {{ background: #0284c7; color: #ffffff; border-color: #38bdf8; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 20px; }}
        .card {{ background: #151d30; border: 1px solid #24324d; border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 12px; cursor: pointer; transition: transform 0.2s, border-color 0.2s; }}
        .card:hover {{ transform: translateY(-3px); border-color: #38bdf8; box-shadow: 0 10px 20px -5px rgba(0,0,0,0.5); }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #24324d; padding-bottom: 8px; }}
        .track-id {{ font-size: 18px; font-weight: 700; color: #38bdf8; }}
        .meta-tag {{ background: #0f172a; color: #c084fc; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; border: 1px solid #581c87; }}
        .visuals {{ display: flex; gap: 12px; align-items: center; }}
        .plot-img {{ width: 160px; height: 130px; background: #0b0f19; border-radius: 8px; padding: 4px; border: 1px solid #334155; object-fit: contain; }}
        .thumbs {{ display: flex; gap: 6px; flex: 1; overflow-x: auto; }}
        .thumb-box {{ display: flex; flex-direction: column; align-items: center; gap: 3px; }}
        .thumb-img {{ width: 85px; height: 85px; object-fit: cover; border-radius: 8px; border: 1px solid #334155; }}
        .thumb-lbl {{ font-size: 10px; color: #94a3b8; font-weight: 600; }}
        .details {{ font-size: 12px; color: #cbd5e1; background: #0b0f19; padding: 8px 12px; border-radius: 6px; display: flex; justify-content: space-between; border: 1px solid #1e293b; }}
        .actions {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
        .save-btn {{ background: #0b0f19; color: #38bdf8; border: 1px solid #0284c7; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; cursor: pointer; transition: all 0.15s; }}
        .save-btn:hover {{ background: #0284c7; color: #ffffff; }}
        .save-btn.done {{ background: #16a34a; color: #ffffff; border-color: #22c55e; }}
        
        /* Zoom Lightbox Modal */
        .modal {{ display: none; position: fixed; inset: 0; background: rgba(0, 0, 0, 0.85); backdrop-filter: blur(8px); z-index: 1000; align-items: center; justify-content: center; padding: 24px; }}
        .modal-content {{ background: #151d30; border: 1px solid #38bdf8; border-radius: 16px; width: 100%; max-width: 900px; padding: 24px; display: flex; flex-direction: column; gap: 20px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7); position: relative; }}
        .close-btn {{ position: absolute; top: 16px; right: 20px; color: #94a3b8; font-size: 28px; font-weight: bold; cursor: pointer; transition: color 0.2s; }}
        .close-btn:hover {{ color: #ef4444; }}
        .modal-body {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .modal-plot {{ width: 440px; height: 330px; object-fit: contain; background: #0b0f19; border-radius: 12px; border: 1px solid #334155; padding: 8px; }}
        .modal-thumbs {{ display: flex; flex-direction: column; gap: 12px; flex: 1; }}
        .modal-thumb-row {{ display: flex; gap: 12px; }}
        .modal-thumb-img {{ width: 120px; height: 120px; border-radius: 8px; border: 1px solid #475569; object-fit: cover; }}
        #toast {{ position: fixed; bottom: 24px; right: 24px; background: #16a34a; color: #fff; padding: 14px 24px; border-radius: 10px; font-weight: 700; display: none; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); z-index: 2000; }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Multi-Vehicle Trajectory Catalog & Zoom Inspector</h1>
            <p>Ego path (Blue) + Neighbors (Colored Dashed Lines). Click any card to zoom in!</p>
        </div>
        <span class="badge-count">{len(candidates)} Candidate Tracks</span>
    </header>

    <div class="filters">
        <button class="filter-btn active" onclick="filterCards('all')">All ({len(candidates)})</button>
        <button class="filter-btn" onclick="filterCards('left')">Left Turns</button>
        <button class="filter-btn" onclick="filterCards('right')">Right Turns</button>
        <button class="filter-btn" onclick="filterCards('slight_left')">Slight Left</button>
        <button class="filter-btn" onclick="filterCards('slight_right')">Slight Right</button>
        <button class="filter-btn" onclick="filterCards('straight')">Straight / Lane Change</button>
    </div>

    <div class="grid" id="catalogGrid">
"""

for c in candidates:
    neighbors_str = ", ".join(str(n) for n in c["neighbors"]) or "None"
    
    html_content += f"""
        <div class="card" data-turn="{c['turn']}" onclick="openModal({c['id']})">
            <div class="card-header">
                <span class="track-id">Ego Vehicle #{c['tid']}</span>
                <span class="meta-tag">Region {c['region']} ({c['turn'].replace('_', ' ').upper()})</span>
            </div>
            
            <div class="visuals">
                <img class="plot-img" id="plot_{c['id']}" src="data:image/png;base64,{c['plot']}" alt="Multi-Vehicle Plot">
                <div class="thumbs">
                    <div class="thumb-box">
                        <img class="thumb-img" id="ts_{c['id']}" src="data:image/jpeg;base64,{c['t_start']}" alt="Start">
                        <span class="thumb-lbl">Start: F{c['start_frame']}</span>
                    </div>
                    <div class="thumb-box">
                        <img class="thumb-img" id="tm_{c['id']}" src="data:image/jpeg;base64,{c['t_mid']}" alt="Mid">
                        <span class="thumb-lbl">Mid: F{(c['start_frame']+c['end_frame'])//2}</span>
                    </div>
                    <div class="thumb-box">
                        <img class="thumb-img" id="te_{c['id']}" src="data:image/jpeg;base64,{c['t_end']}" alt="End">
                        <span class="thumb-lbl">End: F{c['end_frame']}</span>
                    </div>
                </div>
            </div>

            <div class="details">
                <span><b>Frames:</b> {c['start_frame']} - {c['end_frame']} ({c['end_frame']-c['start_frame']+1}f)</span>
                <span><b>Allowed Neighbors:</b> {neighbors_str}</span>
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
            <h2 id="modalTitle" style="margin:0; color:#38bdf8; font-size:22px;">Enlarged Multi-Vehicle Trajectory</h2>
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
        const candidates = """ + json.dumps(candidates) + """;

        function filterCards(turnType) {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            
            document.querySelectorAll('.card').forEach(card => {
                if (turnType === 'all' || card.dataset.turn === turnType) {
                    card.style.display = 'flex';
                } else {
                    card.style.display = 'none';
                }
            });
        }

        function openModal(id) {
            const item = candidates.find(c => c.id === id);
            if (!item) return;

            document.getElementById('modalTitle').innerText = 'Ego Vehicle #' + item.tid + ' (Region ' + item.region + ' - ' + item.turn.replace('_', ' ').toUpperCase() + ')';
            document.getElementById('modalSub').innerText = 'Frames: ' + item.start_frame + ' - ' + item.end_frame + ' | Allowed Neighbors: ' + (item.neighbors.join(', ') || 'None');
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

with open(HTML_OUTPUT, "w", encoding="utf-8") as out:
    out.write(html_content)

print(f"\n==========================================")
print(f"Multi-Vehicle Catalog HTML Generated Successfully: {HTML_OUTPUT}")
print(f"==========================================")
