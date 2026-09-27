import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8080
SCENARIOS_DIR = "/home/raj/Desktop/carlascripts/Scenariosssss"
PYTHON_BIN = "/home/raj/miniconda3/envs/carla_env/bin/python"
VIDEO_FILE = "/home/raj/Desktop/carlascripts/DJI_0003-002.MP4"
ANNOTATIONS_FILE = "/home/raj/Desktop/carlascripts/default.json"
SCRIPT_PATH = "/home/raj/Desktop/carlascripts/overlay_scenario_bboxes.py"

class CatalogHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_POST(self):
        if self.path == '/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))

            category = req.get('category')
            ego_id = req.get('egoId')
            start_frame = req.get('startFrame')
            end_frame = req.get('endFrame')
            neighbors_raw = req.get('neighbors', '')

            # Parse neighbors
            neighbor_ids = []
            if neighbors_raw and neighbors_raw != 'None':
                for part in neighbors_raw.split(','):
                    part = part.strip()
                    if part.isdigit():
                        neighbor_ids.append(int(part))

            cat_dir = os.path.join(SCENARIOS_DIR, category)
            os.makedirs(cat_dir, exist_ok=True)

            # Find next scenario index
            existing = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d)) and d.startswith('scenario_')]
            next_idx = len(existing) + 1
            s_name = f"scenario_{next_idx:04d}"
            s_dir = os.path.join(cat_dir, s_name)
            os.makedirs(s_dir, exist_ok=True)

            summary_path = os.path.join(s_dir, f"{s_name}_summary.txt")

            # Build vehicles list
            all_vehs = [ego_id] + [n for n in neighbor_ids if n != ego_id]

            content = f"Scenario {next_idx}\n"
            content += f"Start Frame: {start_frame}\n"
            content += f"End Frame: {end_frame}\n"
            content += f"Ego ID: {ego_id}\n"
            content += "Highlight IDs:\n"
            for v in all_vehs:
                if v != ego_id:
                    content += f"- {v}\n"
            content += "Status: COMPLETED\n"
            content += "Vehicles:\n"
            for v in all_vehs:
                content += f"- {v}\n"

            with open(summary_path, "w", encoding="utf-8") as out:
                out.write(content)

            print(f"\n[CatalogServer] Saved {category} -> {s_name} (Frames {start_frame}-{end_frame}, Ego {ego_id})")
            print(f"[CatalogServer] Triggering video render for {summary_path}...")

            # Render video in background
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
            subprocess.Popen(cmd)

            self._set_headers(200)
            res = {"status": "success", "scenarioName": s_name}
            self.wfile.write(json.dumps(res).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"status": "not found"}).encode('utf-8'))

def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, CatalogHandler)
    print(f"==========================================")
    print(f"Catalog HTTP Receiver Server Running on http://localhost:{PORT}")
    print(f"Ready for 1-Click Scenario Creation!")
    print(f"==========================================")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
