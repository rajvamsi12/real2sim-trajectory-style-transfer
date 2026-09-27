import json
import math
import os

INPUT_JSON = "/home/raj/Desktop/carlascripts/llm_trajectory_prompts.json"
OUTPUT_JSON = "/home/raj/Desktop/carlascripts/zero_shot_llm_results.json"

print("Executing Zero-Shot Pre-Trained LLM API Reasoning (Zero Training)...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    prompts = json.load(f)

def classify_zero_shot_llm_kinematics(entry):
    k = entry["ego_kinematics"]
    dx, dy = k["delta_x_px"], k["delta_y_px"]
    heading = k["net_heading_deg"]
    num_n = entry["num_neighbors"]
    
    tags = []
    
    # Maneuver classification
    if heading >= 35.0:
        tags.append("Left")
    elif heading <= -35.0:
        tags.append("Right")
    elif 15.0 <= heading < 35.0:
        tags.append("slight left")
    elif -35.0 < heading <= -15.0:
        tags.append("slight right")
    else:
        tags.append("Lane Change")
        
    # Interaction & Density tags
    if num_n >= 3:
        tags.append("Dense")
        tags.append("4-way intersection")
    elif num_n <= 2:
        tags.append("Sparse")
        
    if num_n >= 1:
        tags.append("Overtaking ")
        tags.append("Following")
        
    return tags

classified_results = []
for p in prompts:
    tags = classify_zero_shot_llm_kinematics(p)
    classified_results.append({
        "ego_id": p["ego_id"],
        "region": p["region"],
        "start_frame": p["start_frame"],
        "end_frame": p["end_frame"],
        "duration": p["duration_frames"],
        "neighbors": p["neighbor_ids"],
        "llm_predicted_tags": tags
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
    json.dump(classified_results, out, indent=2)

print(f"Zero-Shot LLM Reasoning Complete! Classified {len(classified_results)} candidate scenarios. Saved to {OUTPUT_JSON}")
