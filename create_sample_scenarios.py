import json
import os
import shutil

INPUT_JSON = "/home/raj/Desktop/carlascripts/zero_shot_llm_results.json"
SCENARIOS_DIR = "/home/raj/Desktop/carlascripts/Scenariosssss"

categories = [
    "4-way intersection",
    "Dense",
    "Following",
    "Lane Change",
    "Left",
    "Overtaking ",
    "Right",
    "Sparse",
    "slight left",
    "slight right"
]

print("Cleaning existing folders and creating 2-video sample test batch per category...")
for cat in categories:
    cat_dir = os.path.join(SCENARIOS_DIR, cat)
    if os.path.exists(cat_dir):
        shutil.rmtree(cat_dir)
    os.makedirs(cat_dir, exist_ok=True)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    results = json.load(f)

scenarios_by_cat = {cat: [] for cat in categories}

for item in results:
    tags = item["llm_predicted_tags"]
    for cat in categories:
        if cat in tags and len(scenarios_by_cat[cat]) < 2:
            scenarios_by_cat[cat].append(item)

# Fallback check to ensure 2 scenarios per category
for cat in categories:
    if len(scenarios_by_cat[cat]) < 2:
        for item in results:
            if item not in scenarios_by_cat[cat] and len(scenarios_by_cat[cat]) < 2:
                scenarios_by_cat[cat].append(item)

total_sample_summaries = 0

for cat in categories:
    cat_dir = os.path.join(SCENARIOS_DIR, cat)
    selected = scenarios_by_cat[cat][:2]
    
    for idx, scen in enumerate(selected, 1):
        s_name = f"scenario_{idx:04d}"
        s_dir = os.path.join(cat_dir, s_name)
        os.makedirs(s_dir, exist_ok=True)
        
        summary_path = os.path.join(s_dir, f"{s_name}_summary.txt")
        ego = scen["ego_id"]
        sf = scen["start_frame"]
        ef = scen["end_frame"]
        neighbors = scen["neighbors"]
        all_vehs = [ego] + [n for n in neighbors if n != ego]
        
        content = f"Scenario {idx}\n"
        content += f"Start Frame: {sf}\n"
        content += f"End Frame: {ef}\n"
        content += f"Ego ID: {ego}\n"
        content += "Highlight IDs:\n"
        for v in all_vehs:
            if v != ego:
                content += f"- {v}\n"
        content += "Status: COMPLETED\n"
        content += "Vehicles:\n"
        for v in all_vehs:
            content += f"- {v}\n"
            
        with open(summary_path, "w", encoding="utf-8") as out:
            out.write(content)
        total_sample_summaries += 1
        print(f"Created {cat} -> {s_name} (Frames {sf}-{ef}, Ego {ego})")

print(f"\nWrote {total_sample_summaries} sample summary text files (2 per category).")
