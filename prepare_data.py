# --- prepare_data.py ---
import os
import json
import pandas as pd

# The root directory where your scenarios3.py script saved the data
ROOT_DIR = "_scenario_output"
OUTPUT_CSV = "metadata.csv"

data_records = []

print("Starting data preparation...")

# Walk through the directory structure created by scenarios3.py
for scenario_folder in sorted(os.listdir(ROOT_DIR)):
    scenario_path = os.path.join(ROOT_DIR, scenario_folder)
    if not os.path.isdir(scenario_path):
        continue

    for vehicle_folder in sorted(os.listdir(scenario_path)):
        vehicle_path = os.path.join(scenario_path, vehicle_folder)
        if not os.path.isdir(vehicle_path):
            continue

        # Define paths to the data files
        image_dir = os.path.join(vehicle_path, "images")
        linear_vel_path = os.path.join(vehicle_path, "linear_velocity.json")
        angular_vel_path = os.path.join(vehicle_path, "angular_velocity.json")

        if not all([os.path.isdir(image_dir), os.path.exists(linear_vel_path), os.path.exists(angular_vel_path)]):
            continue

        # Load the velocity data
        try:
            with open(linear_vel_path) as f:
                linear_data = json.load(f)
            with open(angular_vel_path) as f:
                angular_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load data for {vehicle_path}. Skipping.")
            continue
        
        # Match images with velocity data assuming they are in the same order
        # This is a safe assumption given the data generation logic
        image_files = sorted(os.listdir(image_dir))
        
        num_images = len(image_files)
        num_records = min(len(linear_data), len(angular_data))

        if num_images != num_records:
             print(f"Warning: Mismatch in data for {vehicle_path}. Images: {num_images}, Records: {num_records}. Using minimum.")

        for i in range(min(num_images, num_records)):
            image_file = image_files[i]
            if image_file.endswith(".png"):
                speed = linear_data[i]['speed_ms']
                # The 'z' component of angular velocity is our proxy for steering
                steering = angular_data[i]['z'] 
                
                full_image_path = os.path.join(image_dir, image_file)
                data_records.append([full_image_path, speed, steering])

# Create a pandas DataFrame and save to CSV
df = pd.DataFrame(data_records, columns=["image_path", "speed", "steering"])
df.to_csv(OUTPUT_CSV, index=False)

print(f"Data preparation complete! Created {OUTPUT_CSV} with {len(df)} records.")