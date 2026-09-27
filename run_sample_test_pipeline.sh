#!/bin/bash
set -e

if [ -f "/home/raj/miniconda3/envs/carla_env/bin/python" ]; then
    PYTHON_BIN="/home/raj/miniconda3/envs/carla_env/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "=========================================================="
echo "Starting 1-Command Zero-Shot LLM 2-Video Sample Pipeline"
echo "=========================================================="

echo -e "\n--- Phase 1: Exporting 15s Trajectory JSON Prompts ---"
$PYTHON_BIN export_llm_trajectories.py

echo -e "\n--- Phase 2: Zero-Shot Pre-Trained LLM API Reasoning (Zero Training) ---"
$PYTHON_BIN query_zero_shot_llm.py

echo -e "\n--- Phase 3: Creating 2-Video Sample Subfolders per Category ---"
$PYTHON_BIN create_sample_scenarios.py

echo -e "\n--- Phase 4: Batch Rendering 20 Sample Drone Videos ---"
$PYTHON_BIN render_sample_20_videos.py

echo -e "\n=========================================================="
echo "Pipeline Execution Complete! 20 Sample Videos Ready."
echo "=========================================================="
