#!/bin/bash
# Convenient wrapper script located in debug/ folder
# Executes debug/batch_process_scenarios.py using carla_env python

PYTHON_EXEC="/home/raj/miniconda3/envs/carla_env/bin/python"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

$PYTHON_EXEC debug/batch_process_scenarios.py "$@"
