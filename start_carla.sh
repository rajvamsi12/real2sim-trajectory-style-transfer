#!/bin/bash
# CARLA 0.9.15 Launcher & Environment Helper

export CARLA_ROOT="/home/raj/CARLA_0.9.15"
CONDA_ACTIVATE="/home/raj/miniconda3/bin/activate"

# Activate carla_env automatically
if [ -f "$CONDA_ACTIVATE" ]; then
    source "$CONDA_ACTIVATE" carla_env
fi

if [ "$1" == "--server" ]; then
    echo "Starting CARLA Simulator Server (Offscreen mode)..."
    $CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -nosound
elif [ "$1" == "--display" ]; then
    echo "Starting CARLA Simulator Server (Display window mode)..."
    $CARLA_ROOT/CarlaUE4.sh
else
    echo "=== CARLA 0.9.15 Helper ==="
    echo "Usage:"
    echo "  ./start_carla.sh --display   (Start CARLA simulator server with display)"
    echo "  ./start_carla.sh --server    (Start CARLA simulator server headless/offscreen)"
    echo ""
    echo "To activate Python environment in shell:"
    echo "  . /home/raj/miniconda3/bin/activate carla_env"
    echo "  OR"
    echo "  source /home/raj/miniconda3/bin/activate carla_env"
    echo ""
    echo "To run scripts directly:"
    echo "  /home/raj/miniconda3/envs/carla_env/bin/python carlascripts/map_check.py"
fi
