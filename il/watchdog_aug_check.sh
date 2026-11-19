#!/bin/bash
# Keeps the step-4 CARLA check (replay_aug_cureE) running until its log says "saved": if the replay process
# or its CARLA server (port 2000) dies, restart both; the replay resumes from replay_aug_cureE.csv.
cd ~/Desktop/carlascripts
L=Scenariosssss/IL/replay_aug_cureE.log
W=Scenariosssss/IL/watchdog_aug_check.log
PY=~/miniconda3/envs/carla_env/bin/python
while ! grep -q "saved .*replay_" $L 2>/dev/null; do
  sleep 60
  grep -q "saved .*replay_" $L 2>/dev/null && break
  if ! ps -eo args | grep -v grep | grep -q "replay_scenarios.py --scenarios Scenariosssss/IL/scenarios_aug.json"; then
    echo "$(date) replay not running -> restarting CARLA (2000) and replay" >> $W
    for p in $(ps -eo pid,args | grep "[C]arlaUE4-Linux-Shipping" | grep -v "rpc-port=2010" | awk "{print \$1}"); do kill $p; done
    sleep 5
    ~/CARLA_0.9.15/CarlaUE4.sh -nullrhi -nosound -carla-rpc-port=2000 > /tmp/carla_aug.log 2>&1 &
    for i in $(seq 1 60); do sleep 2; (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null && break; done
    sleep 8
    $PY -u il/replay_scenarios.py --scenarios Scenariosssss/IL/scenarios_aug.json --yaw path --drive velocity --tag aug_cureE >> $L 2>&1 &
  fi
done
echo "$(date) check finished" >> $W
