#!/bin/bash
# Rebuild everything up to augmentation with the right-hand-traffic mapping (timemethod.py switched 2026-11-10), same
# steps as last time. Old LHD outputs: Scenariosssss/IL_lhd_archive/. CARLA steps restart CARLA (port 2000) on a crash
# and resume (replay from its CSV, render from its clips folder).
#   1 usable windows        debug/count_usable_scenarios.py     -> Scenariosssss/usable_scenarios.csv
#   2 8 s scenarios         il/build_scenarios_8s.py            -> Scenariosssss/IL/scenarios_8s.json
#   3 CARLA check (cure E)  il/replay_scenarios.py              -> Scenariosssss/IL/replay_cureE.csv
#   4 500-scenario video    il/render_compare_video.py          -> Scenariosssss/IL/drone_vs_carla_500.mp4
#   5 augmentation x10      il/augment_scenarios.py             -> Scenariosssss/IL/scenarios_aug.json
#   6 CARLA check of copies il/replay_scenarios.py              -> Scenariosssss/IL/replay_aug_cureE.csv
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/rhd_rebuild.log
PORT=2000
carla_pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(carla_pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(carla_pids)" ] && break; sleep 1; done
  for p in $(carla_pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_rhd.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
carla_step() {  # name, command...: retry with a fresh CARLA until the command succeeds (max 10)
  local name=$1; shift
  for attempt in $(seq 1 10); do
    (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || start_carla
    echo "$(date) STEP $name attempt $attempt" >> $LOG
    "$@" >> $LOG 2>&1 && { echo "$(date) STEP $name DONE" >> $LOG; return 0; }
    echo "$(date) STEP $name failed, restarting CARLA" >> $LOG
    start_carla
  done
  echo "$(date) STEP $name GAVE UP" >> $LOG
  exit 1
}
plain_step() {
  local name=$1; shift
  echo "$(date) STEP $name" >> $LOG
  "$@" >> $LOG 2>&1 || { echo "$(date) STEP $name FAILED" >> $LOG; exit 1; }
  echo "$(date) STEP $name DONE" >> $LOG
}
plain_step usable $PY -u debug/count_usable_scenarios.py
plain_step build $PY -u il/build_scenarios_8s.py
carla_step check $PY -u il/replay_scenarios.py --yaw path --drive velocity --tag cureE
carla_step video500 $PY -u il/render_compare_video.py --showcase 500 --clips-dir $IL/video500_clips
plain_step video500_join $PY -u il/render_compare_video.py --join --clips-dir $IL/video500_clips --out $IL/drone_vs_carla_500.mp4
plain_step augment $PY -u il/augment_scenarios.py
carla_step aug_check $PY -u il/replay_scenarios.py --scenarios $IL/scenarios_aug.json --yaw path --drive velocity --tag aug_cureE
stop_carla
echo "RHD REBUILD FINISHED $(date)" >> $LOG
