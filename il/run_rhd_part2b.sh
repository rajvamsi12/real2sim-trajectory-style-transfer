#!/bin/bash
# Part 2b of the RHD rebuild (2026-11-10). Same as part 2 after the glitch filter, but the CARLA checks use a headless
# server (-nullrhi: 0.7 s/scenario last time vs 4.7 s with rendering); only the 500-scenario video uses -RenderOffScreen.
# The check resumes from replay_cureE.csv.
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
start_carla() {  # $1 = headless | render
  stop_carla
  if [ "$1" = headless ]; then
    ~/CARLA_0.9.15/CarlaUE4.sh -nullrhi -nosound -carla-rpc-port=$PORT > /tmp/carla_rhd.log 2>&1 &
  else
    ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_rhd.log 2>&1 &
  fi
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
  CARLA_MODE=$1
}
carla_step() {  # mode, name, command...: fresh CARLA in that mode, retry on failure (max 10)
  local mode=$1 name=$2; shift 2
  [ "$CARLA_MODE" = "$mode" ] || start_carla $mode
  for attempt in $(seq 1 10); do
    (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || start_carla $mode
    echo "$(date) STEP $name attempt $attempt ($mode CARLA)" >> $LOG
    "$@" >> $LOG 2>&1 && { echo "$(date) STEP $name DONE" >> $LOG; return 0; }
    echo "$(date) STEP $name failed, restarting CARLA" >> $LOG
    start_carla $mode
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
CARLA_MODE=none
carla_step headless check $PY -u il/replay_scenarios.py --yaw path --drive velocity --tag cureE
carla_step render video500 $PY -u il/render_compare_video.py --showcase 500 --clips-dir $IL/video500_clips
plain_step video500_join $PY -u il/render_compare_video.py --join --clips-dir $IL/video500_clips --out $IL/drone_vs_carla_500.mp4
echo "RHD PART 2 FINISHED $(date)" >> $LOG
