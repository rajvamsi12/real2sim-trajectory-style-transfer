#!/bin/bash
# Part 3 of the RHD rebuild (2026-11-10): augmentation (3 scene noise + 4 ego offset + 3 ego speed) and its CARLA check,
# in parallel with the 500-scenario video of part 2b. Waits for part 2b's check of the base scenarios (augmentation only
# copies collision-free ones), then uses its own headless CARLA on port 2010 and its own log. Recording waits for Raj.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/rhd_aug.log
PORT=2010
carla_pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(carla_pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(carla_pids)" ] && break; sleep 1; done
  for p in $(carla_pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -nullrhi -nosound -carla-rpc-port=$PORT > /tmp/carla_rhd_aug.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
echo "$(date) STEP wait_base_check" >> $LOG
until grep -q "STEP check DONE" $IL/rhd_rebuild.log; do
  grep -q "STEP check GAVE UP" $IL/rhd_rebuild.log && { echo "$(date) STEP wait_base_check FAILED" >> $LOG; exit 1; }
  sleep 20
done
echo "$(date) STEP wait_base_check DONE" >> $LOG
echo "$(date) STEP augment" >> $LOG
$PY -u il/augment_scenarios.py >> $LOG 2>&1 || { echo "$(date) STEP augment FAILED" >> $LOG; exit 1; }
echo "$(date) STEP augment DONE" >> $LOG
for attempt in $(seq 1 10); do
  (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || start_carla
  echo "$(date) STEP aug_check attempt $attempt (headless CARLA, port $PORT)" >> $LOG
  $PY -u il/replay_scenarios.py --scenarios $IL/scenarios_aug.json --yaw path --drive velocity --tag aug_cureE --port $PORT \
    >> $LOG 2>&1 && { echo "$(date) STEP aug_check DONE" >> $LOG; stop_carla; echo "RHD PART 3 FINISHED $(date)" >> $LOG; exit 0; }
  echo "$(date) STEP aug_check failed, restarting CARLA" >> $LOG
  start_carla
done
echo "$(date) STEP aug_check GAVE UP" >> $LOG
stop_carla
