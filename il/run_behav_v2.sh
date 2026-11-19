#!/bin/bash
# Behaviour check v2 trial (Raj, 2026-11-14): 20 random checkable test scenarios, 4 drivers, driven paths saved for il/behaviour_v2.py
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/behav_v2.log
PORT=2000
M=rhd_scratch@cl
IDS="00002 00010 00060 00168 00219 00307 00381 00424 00430 00473 00520 00570 00613 00669 00835 01106 01182 01224 01524 01532"
export CL_TRAILS=$IL/trails_behav_v2
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_behav_v2.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
start_carla
for attempt in $(seq 1 4); do
  echo "$(date) STEP run attempt $attempt" >> $LOG
  $PY -u il/closed_loop_multi.py --port $PORT --split test --tag behav_v2 --ids $IDS \
    --drivers expert constvel+ttc:0.7:0.2 $M $M+ttc:0.7:0.2 >> $LOG 2>&1 && { echo "$(date) STEP run DONE" >> $LOG; break; }
  echo "$(date) STEP run failed, restarting CARLA" >> $LOG
  start_carla
done
stop_carla
$PY il/behaviour_v2.py $CL_TRAILS --tol 5 10 >> $LOG 2>&1
echo "BEHAV V2 FINISHED $(date)" >> $LOG
