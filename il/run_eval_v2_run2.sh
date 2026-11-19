#!/bin/bash
# Evaluation run 2 (Raj, 2026-11-15 "store this as run 1 and run this again"): the same 114 test scenarios x 6 drivers; run 1 = tag eval_v2.
#   expert, constvel+ttc 0.7/0.2, rhd_scratch@cl, rhd_scratch@cl+ttc 0.7/0.2, autopilot, idm   (autopilot and idm without TTC)
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/eval_v2_run2.log
PORT=2000
M=rhd_scratch@cl
export CL_TRAILS=$IL/trails_eval_v2_run2
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_eval_v2_run2.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
start_carla
OK=0
for attempt in $(seq 1 8); do
  echo "$(date) STEP eval attempt $attempt" >> $LOG
  $PY -u il/closed_loop_multi.py --port $PORT --split test --tag eval_v2_run2 \
    --drivers expert constvel+ttc:0.7:0.2 $M $M+ttc:0.7:0.2 autopilot idm >> $LOG 2>&1 && { echo "$(date) STEP eval DONE" >> $LOG; OK=1; break; }
  echo "$(date) STEP eval failed, restarting CARLA" >> $LOG
  start_carla
done
[ $OK -eq 0 ] && echo "$(date) STEP eval GAVE UP" >> $LOG
stop_carla
echo "$(date) STEP tables attempt 1" >> $LOG
$PY -W ignore il/metrics_v2.py $CL_TRAILS --csv $IL/closed_loop_eval_v2_run2.csv >> $LOG 2>&1
$PY -W ignore il/dist_v2.py $CL_TRAILS >> $LOG 2>&1
echo "$(date) STEP tables DONE" >> $LOG
$PY -W ignore il/compare_runs.py eval_v2 eval_v2_run2 >> $LOG 2>&1
echo "EVAL V2 RUN2 FINISHED $(date)" >> $LOG
