#!/bin/bash
# Evaluation of the lead-vehicle-input model (2026-11-15): 114 test scenarios, rhd_lead@ep08 with and without the TTC layer (0.7/0.2).
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/eval_lead.log
PORT=2000
M=rhd_lead@ep08
export CL_TRAILS=$IL/trails_eval_lead
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_eval_lead.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
start_carla
OK=0
for attempt in $(seq 1 8); do
  echo "$(date) STEP eval attempt $attempt" >> $LOG
  $PY -u il/closed_loop_multi.py --port $PORT --split test --tag eval_lead \
    --drivers $M $M+ttc:0.7:0.2 >> $LOG 2>&1 && { echo "$(date) STEP eval DONE" >> $LOG; OK=1; break; }
  echo "$(date) STEP eval failed, restarting CARLA" >> $LOG
  start_carla
done
[ $OK -eq 0 ] && echo "$(date) STEP eval GAVE UP" >> $LOG
stop_carla
echo "$(date) STEP tables attempt 1" >> $LOG
$PY -W ignore il/metrics_v2.py $CL_TRAILS --csv $IL/closed_loop_eval_lead.csv >> $LOG 2>&1
$PY -W ignore il/dist_v2.py $CL_TRAILS >> $LOG 2>&1
echo "$(date) STEP tables DONE" >> $LOG
echo "EVAL LEAD FINISHED $(date)" >> $LOG
