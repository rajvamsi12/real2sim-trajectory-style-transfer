#!/bin/bash
# Ten evaluation runs (Raj, 2026-11-15: "i need 10 runs like we did 2 earlier", "add stock Autopilot for baseline").
# 114 test scenarios x 9 drivers per run:
#   expert, constvel+ttc, rhd_scratch@cl (+ttc), autopilot (recorded path + speed), idm, autopilot_stock, rhd_lead@ep08 (+ttc)
# Run 1 = tag eval_v2, run 2 = eval_v2_run2: only their missing drivers are driven (finished episodes are skipped by the CSV).
# Runs 3-10 = eval_v2_run3 .. eval_v2_run10: all 9 drivers. Traffic Manager seed = run number.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/eval10.log
PORT=2000
S=rhd_scratch@cl
L=rhd_lead@ep08
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_eval10.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
TAGS="eval_v2 eval_v2_run2 eval_v2_run3 eval_v2_run4 eval_v2_run5 eval_v2_run6 eval_v2_run7 eval_v2_run8 eval_v2_run9 eval_v2_run10"
N=0
for TAG in $TAGS; do
  N=$((N + 1))
  export CL_TRAILS=$IL/trails_$TAG
  export CL_TM_SEED=$N
  start_carla
  OK=0
  for attempt in $(seq 1 8); do
    echo "$(date) STEP run$N attempt $attempt" >> $LOG
    $PY -u il/closed_loop_multi.py --port $PORT --split test --tag $TAG \
      --drivers expert constvel+ttc:0.7:0.2 $S $S+ttc:0.7:0.2 autopilot idm autopilot_stock $L $L+ttc:0.7:0.2 >> $LOG 2>&1 \
      && { echo "$(date) STEP run$N DONE" >> $LOG; OK=1; break; }
    echo "$(date) STEP run$N failed, restarting CARLA" >> $LOG
    start_carla
  done
  [ $OK -eq 0 ] && echo "$(date) STEP run$N GAVE UP" >> $LOG
  stop_carla
  $PY -W ignore il/metrics_v2.py $CL_TRAILS --csv $IL/closed_loop_$TAG.csv > $IL/tables_$TAG.log 2>&1
  $PY -W ignore il/dist_v2.py $CL_TRAILS >> $IL/tables_$TAG.log 2>&1
  echo "$(date) run$N tables written" >> $LOG
done
echo "$(date) STEP compare attempt 1" >> $LOG
$PY -W ignore il/compare_runs.py $TAGS > $IL/compare_10runs.log 2>&1
echo "$(date) STEP compare DONE" >> $LOG
echo "EVAL10 FINISHED $(date)" >> $LOG
