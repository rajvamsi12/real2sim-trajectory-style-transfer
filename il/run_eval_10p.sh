#!/bin/bash
# Ten evaluation runs, two lanes in parallel (Raj, 2026-11-15: "yeah set it up"). Same tags, CSVs and trails as il/run_eval_10.sh;
# finished episodes are skipped.
#   lane A, CARLA port 2000, Epic, rendering on : the 4 model drivers (camera + LiDAR)
#   lane B, CARLA port 2100, Low, rendering off : expert, constvel+ttc, autopilot, idm, autopilot_stock (no sensors)
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/eval10p.log
S=rhd_scratch@cl
L=rhd_lead@ep08
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$1" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids $1); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids $1)" ] && break; sleep 1; done
  for p in $(pids $1); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$1) 2>/dev/null || break; sleep 1; done
}
start_carla() {   # port quality
  stop_carla $1
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=$2 -carla-rpc-port=$1 > /tmp/carla_eval10p_$1.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$1) 2>/dev/null && break; done
  sleep 15
}
lane() {   # name port quality tag extra-args drivers...
  local NAME=$1 PORT=$2 Q=$3 TAG=$4 EXTRA=$5; shift 5
  local LL=$IL/eval10p_lane$NAME.log OK=0
  start_carla $PORT $Q
  for attempt in $(seq 1 8); do
    echo "$(date) $TAG lane $NAME attempt $attempt" >> $LL
    $PY -u il/closed_loop_multi.py --port $PORT --split test --tag $TAG $EXTRA --drivers "$@" >> $LL 2>&1 && { OK=1; break; }
    echo "$(date) $TAG lane $NAME failed, restarting CARLA" >> $LL
    start_carla $PORT $Q
  done
  stop_carla $PORT
  [ $OK -eq 1 ] && echo "$(date) $TAG lane $NAME DONE" >> $LOG || echo "$(date) $TAG lane $NAME GAVE UP" >> $LOG
}
TAGS="eval_v2 eval_v2_run2 eval_v2_run3 eval_v2_run4 eval_v2_run5 eval_v2_run6 eval_v2_run7 eval_v2_run8 eval_v2_run9 eval_v2_run10"
N=0
for TAG in $TAGS; do
  N=$((N + 1))
  [ $N -lt 10 ] && continue                                  # runs 1 to 9 are complete
  CSV=$IL/closed_loop_$TAG.csv
  [ -f $CSV ] || head -1 $IL/closed_loop_eval_v2.csv > $CSV   # header first, so the two lanes only ever append rows
  export CL_TRAILS=$IL/trails_$TAG
  export CL_TM_SEED=$N
  echo "$(date) STEP run$N attempt 1" >> $LOG
  lane A 2000 Epic $TAG "" $S $S+ttc:0.7:0.2 $L $L+ttc:0.7:0.2 &
  PA=$!
  lane B 2100 Low $TAG "--no-render" expert constvel+ttc:0.7:0.2 autopilot idm autopilot_stock &
  PB=$!
  while kill -0 $PA 2>/dev/null || kill -0 $PB 2>/dev/null; do
    echo "  [$(($(wc -l < $CSV) - 1))/1026] episodes" >> $LOG
    sleep 30
  done
  echo "  [$(($(wc -l < $CSV) - 1))/1026] episodes" >> $LOG
  echo "$(date) STEP run$N DONE" >> $LOG
  ( $PY -W ignore il/metrics_v2.py $CL_TRAILS --csv $CSV > $IL/tables_$TAG.log 2>&1
    $PY -W ignore il/dist_v2.py $CL_TRAILS >> $IL/tables_$TAG.log 2>&1
    echo "$(date) run$N tables written" >> $LOG ) &          # CPU only: runs beside the next run
done
wait
echo "$(date) STEP compare attempt 1" >> $LOG
$PY -W ignore il/compare_runs.py $TAGS > $IL/compare_10runs.log 2>&1
echo "$(date) STEP compare DONE" >> $LOG
echo "EVAL10 FINISHED $(date)" >> $LOG
