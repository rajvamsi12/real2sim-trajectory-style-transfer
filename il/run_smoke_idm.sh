#!/bin/bash
# Smoke test of the IDM driver (Raj, 2026-11-15 "lets start with idm"): the 3 smoke scenarios, with and without TTC on all.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/smoke_idm.log
PORT=2000
export CL_TRAILS=$IL/trails_smoke_v2
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_smoke_idm.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
start_carla
echo "$(date) STEP smoke_idm attempt 1" >> $LOG
$PY -u il/closed_loop_multi.py --port $PORT --split test --tag smoke_v2 --ids 00060 00381 00307 \
  --drivers idm idm+ttc:0.7:0.2 >> $LOG 2>&1 && echo "$(date) STEP smoke_idm DONE" >> $LOG || echo "$(date) STEP smoke_idm FAILED" >> $LOG
stop_carla
$PY -W ignore il/metrics_v2.py $CL_TRAILS --csv $IL/closed_loop_smoke_v2.csv >> $LOG 2>&1
$PY -W ignore il/dist_v2.py $CL_TRAILS >> $LOG 2>&1
echo "SMOKE IDM FINISHED $(date)" >> $LOG
