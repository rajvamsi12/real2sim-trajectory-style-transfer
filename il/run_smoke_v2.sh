#!/bin/bash
# Smoke test (Raj, 2026-11-15): per-tick logging, behaviour check v2 columns, metrics_v2, dist_v2 and the autopilot driver
# on 3 test scenarios (1 Overtaking Left, 1 Following, 1 Overtaking Right); no model driver.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/smoke_v2.log
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
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_smoke_v2.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
start_carla
echo "$(date) STEP smoke attempt 1" >> $LOG
$PY -u il/closed_loop_multi.py --port $PORT --split test --tag smoke_v2 --ids 00060 00381 00307 \
  --drivers expert constvel+ttc:0.7:0.2 autopilot+ttc:0.7:0.2 >> $LOG 2>&1 && echo "$(date) STEP smoke DONE" >> $LOG || echo "$(date) STEP smoke FAILED" >> $LOG
stop_carla
$PY il/metrics_v2.py $CL_TRAILS --csv $IL/closed_loop_smoke_v2.csv >> $LOG 2>&1
$PY il/dist_v2.py $CL_TRAILS >> $LOG 2>&1
echo "SMOKE V2 FINISHED $(date)" >> $LOG
