#!/bin/bash
# Smoke test of the stock Autopilot driver (2026-11-15): expert, autopilot (recorded path + speed), autopilot_stock, 3 test scenarios.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/smoke_stock.log
PORT=2000
export CL_TRAILS=$IL/trails_smoke_stock
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_smoke_stock.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
start_carla
echo "$(date) STEP smoke attempt 1" >> $LOG
$PY -u il/closed_loop_multi.py --port $PORT --split test --tag smoke_stock --ids 00060 00381 00307 \
  --drivers expert autopilot autopilot_stock >> $LOG 2>&1 && echo "$(date) STEP smoke DONE" >> $LOG || echo "$(date) STEP smoke FAILED" >> $LOG
stop_carla
$PY il/metrics_v2.py $CL_TRAILS --csv $IL/closed_loop_smoke_stock.csv >> $LOG 2>&1
$PY il/dist_v2.py $CL_TRAILS >> $LOG 2>&1
echo "SMOKE STOCK FINISHED $(date)" >> $LOG
