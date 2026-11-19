#!/bin/bash
# Check (2026-11-15): do the drivers without sensors give the same results with rendering off, on a second CARLA (port 2100)?
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/norender_check.log
PORT=2100
export CL_TRAILS=$IL/trails_norender_check
export CL_TM_SEED=1
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
}
stop_carla
~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=$PORT > /tmp/carla_norender.log 2>&1 &
for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
sleep 15
echo "$(date) STEP check attempt 1" >> $LOG
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $LOG
$PY -u il/closed_loop_multi.py --port $PORT --split test --tag norender_check --limit 25 --no-render \
  --drivers expert constvel+ttc:0.7:0.2 autopilot idm autopilot_stock >> $LOG 2>&1 && echo "$(date) STEP check DONE" >> $LOG || echo "$(date) STEP check FAILED" >> $LOG
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $LOG
stop_carla
echo "NORENDER CHECK FINISHED $(date)" >> $LOG
