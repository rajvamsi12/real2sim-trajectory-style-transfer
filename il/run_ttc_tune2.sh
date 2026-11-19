#!/bin/bash
# Extra TTC settings (Raj, 2026-11-14): 0.5/0.01, 0.5/0.1, 0.3/0.01 on the same 50 val scenarios as the first tuning
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/ttc_tune2.log
PORT=2000
M=rhd_scratch@cl
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_tune2.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
start_carla
for attempt in $(seq 1 6); do
  echo "$(date) STEP tune2 attempt $attempt" >> $LOG
  $PY -u il/closed_loop_multi.py --port $PORT --split val --limit 50 --tag ttcall_tune \
    --drivers $M+ttc:0.5:0.01 $M+ttc:0.5:0.1 $M+ttc:0.3:0.01 >> $LOG 2>&1 && { echo "$(date) STEP tune2 DONE" >> $LOG; break; }
  echo "$(date) STEP tune2 failed, restarting CARLA" >> $LOG
  start_carla
done
stop_carla
echo "TTC TUNE2 FINISHED $(date)" >> $LOG
