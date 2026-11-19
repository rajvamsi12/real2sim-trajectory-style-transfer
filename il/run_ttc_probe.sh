#!/bin/bash
# TTC layer + probe run (Raj, 2026-11-13), all 114 test scenarios, CARLA port 2000:
#   ttc_test  all drivers again with the hit direction recorded: expert, constvel, constvel+ttc, rhd_scratch@cl,
#             rhd_scratch@cl+ttc (tag rhd_ttc; the 09:01 test has no hit direction)
#   probe     CL_PROBE=1: the expert drives, rhd_scratch@cl only predicts (PROBE lines -> rhd_probe.log)
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/rhd_ttc.log
PORT=2000
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_ttc.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
carla_step() {  # name, command...: retry with a fresh CARLA (the csv resumes)
  local name=$1; shift
  for attempt in $(seq 1 8); do
    (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || start_carla
    echo "$(date) STEP $name attempt $attempt" >> $LOG
    "$@" && { echo "$(date) STEP $name DONE" >> $LOG; return 0; }
    echo "$(date) STEP $name failed, restarting CARLA" >> $LOG
    start_carla
  done
  echo "$(date) STEP $name GAVE UP" >> $LOG
  exit 1
}
ttc() { $PY -u il/closed_loop.py --port $PORT --tag rhd_ttc --drivers expert constvel constvel+ttc rhd_scratch@cl rhd_scratch@cl+ttc >> $LOG 2>&1; }
probe() { CL_PROBE=1 $PY -u il/closed_loop.py --port $PORT --tag rhd_probe --drivers rhd_scratch@cl >> $LOG 2>> $IL/rhd_probe.log; }
carla_step ttc_test ttc
carla_step probe probe
stop_carla
echo "RHD TTC FINISHED $(date)" >> $LOG
