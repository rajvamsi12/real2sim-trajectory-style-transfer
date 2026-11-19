#!/bin/bash
# Showcase folders, model ego + TTC on all (0.7 s / 0.2 m) vs CARLA from drone vs real drone (2026-11-13)
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/showcase_ttc.log
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
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_show.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
echo "$(date) STEP pick" >> $LOG
$PY il/showcase_pick.py >> $LOG 2>&1 || { echo "$(date) STEP pick FAILED" >> $LOG; exit 1; }
echo "$(date) STEP pick DONE" >> $LOG
IDS=$(grep "^IDS" $LOG | tail -1 | cut -d" " -f2-)
start_carla
for attempt in $(seq 1 6); do
  echo "$(date) STEP render attempt $attempt" >> $LOG
  CL_VIDEO=$IL/showcase_ttc_raw $PY -u il/closed_loop_multi.py --port $PORT --any-split --ids $IDS --tag showcase_ttc \
    --drivers expert rhd_scratch@cl+ttc:0.7:0.2 >> $LOG 2>&1 && { echo "$(date) STEP render DONE" >> $LOG; break; }
  echo "$(date) STEP render failed, restarting CARLA" >> $LOG
  start_carla
done
stop_carla
echo "$(date) STEP drone" >> $LOG
$PY -u $IL/make_drone_clips.py $IL/showcase_ttc_drone $IDS >> $LOG 2>&1 || { echo "$(date) STEP drone FAILED" >> $LOG; exit 1; }
echo "$(date) STEP drone DONE" >> $LOG
echo "$(date) STEP organize" >> $LOG
$PY il/showcase_organize.py >> $LOG 2>&1 || { echo "$(date) STEP organize FAILED" >> $LOG; exit 1; }
echo "$(date) STEP organize DONE" >> $LOG
echo "SHOWCASE TTC FINISHED $(date)" >> $LOG
