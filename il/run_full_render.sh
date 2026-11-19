#!/bin/bash
# Step 7 full render (ML_DECISIONS.md T4, middle option) -> /mnt/data/raj_il/samples. Restarts CARLA (port 2000) and the
# recorder whenever either dies; the recorder resumes from timing.csv, skips a scenario after 2 crashes (errors.csv)
# and pauses itself if /mnt/data is unmounted or nearly full.
cd ~/Desktop/carlascripts
OUT=/mnt/data/raj_il/samples
LOG=Scenariosssss/IL/full_render.log
PORT=2000
mkdir -p $OUT
carla_pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  # CARLA needs several seconds to exit; starting a new server before the old one is gone leaves the port held by a
  # dying server (the recorder then times out). Wait until it is gone, force-kill after 30 s.
  for p in $(carla_pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(carla_pids)" ] && break; sleep 1; done
  for p in $(carla_pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
for attempt in $(seq 1 50); do
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_full.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
  echo "$(date) start attempt $attempt" >> $LOG
  ~/miniconda3/envs/carla_env/bin/python -u il/record_samples.py --full --port $PORT --out $OUT --require-mount /mnt/data >> $LOG 2>&1 && break
  echo "$(date) recorder exited with an error, restarting" >> $LOG
  sleep 10
done
stop_carla
echo "FULL RENDER FINISHED $(date)" >> $LOG
