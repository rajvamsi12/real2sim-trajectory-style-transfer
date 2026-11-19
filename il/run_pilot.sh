#!/bin/bash
# Step 6 pilot: 50 base scenarios through il/record_samples.py on a GPU CARLA at port 2010. Resumable; retried up to
# 5 times if CARLA or the recorder dies. Only touches the CARLA server on port 2010.
cd ~/Desktop/carlascripts
OUT=Scenariosssss/IL/samples_pilot
LOG=Scenariosssss/IL/pilot.log
kill_2010() { for p in $(ps -eo pid,args | grep "[C]arlaUE4-Linux-Shipping" | grep "rpc-port=2010" | awk "{print \$1}"); do kill $p; done; }
for attempt in 1 2 3 4 5; do
  kill_2010; sleep 5
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2010 > /tmp/carla_pilot.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/2010) 2>/dev/null && break; done
  sleep 15
  echo "$(date) attempt $attempt" >> $LOG
  nice -n 10 ~/miniconda3/envs/carla_env/bin/python -u il/record_samples.py --port 2010 --limit 50 --out $OUT >> $LOG 2>&1 && break
done
kill_2010
echo "PILOT FINISHED $(date)" >> $LOG
