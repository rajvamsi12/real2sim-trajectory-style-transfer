#!/bin/bash
# Step 11 closed-loop run: restarts CARLA (port 2000) and il/closed_loop.py whenever either dies (the script resumes
# from its CSV). Args are passed to closed_loop.py.
cd ~/Desktop/carlascripts
PORT=2000
LOG=Scenariosssss/IL/closed_loop.log
carla_up() { (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; }
for attempt in $(seq 1 20); do
  if ! carla_up; then
    pkill -f "CarlaUE4.*rpc-port=$PORT"; sleep 10
    ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_cl.log 2>&1 &
    for i in $(seq 1 90); do sleep 2; carla_up && break; done
    sleep 15
  fi
  echo "$(date) start attempt $attempt" >> $LOG
  ~/miniconda3/envs/carla_env/bin/python -u il/closed_loop.py --port $PORT "$@" >> $LOG 2>&1 && break
  echo "$(date) closed_loop.py exited with an error, restarting" >> $LOG
  pkill -f "CarlaUE4.*rpc-port=$PORT"; sleep 20
done
echo "CLOSED LOOP FINISHED $(date)" >> $LOG
