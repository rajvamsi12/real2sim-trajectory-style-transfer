#!/bin/bash
# Part 4b (resume after the speed-history gate was relaxed) of the RHD rebuild (2026-11-10, Raj: "apply the 2 augmentation tweaks + record original also"):
#   augment2   augmentation with speed copies v2 (v1 kept in Scenariosssss/IL/aug_v1)
#   aug_reuse  keep the CARLA-check rows of copies identical to v1, so only new / changed copies are checked
#   aug_check2 CARLA check of those (headless, port 2010; resumes from replay_aug_cureE.csv)
#   rec_test   short test recording (3 scenarios per type) + gate on the speed-history columns
#   record     full recording: base scenarios + scene noise (all vehicles) + ego offset / speed (ego only)
#              -> /mnt/data/raj_il/samples_rhd (the LHD recording in samples/ is left alone); resumes from timing.csv
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/rhd_part4.log
OUT=/mnt/data/raj_il/samples_rhd
TEST=/mnt/data/raj_il/samples_rhd_test
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$1" | awk "{print \$1}"; }
stop_carla() {  # $1 = port
  for p in $(pids $1); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids $1)" ] && break; sleep 1; done
  for p in $(pids $1); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$1) 2>/dev/null || break; sleep 1; done
}
start_carla() {  # $1 = port, $2 = headless | render
  stop_carla $1
  if [ "$2" = headless ]; then
    ~/CARLA_0.9.15/CarlaUE4.sh -nullrhi -nosound -carla-rpc-port=$1 > /tmp/carla_$1.log 2>&1 &
  else
    ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$1 > /tmp/carla_$1.log 2>&1 &
  fi
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$1) 2>/dev/null && break; done
  sleep 15
}
plain_step() {
  local name=$1; shift
  echo "$(date) STEP $name" >> $LOG
  "$@" >> $LOG 2>&1 || { echo "$(date) STEP $name FAILED" >> $LOG; exit 1; }
  echo "$(date) STEP $name DONE" >> $LOG
}
retry_step() {  # port, mode, name, tries, command...: restart CARLA and retry when the command fails
  local port=$1 mode=$2 name=$3 tries=$4; shift 4
  for attempt in $(seq 1 $tries); do
    [ $attempt -gt 1 ] && start_carla $port $mode
    (echo > /dev/tcp/127.0.0.1/$port) 2>/dev/null || start_carla $port $mode
    echo "$(date) STEP $name attempt $attempt ($mode CARLA, port $port)" >> $LOG
    "$@" >> $LOG 2>&1 && { echo "$(date) STEP $name DONE" >> $LOG; return 0; }
    echo "$(date) STEP $name failed, restarting CARLA" >> $LOG
    sleep 10
  done
  echo "$(date) STEP $name GAVE UP" >> $LOG
  exit 1
}




plain_step rec_test_check $PY -u il/rec_test_check.py $TEST
mkdir -p $OUT
retry_step 2000 render record 50 $PY -u il/record_samples.py --full --port 2000 --out $OUT --require-mount /mnt/data
stop_carla 2000
echo "RHD PART 4 FINISHED $(date)" >> $LOG
