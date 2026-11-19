#!/bin/bash
# RHD training chain (2026-11-12; ML_DECISIONS 2026-11-10 approved changes). Waits for the feature cache, then
#   rhd_ft       fine-tune of the LHD 1-mode model (runs/lhd_s_K1 = IL_lhd_archive/runs/s_K1)
#   rhd_scratch  same settings from scratch
# both: 1 mode, setting C (lr 1e-4, wd 0.05, dropout 0.3, feature dropout 0.2), 20 epochs x 65,536 samples, speed-history
# state, along-track loss x2, a checkpoint every 4 epochs. Then each run's checkpoints drive 50 VAL scenarios in CARLA
# (il/select_ckpt_closed_loop.py) and the picks are tested closed loop on all test scenarios next to expert and const-vel.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/rhd_train.log
PORT=2000
DATA=/mnt/data/raj_il/samples_rhd:$IL/feats_rhd
COMMON="--data $DATA --epochs 20 --epoch-samples 65536 --modes 1 --lr 1e-4 --wd 0.05 --dropout 0.3 --feat-drop 0.2 --state speedhist --along-w 2 --ckpt-every 4 --cv-check-epoch 3"
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_train.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
train_step() {  # name, extra args: resumes from last.pt after a crash; exit 3 (const-vel check failed) is not retried
  local name=$1; shift
  for attempt in $(seq 1 6); do
    echo "$(date) STEP train_$name attempt $attempt" >> $LOG
    $PY -u il/train.py $COMMON --name $name --resume "$@" >> $LOG 2>&1
    rc=$?
    [ $rc -eq 0 ] && { echo "$(date) STEP train_$name DONE" >> $LOG; return 0; }
    [ $rc -eq 3 ] && { echo "$(date) STEP train_$name FAILED (const-vel check)" >> $LOG; return 1; }
    echo "$(date) STEP train_$name failed (exit $rc), resuming" >> $LOG
    sleep 30
  done
  echo "$(date) STEP train_$name GAVE UP" >> $LOG
  return 1
}
carla_step() {  # name, command...: fresh CARLA, retry on failure (the closed-loop csv resumes)
  local name=$1; shift
  for attempt in $(seq 1 8); do
    (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || start_carla
    echo "$(date) STEP $name attempt $attempt" >> $LOG
    "$@" >> $LOG 2>&1 && { echo "$(date) STEP $name DONE" >> $LOG; return 0; }
    echo "$(date) STEP $name failed, restarting CARLA" >> $LOG
    start_carla
  done
  echo "$(date) STEP $name GAVE UP" >> $LOG
  return 1
}
echo "$(date) STEP wait_feats" >> $LOG
until grep -q "RHD FEATS FINISHED" $IL/rhd_feats.log; do sleep 60; done
echo "$(date) STEP wait_feats DONE" >> $LOG
OK=""
train_step rhd_ft --init-from lhd_s_K1 && OK="$OK rhd_ft"
train_step rhd_scratch && OK="$OK rhd_scratch"
[ -z "$OK" ] && { echo "$(date) no run passed, stopping" >> $LOG; exit 1; }
start_carla
DRIVERS="expert constvel"
for r in $OK; do
  carla_step select_$r $PY -u il/select_ckpt_closed_loop.py --run $r --limit 50 --port $PORT && DRIVERS="$DRIVERS $r@cl"
done
carla_step cl_test $PY -u il/closed_loop.py --port $PORT --tag rhd_test --drivers $DRIVERS lhd_s_K1
stop_carla
echo "RHD TRAIN FINISHED $(date)" >> $LOG
