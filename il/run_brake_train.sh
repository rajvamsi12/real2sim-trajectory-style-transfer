#!/bin/bash
# Braking retrain (Raj, 2026-11-14 "go ahead with retraining"): rhd_scratch settings plus
#   rhd_brk_w5     slow-down samples weighted x5 in the loss
#   rhd_brk_hd     speed history hidden for half of the training samples
#   rhd_brk_w5hd   both
# then the offline probe (share of the expert's slow-down predicted) on val and test for these runs and rhd_scratch. No CARLA.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/brake_train.log
DATA=/mnt/data/raj_il/samples_rhd:$IL/feats_rhd
COMMON="--data $DATA --epochs 20 --epoch-samples 65536 --modes 1 --lr 1e-4 --wd 0.05 --dropout 0.3 --feat-drop 0.2 --state speedhist --along-w 2 --ckpt-every 4 --cv-check-epoch 0"
train_step() {
  local name=$1; shift
  for attempt in $(seq 1 4); do
    echo "$(date) STEP train_$name attempt $attempt" >> $LOG
    $PY -u il/train.py $COMMON --name $name --resume "$@" >> $LOG 2>&1 && { echo "$(date) STEP train_$name DONE" >> $LOG; return 0; }
    echo "$(date) STEP train_$name failed, resuming" >> $LOG
    sleep 30
  done
  echo "$(date) STEP train_$name GAVE UP" >> $LOG
  return 1
}
train_step rhd_brk_w5 --slow-w 5
train_step rhd_brk_hd --hist-drop 0.5
train_step rhd_brk_w5hd --slow-w 5 --hist-drop 0.5
for split in val test; do
  echo "$(date) STEP probe_$split attempt 1" >> $LOG
  $PY -u il/probe_offline.py --data $DATA --split $split --runs rhd_scratch rhd_brk_w5 rhd_brk_hd rhd_brk_w5hd >> $LOG 2>&1 \
    && echo "$(date) STEP probe_$split DONE" >> $LOG || echo "$(date) STEP probe_$split FAILED" >> $LOG
done
echo "BRAKE TRAIN FINISHED $(date)" >> $LOG
