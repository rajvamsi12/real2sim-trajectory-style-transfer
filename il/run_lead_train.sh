#!/bin/bash
# Lead-vehicle input retrain (Raj, 2026-11-15 "go ahead with ground truth first"): rhd_scratch settings plus --lead (il/lead_input.py)
# then the offline probe (share of the expert's slow-down predicted) on val and test for these runs and rhd_scratch. No CARLA.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/lead_train.log
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
train_step rhd_lead --lead
train_step rhd_lead_hd --lead --hist-drop 0.5
train_step rhd_lead_w2 --lead --slow-w 2
for split in val test; do
  echo "$(date) STEP probe_$split attempt 1" >> $LOG
  $PY -u il/probe_offline.py --data $DATA --split $split --runs rhd_scratch rhd_lead rhd_lead_hd rhd_lead_w2 >> $LOG 2>&1 \
    && echo "$(date) STEP probe_$split DONE" >> $LOG || echo "$(date) STEP probe_$split FAILED" >> $LOG
done
echo "LEAD TRAIN FINISHED $(date)" >> $LOG
