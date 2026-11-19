#!/bin/bash
# Overnight sweep runner (29 Oct): runs the lines of a queue file one after another ("name extra-args"), all with
# setting C as the base. Lines can be appended while it runs; a line "STOP" ends it. Log per queue.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
Q=$1; LOG=${Q%.txt}.log
BASE="--data /mnt/data/raj_il/samples:Scenariosssss/IL/feats_full --epochs 20 --epoch-samples 65536 --device cuda --workers 4
      --cv-check-epoch 0 --dropout 0.3 --feat-drop 0.2 --wd 0.05 --lr 1e-4"
i=1
while true; do
  line=$(sed -n "${i}p" $Q)
  if [ -z "$line" ]; then sleep 60; continue; fi
  [ "$line" = STOP ] && { echo "$(date) QUEUE DONE" >> $LOG; exit 0; }
  set -- $line; name=$1; shift
  echo "$(date) $name: start $*" >> $LOG
  $PY -u il/train.py $BASE --name $name "$@" >> $LOG 2>&1
  echo "$(date) $name: exit $?" >> $LOG
  i=$((i+1))
done
