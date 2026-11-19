#!/bin/bash
# Short regularisation sweep after the full run overfit after epoch 1 (IL_STATUS.md step 9).
# Each run: 20 "epochs" of 65,536 random train samples (~2 full passes), val after each, same cosine schedule length.
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
DATA=/mnt/data/raj_il/samples:Scenariosssss/IL/feats_full
LOG=Scenariosssss/IL/regsweep.log
COMMON="--data $DATA --epochs 20 --epoch-samples 65536 --device cuda --workers 4 --cv-check-epoch 0"
run() { name=$1; shift; echo "$(date) $name: start $*" >> $LOG
        $PY -u il/train.py $COMMON --name $name "$@" >> $LOG 2>&1; echo "$(date) $name: exit $?" >> $LOG; }
run reg_A_base
run reg_B_drop  --dropout 0.3 --feat-drop 0.2 --wd 0.05
run reg_C_droplr --dropout 0.3 --feat-drop 0.2 --wd 0.05 --lr 1e-4
echo "REGSWEEP FINISHED $(date)" >> $LOG
