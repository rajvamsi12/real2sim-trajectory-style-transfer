#!/bin/bash
# Step 9 (IL_STATUS.md): waits for step 8 (il/run_step8.sh) to finish, then trains the full model (30 epochs) and the
# no-route ablation, one after the other, with the automatic const-vel check (train.py exit code 3 = check failed ->
# stop everything, the setup needs a look). Any other crash is resumed from last.pt (up to 5 times per run).
# If step 8 stopped without finishing, nothing is trained. Settings can be overridden by env vars (used for the pilot dry run).
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
STEP8_LOG=${STEP8_LOG:-Scenariosssss/IL/step8.log}
STEP8_PROC=${STEP8_PROC:-"^bash il/[r]un_step8.sh"}   # the runner itself, not a wrapper whose command line mentions it
DATA=${DATA:-/mnt/data/raj_il/samples:Scenariosssss/IL/feats_full}
EPOCHS=${EPOCHS:-30}
PREFIX=${PREFIX:-full}
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-4}
CV_EPOCH=${CV_EPOCH:-3}
LOG=${LOG:-Scenariosssss/IL/step9.log}

echo "$(date) waiting for step 8 ($STEP8_LOG)" >> $LOG
while ps -eo args | grep -q "$STEP8_PROC"; do sleep 300; done
if ! grep -q "STEP 8 FINISHED" $STEP8_LOG; then
  echo "$(date) STOPPED: step 8 is not running and did not finish (see $STEP8_LOG); nothing trained" >> $LOG
  exit 1
fi
if [ "$DEVICE" = cuda ]; then
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 1000 ]; do
    echo "$(date) GPU busy, waiting" >> $LOG; sleep 300
  done
fi

train() {  # $1 = run name, rest = extra train.py args
  name=$1; shift
  for attempt in 1 2 3 4 5 6; do
    resume=""; [ $attempt -gt 1 ] && resume="--resume"
    echo "$(date) $name: start attempt $attempt $resume" >> $LOG
    $PY -u il/train.py --data $DATA --epochs $EPOCHS --name $name --device $DEVICE --workers $WORKERS \
        --cv-check-epoch $CV_EPOCH $resume "$@" >> $LOG 2>&1
    rc=$?
    [ $rc -eq 0 ] && { echo "$(date) $name: FINISHED" >> $LOG; return 0; }
    [ $rc -eq 3 ] && { echo "$(date) $name: CONST-VEL CHECK FAILED, stopping step 9" >> $LOG; return 3; }
    echo "$(date) $name: exited with code $rc, resuming" >> $LOG
    sleep 30
  done
  echo "$(date) $name: gave up after 6 attempts" >> $LOG
  return 1
}

train $PREFIX || exit $?
train ${PREFIX}_noroute --no-route || exit $?
echo "STEP 9 FINISHED $(date)" >> $LOG
