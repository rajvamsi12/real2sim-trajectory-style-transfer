#!/bin/bash
# Step 8 full feature cache (IL_STATUS.md): frozen SegFormer + PointPillars over every recorded frame in
# /mnt/data/raj_il/samples -> SSD (Scenariosssss/IL/feats_full, ~103 GB est.). cache_features.py skips vehicles already
# done, so a crash just restarts it. Waits while /mnt/data is unmounted; stops if the SSD has < 15 GB free.
cd ~/Desktop/carlascripts
SAMPLES=/mnt/data/raj_il/samples
FEATS=Scenariosssss/IL/feats_full
LOG=Scenariosssss/IL/step8.log
MIN_FREE_GB=15
mkdir -p $FEATS
for attempt in $(seq 1 20); do
  while ! mountpoint -q /mnt/data; do echo "$(date) /mnt/data not mounted, waiting" >> $LOG; sleep 60; done
  free=$(df -BG --output=avail $FEATS | tail -1 | tr -dc 0-9)
  if [ "$free" -lt $MIN_FREE_GB ]; then echo "$(date) STOPPED: only ${free} GB free on the SSD" >> $LOG; exit 1; fi
  echo "$(date) start attempt $attempt" >> $LOG
  ~/miniconda3/envs/carla_env/bin/python -u il/cache_features.py --samples $SAMPLES --feats $FEATS --device cuda >> $LOG 2>&1 && break
  echo "$(date) cache_features exited with an error, restarting" >> $LOG
  sleep 30
done
echo "STEP 8 FINISHED $(date)" >> $LOG
