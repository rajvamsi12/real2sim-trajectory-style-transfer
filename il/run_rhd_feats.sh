#!/bin/bash
# RHD feature cache (2026-11-12): frozen SegFormer + PointPillars over every recorded frame in
# /mnt/data/raj_il/samples_rhd -> SSD (Scenariosssss/IL/feats_rhd, ~60 GB est.); same LiDAR PCA as the LHD model. cache_features.py skips vehicles already
# done, so a crash just restarts it. Waits while /mnt/data is unmounted; stops if the SSD has < 15 GB free.
cd ~/Desktop/carlascripts
SAMPLES=/mnt/data/raj_il/samples_rhd
FEATS=Scenariosssss/IL/feats_rhd
LOG=Scenariosssss/IL/rhd_feats.log
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
echo "RHD FEATS FINISHED $(date)" >> $LOG
