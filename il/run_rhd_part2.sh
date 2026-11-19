#!/bin/bash
# Part 2 of the RHD rebuild (2026-11-10): the chain (il/run_rhd_rebuild.sh) was stopped before augmentation so the approved
# changes (glitch filter, new augmentation mix) can go in first. Waits for the running build, then: glitch filter
# (il/filter_glitches.py), CARLA check of the 8 s scenarios and the 500-scenario video. Augmentation runs later from its own script.
cd ~/Desktop/carlascripts
source <(sed -n "/^PY=/,/^plain_step() {/p" il/run_rhd_rebuild.sh | sed '$d')
plain_step() {
  local name=$1; shift
  echo "$(date) STEP $name" >> $LOG
  "$@" >> $LOG 2>&1 || { echo "$(date) STEP $name FAILED" >> $LOG; exit 1; }
  echo "$(date) STEP $name DONE" >> $LOG
}
while pgrep -f "il/build_scenarios_8s.py" > /dev/null; do sleep 10; done
grep -q "saved .*scenarios_8s.json" $LOG || { echo "$(date) build did not finish" >> $LOG; exit 1; }
echo "$(date) STEP build DONE" >> $LOG
plain_step glitch_filter $PY -u il/filter_glitches.py
carla_step check $PY -u il/replay_scenarios.py --yaw path --drive velocity --tag cureE
carla_step video500 $PY -u il/render_compare_video.py --showcase 500 --clips-dir $IL/video500_clips
plain_step video500_join $PY -u il/render_compare_video.py --join --clips-dir $IL/video500_clips --out $IL/drone_vs_carla_500.mp4
echo "RHD PART 2 FINISHED $(date)" >> $LOG
