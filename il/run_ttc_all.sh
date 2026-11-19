#!/bin/bash
# TTC on every vehicle, model on the ego only (Raj's professor, 2026-11-13). il/closed_loop_multi.py (default mode):
#   smoke   2 test scenarios, 4 drivers (stops the chain if it fails)
#   tune    50 VAL scenarios, the model ego + TTC on all with 4 settings (brake if conflict sooner than B s / gap < G m);
#           pick = lowest (any-vehicle collision rate + share of behaviours not kept)  -> $IL/ttc_all_pick.txt
#   test    all 114 test scenarios: expert / const-vel / model ego, each without and with the picked TTC on all vehicles,
#           plus the model with the default setting (1.5 s / 0.4 m)
cd ~/Desktop/carlascripts
PY=~/miniconda3/envs/carla_env/bin/python
IL=Scenariosssss/IL
LOG=$IL/rhd_ttc_all.log
PORT=2000
M=rhd_scratch@cl
pids() { ps -eo pid,args | grep "[C]arlaUE4" | grep "rpc-port=$PORT" | awk "{print \$1}"; }
stop_carla() {
  for p in $(pids); do kill $p 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(pids)" ] && break; sleep 1; done
  for p in $(pids); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 1; done
}
start_carla() {
  stop_carla
  ~/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=$PORT > /tmp/carla_ttc_all.log 2>&1 &
  for i in $(seq 1 90); do sleep 2; (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; done
  sleep 15
}
carla_step() {  # name, tries, command...: fresh CARLA on failure (the csv resumes)
  local name=$1 tries=$2; shift 2
  for attempt in $(seq 1 $tries); do
    echo "$(date) STEP $name attempt $attempt" >> $LOG
    "$@" >> $LOG 2>&1 && { echo "$(date) STEP $name DONE" >> $LOG; return 0; }
    echo "$(date) STEP $name failed, restarting CARLA" >> $LOG
    start_carla
  done
  echo "$(date) STEP $name GAVE UP" >> $LOG
  stop_carla
  exit 1
}
start_carla
carla_step smoke 2 $PY -u il/closed_loop_multi.py --port $PORT --limit 2 --tag ttcall_smoke --drivers expert expert+ttc $M $M+ttc
carla_step tune 8 $PY -u il/closed_loop_multi.py --port $PORT --split val --limit 50 --tag ttcall_tune \
  --drivers $M+ttc $M+ttc:1.0:0.4 $M+ttc:1.0:0.2 $M+ttc:0.7:0.2
echo "$(date) STEP pick" >> $LOG
$PY - >> $LOG 2>&1 <<'EOF'
import pandas as pd
d = pd.read_csv("Scenariosssss/IL/closed_loop_ttcall_tune.csv", dtype={"scenario": str})
d = d[d.result != "spawn failed"]
chk = d[d.behaviour_checked.astype(str) == "True"]
g = d.groupby("driver").agg(any_collision=("result", lambda r: (r == "collision").mean()), progress=("mean_progress", "mean"))
g["behaviour_kept"] = chk.groupby("driver").behaviour_kept.apply(lambda x: (x.astype(str) == "True").mean())
g["score"] = g.any_collision + (1 - g.behaviour_kept)
g = g.sort_values(["score", "progress"], ascending=[True, False])
print(g.round(3).to_string())
suffix = g.index[0].split("+ttc", 1)[1]
open("Scenariosssss/IL/ttc_all_pick.txt", "w").write(suffix)
print("picked +ttc" + suffix)
EOF
echo "$(date) STEP pick DONE" >> $LOG
P="+ttc$(cat $IL/ttc_all_pick.txt)"
D="expert expert$P constvel constvel$P $M $M$P"
[ "$P" != "+ttc" ] && D="$D $M+ttc"
carla_step test 8 $PY -u il/closed_loop_multi.py --port $PORT --tag ttcall_test --drivers $D
stop_carla
echo "RHD TTC ALL FINISHED $(date)" >> $LOG
