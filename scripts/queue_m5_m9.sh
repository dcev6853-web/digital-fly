#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# Sequential experiment queue (one run at a time: each run already uses all cores).
set -uo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python
export MUJOCO_GL=egl
R=experiments/results
log() { echo "[$(date +%H:%M:%S)] $*"; }

base_pid=$(cat $R/baseline_mlp.pid)
while kill -0 "$base_pid" 2>/dev/null; do sleep 20; done
log "baseline finished"; scripts/evaluate_run.sh $R/baseline_mlp > $R/baseline_mlp/eval.log 2>&1; log "baseline evaluated"

for name in manc manc_shuffled manc_shuffled_side; do
  log "train $name"
  $py experiments/train.py --headless --config configs/reach_target_$name.yaml --out $R/$name > $R/$name.log 2>&1 \
    || log "TRAIN FAILED $name"
  scripts/evaluate_run.sh $R/$name > $R/$name/eval.log 2>&1 || log "EVAL FAILED $name"
  log "done $name"
done

for b in reflex random; do
  $py experiments/evaluate.py --brain $b --episodes 44 --tag test > /dev/null 2>&1
  $py experiments/evaluate.py --brain $b --episodes 44 --tag g1_obstacles --n-obstacles 2 > /dev/null 2>&1
  $py experiments/evaluate.py --brain $b --episodes 44 --tag g2_far --env "target_distance_range=[12.0, 18.0]" episode_length=5.0 > /dev/null 2>&1
  log "reference $b evaluated"
done
log "QUEUE COMPLETE"
