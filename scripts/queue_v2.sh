#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# Replacement queue after the `manc` readout-saturation bug: MANC-v2 + both v2 controls,
# then the reference policies. All outputs go to new directories (nothing is overwritten).
set -uo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python
export MUJOCO_GL=egl
R=experiments/results
log() { echo "[$(date +%H:%M:%S)] $*"; }

for name in manc_v2 manc_v2_shuffled manc_v2_shuffled_side; do
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
