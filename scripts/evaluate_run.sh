#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# Test a trained run on the held-out test set and the Milestone 9 generalisation sets.
#   scripts/evaluate_run.sh experiments/results/<run> [episodes]
set -euo pipefail
cd "$(dirname "$0")/.."
run="$1"; n="${2:-44}"
py=.venv/bin/python
export MUJOCO_GL=egl
$py experiments/evaluate.py --run "$run" --episodes "$n" --tag test
$py experiments/evaluate.py --run "$run" --episodes "$n" --tag g1_obstacles --n-obstacles 2
$py experiments/evaluate.py --run "$run" --episodes "$n" --tag g2_far \
    --env "target_distance_range=[12.0, 18.0]" episode_length=5.0
