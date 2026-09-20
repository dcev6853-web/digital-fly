#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# Science-fair extension queue (steps 1-4 and 6). Starts only after queue_v2 has exited
# and the step-1 check passes. Every output goes to a new directory.
set -uo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python
export MUJOCO_GL=egl
R=experiments/results
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- step 1: wait for the controlled comparison to finish, then verify it
while kill -0 "$(cat $R/queue_v2.pid)" 2>/dev/null; do sleep 30; done
log "queue_v2 exited"
ok=1
for d in baseline_mlp/eval_test_regenerated manc_v2/eval_test manc_v2_shuffled/eval_test manc_v2_shuffled_side/eval_test; do
  [ -f "$R/$d/summary.json" ] || { log "STEP1 MISSING $d"; ok=0; }
done
$py experiments/report.py baseline_mlp manc_v2 manc_v2_shuffled manc_v2_shuffled_side > $R/report_step1.log 2>&1
[ -f $R/report.md ] && [ -f $R/learning_curves.png ] || { log "STEP1 MISSING report.md/learning_curves.png"; ok=0; }
[ $ok = 1 ] || { log "STEP1 FAILED - stopping"; exit 1; }
log "STEP1 OK"

# ---- step 2: baseline v2 (heading reward), same budget
log "train baseline_v2"
$py experiments/train.py --headless --config configs/reach_target_v2.yaml --out $R/baseline_v2 > $R/baseline_v2.log 2>&1 || log "TRAIN FAILED baseline_v2"
scripts/evaluate_run.sh $R/baseline_v2 > $R/baseline_v2/eval.log 2>&1 || log "EVAL FAILED baseline_v2"
seeds=$(seq 20000000 20000043)
$py experiments/diagnose_looping.py --run $R/baseline_mlp --seeds $seeds --out $R/diagnosis_looping/v1_test44 > $R/diagnosis_looping/v1_test44.log 2>&1 || log "DIAG FAILED v1"
$py experiments/diagnose_looping.py --run $R/baseline_v2 --seeds $seeds --out $R/diagnosis_looping/v2_test44 > $R/diagnosis_looping/v2_test44.log 2>&1 || log "DIAG FAILED v2"
log "done step2"

# ---- step 3: sparsity sweep (min_weight 5 and 20; 10 = manc_v2)
for w in 5 20; do
  log "train manc_v2_w$w"
  $py experiments/train.py --headless --config configs/reach_target_manc_v2_w$w.yaml --out $R/manc_v2_w$w > $R/manc_v2_w$w.log 2>&1 || log "TRAIN FAILED manc_v2_w$w"
  scripts/evaluate_run.sh $R/manc_v2_w$w > $R/manc_v2_w$w/eval.log 2>&1 || log "EVAL FAILED manc_v2_w$w"
done
$py experiments/sparsity_sweep.py > $R/sparsity_sweep/sweep.log 2>&1 || log "SWEEP FAILED"
log "done step3"

# ---- step 4: obstacle-count generalisation
$py experiments/obstacle_sweep.py > $R/obstacle_sweep/sweep.log 2>&1 || log "OBSTACLE SWEEP FAILED"
log "done step4"

# ---- step 6: demo readiness (cold-start renders; blind clips)
JOBS=4 scripts/demo_check.sh > $R/demo_check/run.log 2>&1 || log "DEMO CHECK FAILED"
scripts/blind_demo.sh > $R/blind_demo.log 2>&1 || log "BLIND DEMO FAILED"
log "done step6"
log "QUEUE_SF COMPLETE"
