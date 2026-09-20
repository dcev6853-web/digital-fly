#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# One-screen status of the experiment queue and every training run.
cd "$(dirname "$0")/.."
R=experiments/results
for qlog in $(ls $R/queue*.log 2>/dev/null); do
  q=$(basename "$qlog" .log)
  echo "== $q ($(kill -0 "$(cat $R/$q.pid 2>/dev/null)" 2>/dev/null && echo running || echo stopped))"
  tail -6 "$qlog"
done
for log in $(grep -l '\] fit' $R/*.log 2>/dev/null | grep -v '/queue'); do
  [ -f "$log" ] || continue
  echo; echo "== $(basename "$log" .log): $(grep -c '\] fit' "$log") generations done"
  grep EVAL "$log" | tail -4
done
echo; for s in $R/*/eval_*/summary.json $R/eval_*/summary.json; do
  [ -f "$s" ] && python3 -c "import json,sys;d=json.load(open('$s'));print(f\"{'$s'.split('/')[-3]+'/'+'$s'.split('/')[-2]:50s} success {d['success_rate']:.2f}  t2t {d['mean_time_to_target']}  coll {d['mean_collision_ticks']:.1f}  n={d['episodes']}\")"
done 2>/dev/null
exit 0
