#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# Science-fair step 6: one unlabeled top-down clip per brain (reflex / baseline / MANC) on
# the SAME episode, for a blind "guess which is smartest" demo. Letters are assigned at
# random; the mapping goes to ANSWER_KEY_PRIVATE.txt (presenter only, not for display).
#   scripts/blind_demo.sh [seed]     (default 20000001: target starts behind the fly)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
R="$root/experiments/results"; OUT="$R/blind_demo"; seed="${1:-20000001}"
[ -e "$OUT/ANSWER_KEY_PRIVATE.txt" ] && { echo "blind demo already exists in $OUT (not overwriting)"; exit 1; }
mkdir -p "$OUT/_renders"
declare -A SRC=( [reflex]="--brain reflex" [baseline]="--run $R/baseline_mlp" [MANC]="--run $R/manc_v2" )
letters=(A B C); order=($(printf "%s\n" reflex baseline MANC | shuf))
{ echo "BLIND DEMO ANSWER KEY — presenter only. Episode seed $seed (same target for all clips)."; } > "$OUT/ANSWER_KEY_PRIVATE.txt"
for i in 0 1 2; do
  b="${order[$i]}"; L="${letters[$i]}"; out="$OUT/_renders/$b"
  t0=$(date +%s)
  env -i HOME="$HOME" PATH=/usr/bin:/bin bash -c \
    "cd '$root' && .venv/bin/python experiments/evaluate.py ${SRC[$b]} --episodes 1 --workers 1 --seed-offset $seed --visual --videos 1 --out '$out'" > "$out.log" 2>&1
  t1=$(date +%s)
  vid=$(find "$out/videos" -name topdown.mp4 | head -1)
  res=$(python3 -c "import json;s=json.load(open('$out/summary.json'));print(f\"success={bool(s['success_rate'])} time_to_target={s['mean_time_to_target']}\")")
  cp "$vid" "$OUT/clip_$L.mp4"
  echo "clip_$L.mp4 = $b   ($res; render wall ${t1}-${t0}=$((t1 - t0)) s)" >> "$OUT/ANSWER_KEY_PRIVATE.txt"
  echo "clip_$L.mp4 rendered in $((t1 - t0)) s"
done
chmod 600 "$OUT/ANSWER_KEY_PRIVATE.txt"
echo "clips: $OUT/clip_{A,B,C}.mp4   key: $OUT/ANSWER_KEY_PRIVATE.txt"
