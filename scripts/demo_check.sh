#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# Science-fair step 6: cold-start check of the live-demo command for every brain at
# N = 0, 1, 2 obstacles. Each render runs in a clean environment (env -i: no MUJOCO_GL,
# no activated venv, minimal PATH), exactly like a fresh terminal on demo day.
# Output: experiments/results/demo_check/<brain>_n<N>/ + timings.tsv. Never overwrites.
#   scripts/demo_check.sh            (JOBS=4 renders in parallel by default)
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
R="$root/experiments/results"; OUT="$R/demo_check"; mkdir -p "$OUT"
brains="${BRAINS:-reflex baseline_mlp baseline_v2 manc_v2 manc_v2_shuffled manc_v2_shuffled_side}"
jobs="${JOBS:-4}"

one() {
  b="$1"; n="$2"; out="$OUT/${b}_n$n"
  [ -e "$out/summary.json" ] && { echo "skip $b n=$n (exists)"; return 0; }
  if [ "$b" = reflex ]; then src="--brain reflex"; else
    [ -f "$R/$b/theta_best.npy" ] || { echo "skip $b n=$n (not trained)"; return 0; }; src="--run $R/$b"; fi
  t0=$(date +%s)
  env -i HOME="$HOME" PATH=/usr/bin:/bin bash -c \
    "cd '$root' && .venv/bin/python experiments/evaluate.py $src --episodes 1 --workers 1 --n-obstacles $n --visual --videos 1 --out '$out'" \
    > "$out.log" 2>&1
  rc=$?; t1=$(date +%s)
  vids=$(find "$out/videos" -name '*.mp4' 2>/dev/null | wc -l)
  printf '%s\t%s\t%s\t%s\t%s\n' "$b" "$n" "$rc" "$((t1 - t0))" "$vids" | tee -a "$OUT/timings.tsv"
}
export -f one; export R OUT root
[ -f "$OUT/timings.tsv" ] || printf 'brain\tobstacles\texit_code\twall_s\tvideos\n' > "$OUT/timings.tsv"
for b in $brains; do for n in ${COUNTS:-0 1 2}; do echo "$b $n"; done; done | xargs -P "$jobs" -n 2 bash -c 'one "$0" "$1"'
