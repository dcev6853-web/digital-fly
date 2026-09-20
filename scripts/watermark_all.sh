#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# Write a watermarked copy of every rendered video, alongside the original as <name>_wm.mp4.
# Raw renders are never overwritten or removed; an existing _wm copy is left alone.
#   scripts/watermark_all.sh [root]        (default: experiments/results)
set -uo pipefail
cd "$(dirname "$0")/.."
root="${1:-experiments/results}"
made=0; skipped=0; failed=0
while IFS= read -r -d '' src; do
  out="${src%.mp4}_wm.mp4"
  if [ -e "$out" ]; then skipped=$((skipped + 1)); continue; fi
  if scripts/watermark_video.sh "$src" "$out" </dev/null >/dev/null 2>&1; then
    made=$((made + 1))
  else
    failed=$((failed + 1)); echo "FAILED: $src" >&2
  fi
done < <(find "$root" -name "*.mp4" ! -name "*_wm.mp4" -print0 | sort -z)
echo "watermarked copies written: $made, already present: $skipped, failed: $failed"
