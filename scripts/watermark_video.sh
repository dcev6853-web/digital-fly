#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# Burn a small semi-transparent authorship watermark (bottom-right) into a video.
# Writes a NEW file; refuses to overwrite the input or an existing output.
#
#   scripts/watermark_video.sh INPUT.mp4 OUTPUT.mp4 ["text"]
#
# Uses only what is already installed: the ffmpeg binary bundled with imageio-ffmpeg
# (whose build has no drawtext filter, so the text is rendered to a transparent PNG with
# Pillow using matplotlib's bundled DejaVuSans, then burned in with ffmpeg's overlay).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
py="$root/.venv/bin/python"
in="${1:?usage: watermark_video.sh INPUT.mp4 OUTPUT.mp4 [text]}"
out="${2:?usage: watermark_video.sh INPUT.mp4 OUTPUT.mp4 [text]}"
text="${3:-Henry Cao — 2026-09-19}"

[ -f "$in" ] || { echo "input not found: $in" >&2; exit 1; }
[ "$(realpath -m "$in")" != "$(realpath -m "$out")" ] || { echo "refusing to overwrite the input" >&2; exit 1; }
[ ! -e "$out" ] || { echo "output exists, refusing to overwrite: $out" >&2; exit 1; }
mkdir -p "$(dirname "$out")"

ffmpeg="$("$py" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
# `ffmpeg -i` without an output always exits 1 (it only prints stream info), so ignore its status.
width="$({ "$ffmpeg" -hide_banner -i "$in" 2>&1 || true; } | grep -m1 -oE 'Video: .*' | grep -oE ' [0-9]{2,5}x[0-9]{2,5}' | head -1 | cut -dx -f1 | tr -d ' ')"
[[ "$width" =~ ^[0-9]+$ ]] || { echo "could not read video width from $in" >&2; exit 1; }
png="$(mktemp --suffix .png)"
trap 'rm -f "$png"' EXIT

"$py" - "$text" "$png" "$width" <<'PY'
import sys
from pathlib import Path
import matplotlib
from PIL import Image, ImageDraw, ImageFont
text, out, width = sys.argv[1], sys.argv[2], int(sys.argv[3])
font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), max(10, width // 30))
x0, y0, x1, y1 = font.getbbox(text)
pad = 3
img = Image.new("RGBA", (x1 - x0 + 2 * pad + 1, y1 - y0 + 2 * pad + 1), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.text((pad - x0 + 1, pad - y0 + 1), text, font=font, fill=(0, 0, 0, 110))      # soft shadow
d.text((pad - x0, pad - y0), text, font=font, fill=(255, 255, 255, 150))        # semi-transparent text
img.save(out)
PY

"$ffmpeg" -hide_banner -loglevel error -i "$in" -i "$png" \
  -filter_complex "[0:v][1:v]overlay=W-w-6:H-h-6:format=auto,format=yuv420p" \
  -c:v libx264 -crf 18 -an "$out"
echo "wrote $out"
