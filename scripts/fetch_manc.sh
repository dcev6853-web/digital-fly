#!/usr/bin/env bash
# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
# Fetch the MANC v1.0 traced-neuron adjacency tables and neuron properties from the
# official public Janelia FlyEM bucket (no token needed). Data: CC-BY 4.0, Janelia FlyEM.
# We do not redistribute these files; data/ is gitignored.
set -euo pipefail
dest="$(dirname "$0")/../data/manc_v1.0"
base=https://storage.googleapis.com/flyem-manc-exports/v1.0
mkdir -p "$dest" && cd "$dest"
for f in manc-traced-adjacencies-v1.0/README \
         manc-traced-adjacencies-v1.0/traced-neurons.csv \
         manc-traced-adjacencies-v1.0/traced-connections.csv \
         manc-v1.0-neuron-properties.feather; do
  [ -f "$(basename "$f")" ] || curl -fsS -O "$base/$f"
done
cat > SHA256SUMS.expected <<'SUMS'
fdb7226dd60a1d0794c78ef3050057d79b222fc973e0f70ee80270e3be1c2fa3  README
0c4476528906bb0a20e05e1f01e83fc2e5a582761536171ede5463669ca0b891  manc-v1.0-neuron-properties.feather
4553191e8fd0a198e3588e188b649b7cb39b0550e35d6a11d63b0cd7d8145fda  traced-connections.csv
6b6e1dfdba71f0cef9870d8414a5cbdaa2edf25dcf6d56de0467c79debac3a5a  traced-neurons.csv
SUMS
sha256sum -c SHA256SUMS.expected
