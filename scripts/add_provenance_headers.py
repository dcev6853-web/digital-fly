# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Prepend the provenance header to authored source files (idempotent, logic-preserving).

    .venv/bin/python scripts/add_provenance_headers.py            # apply
    .venv/bin/python scripts/add_provenance_headers.py --dry-run  # list only

Scope: brain/, environment/, experiments/, configs/, scripts/ (*.py, *.yaml, *.sh).
Never touches simulator/, data/, .venv/, experiments/results/, or the files in SKIP.

Safety: a file is rewritten only if the change is provably comment-only:
  *.py   -> ast.dump() identical before/after
  *.yaml -> yaml.safe_load() identical before/after
  *.sh   -> content after the header identical to the original body, shebang kept first
Writes are atomic (temp file + os.replace) and keep the file mode (e.g. +x).
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain._provenance import AUTHOR_NAME, PROJECT_DATE, PROJECT_ID  # noqa: E402

HEADER_LINES = [
    f"Original concept & implementation: {AUTHOR_NAME}",
    f"Project: Digital Fly / MANC Learning Architecture — started {PROJECT_DATE}",
    f"Project ID: {PROJECT_ID}",
]
MARKER = f"Project ID: {PROJECT_ID}"
SCOPE = ["brain", "environment", "experiments", "configs", "scripts"]
EXCLUDE_DIRS = {"results", "__pycache__", "simulator", "data", ".venv"}
SKIP = {
    # FlyGym (Apache-2.0, NeLy-EPFL) tutorial code, verbatim: not authored here.
    "experiments/m1_upstream_turning_demo.py": "FlyGym tutorial code (Apache-2.0), not original",
    # Transcribes FlyGym's HybridController.step logic (with caching): derived work.
    "environment/fast_controller.py": "derived from FlyGym controller code (Apache-2.0)",
    # Executing right now (bash reads scripts incrementally): never rewrite a running queue.
    "scripts/queue_v2.sh": "running experiment queue (already carries the header)",
}


def header_for(path: Path) -> str:
    return "".join(f"# {line}\n" for line in HEADER_LINES)


def with_header(path: Path, text: str) -> str:
    head = header_for(path)
    if path.suffix == ".sh" and text.startswith("#!"):
        first, _, rest = text.partition("\n")
        return f"{first}\n{head}{rest}"
    return head + text


def unchanged(path: Path, old: str, new: str) -> bool:
    if path.suffix == ".py":
        return ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))
    if path.suffix == ".yaml":
        return yaml.safe_load(old) == yaml.safe_load(new)
    if path.suffix == ".sh":
        first, _, rest = old.partition("\n")
        return new == f"{first}\n{header_for(path)}{rest}" if old.startswith("#!") else new == header_for(path) + old
    return False


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.provenance.tmp")
    tmp.write_text(text)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


def candidates():
    for top in SCOPE:
        for p in sorted((ROOT / top).rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_file() and p.suffix in {".py", ".yaml", ".sh"} and not (set(rel.parts) & EXCLUDE_DIRS):
                yield p, rel.as_posix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    modified, skipped = [], []
    for path, rel in candidates():
        if rel in SKIP:
            skipped.append((rel, SKIP[rel]))
            continue
        old = path.read_text()
        if MARKER in old:
            skipped.append((rel, "already has header"))
            continue
        new = with_header(path, old)
        if not unchanged(path, old, new):
            skipped.append((rel, "REFUSED: header would change parsed content"))
            continue
        if not args.dry_run:
            atomic_write(path, new)
        modified.append(rel)
    print(("would modify" if args.dry_run else "modified") + f" ({len(modified)}):")
    for m in modified:
        print("  ", m)
    print(f"skipped ({len(skipped)}):")
    for s, why in skipped:
        print(f"   {s}  -- {why}")


if __name__ == "__main__":
    main()
