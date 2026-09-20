# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Train a brain on the reach-target task with evolution strategies.

    python experiments/train.py --headless                          # baseline MLP
    python experiments/train.py --config configs/<other>.yaml       # other brain / env
    python experiments/train.py                                     # + video of result

Outputs go to experiments/results/<name>_<timestamp>/: config.json,
train_episodes.csv (every training episode), eval_episodes.csv (held-out seeds every
`eval_every` generations), generations.csv, theta_best.npy / theta_last.npy,
summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import RESULTS, load_config  # noqa: I001  (sets sys.path)

from brain._provenance import AUTHOR_NAME, PROJECT_DATE, PROJECT_ID
from brain.model import make_brain
from brain.trainer import ESTrainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "reach_target.yaml"))
    ap.add_argument("--headless", action="store_true", help="no rendering at all (otherwise: video of the trained brain at the end)")
    ap.add_argument("--generations", type=int, help="override the budget derived from `episodes`")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", help="output dir (default: experiments/results/<name>_<timestamp>)")
    ap.add_argument("--init", help="initial parameters (.npy), e.g. to continue a run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    es = cfg["es"]
    if args.generations is not None:
        es.generations = args.generations
    if args.workers is not None:
        es.workers = args.workers
    out = Path(args.out) if args.out else RESULTS / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"

    brain = make_brain(cfg["brain"], seed=cfg["env_seed"])
    if args.init:
        import numpy as np

        brain.set_params(np.load(args.init))
    print(f"brain {cfg['brain']} ({brain.n_params} params) | {es.generations} generations x "
          f"{2 * es.pairs * es.episodes_per_candidate} episodes | {es.workers} workers | out {out}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps({
        "project_id": PROJECT_ID, "author": AUTHOR_NAME, "project_date": PROJECT_DATE,
        "run_started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": str(args.config)}, indent=2))
    summary = ESTrainer(brain, cfg["brain"], cfg["env"], es, out).train()
    for name in ("config.json", "summary.json"):  # metadata only, added after the trainer wrote them
        f = out / name
        f.write_text(json.dumps(json.loads(f.read_text()) | {"project_id": PROJECT_ID}, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.headless:
        import subprocess
        import sys

        subprocess.run([sys.executable, str(Path(__file__).parent / "evaluate.py"), "--config", args.config,
                        "--run", str(out), "--episodes", "11", "--visual", "--videos", "2"], check=False)


if __name__ == "__main__":
    main()
