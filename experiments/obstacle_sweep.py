# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Science-fair step 4: success rate vs obstacle count (0 / 1 / 2) for every brain.

All brains use the same 44 test seeds (20,000,000+). The target position for a seed is
identical at every obstacle count (it is drawn before the obstacles), and obstacle 1 is
the same at N=1 and N=2, so the sets are nested.

Existing official evaluations are reused read-only: N=0 is `eval_test` (for the baseline,
`eval_test_regenerated`), and N=2 is `eval_g1_obstacles` (same seeds, same 2-obstacle
env). Anything missing is evaluated into results/obstacle_sweep/<brain>_n<N>/ and never
overwrites anything. Writes results/obstacle_sweep/{table.md, obstacles.png, summary.json}.

    python experiments/obstacle_sweep.py            # run missing evals, then aggregate
    python experiments/obstacle_sweep.py --no-run   # aggregate only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import RESULTS, ROOT  # noqa: I001

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = RESULTS / "obstacle_sweep"
BRAINS = {  # label -> training run dir, or reference brain type
    "reflex (hand-coded)": ("ref", "reflex"),
    "random": ("ref", "random"),
    "MLP v1": ("run", "baseline_mlp"),
    "MLP v2 (heading)": ("run", "baseline_v2"),
    "MANC v2": ("run", "manc_v2"),
    "MANC v2 shuffled": ("run", "manc_v2_shuffled"),
    "MANC v2 shuffled-side": ("run", "manc_v2_shuffled_side"),
}
COUNTS = (0, 1, 2)
EPISODES = 44


def existing(kind, name, n) -> Path | None:
    base = RESULTS / name if kind == "run" else RESULTS
    official = {0: ["eval_test_regenerated", "eval_test"], 2: ["eval_g1_obstacles"]}.get(n, [])
    prefix = "" if kind == "run" else f"eval_{name}_"
    for tag in official:
        d = base / (tag if kind == "run" else prefix + tag.removeprefix("eval_"))
        if (d / "summary.json").exists() and json.loads((d / "summary.json").read_text())["episodes"] == EPISODES:
            return d
    d = OUT / f"{name}_n{n}"
    return d if (d / "summary.json").exists() else None


def run_eval(kind, name, n) -> None:
    out = OUT / f"{name}_n{n}"
    cmd = [sys.executable, str(ROOT / "experiments" / "evaluate.py"), "--episodes", str(EPISODES),
           "--n-obstacles", str(n), "--out", str(out)]
    cmd += ["--run", str(RESULTS / name)] if kind == "run" else ["--brain", name]
    print("running:", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    table, sources = {}, {}
    for label, (kind, name) in BRAINS.items():
        if kind == "run" and not (RESULTS / name / "theta_best.npy").exists():
            print(f"skip {label}: run {name} not trained yet")
            continue
        for n in COUNTS:
            d = existing(kind, name, n)
            if d is None and not args.no_run:
                run_eval(kind, name, n)
                d = existing(kind, name, n)
            if d is None:
                continue
            s = json.loads((d / "summary.json").read_text())
            table.setdefault(label, {})[n] = s
            sources[f"{label} N={n}"] = str(d.relative_to(RESULTS))

    lines = ["| brain | " + " | ".join(f"N={n} success" for n in COUNTS) + " | " +
             " | ".join(f"N={n} collision ticks" for n in COUNTS[1:]) + " |",
             "|---|" + "---|" * (2 * len(COUNTS) - 1)]
    for label, by_n in table.items():
        succ = [f"{by_n[n]['success_rate']:.2f} ({round(by_n[n]['success_rate'] * EPISODES)}/{EPISODES})" if n in by_n else "–" for n in COUNTS]
        coll = [f"{by_n[n]['mean_collision_ticks']:.1f}" if n in by_n else "–" for n in COUNTS[1:]]
        lines.append(f"| {label} | " + " | ".join(succ) + " | " + " | ".join(coll) + " |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps({"sources": sources, "table": {
        k: {str(n): {m: v[m] for m in ("success_rate", "mean_time_to_target", "mean_collision_ticks", "mean_collision_onsets")}
            for n, v in by_n.items()} for k, by_n in table.items()}}, indent=2))
    print("\n".join(lines))

    labels = list(table)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(labels)), 4.2), constrained_layout=True)
    for i, n in enumerate(COUNTS):
        vals = [table[l].get(n, {}).get("success_rate", np.nan) for l in labels]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=f"{n} obstacle{'s' if n != 1 else ''}")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylabel=f"success rate ({EPISODES} episodes, same seeds)", ylim=(0, 1.05),
           title="Generalisation to obstacles (never seen in training)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "obstacles.png", dpi=120)
    print("wrote", OUT / "table.md", OUT / "obstacles.png")


if __name__ == "__main__":
    main()
