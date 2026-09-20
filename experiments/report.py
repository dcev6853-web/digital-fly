# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Aggregate training runs and test evaluations into a table and learning curves.

    python experiments/report.py baseline_mlp manc manc_shuffled ...   # run dir names under results/

Writes results/report.md (markdown tables of every eval_*/summary.json found in each
run) and results/learning_curves.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common import RESULTS  # noqa: I001

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLS = ["success_rate", "mean_time_to_target", "mean_steps_to_target", "mean_distance_traveled",
        "mean_collision_ticks", "mean_collision_onsets", "mean_reward", "mean_final_distance", "episodes"]


def fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main(runs: list[str]):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name in runs:
        run = RESULTS / name
        gens = run / "generations.csv"
        if gens.exists():
            g = pd.read_csv(gens)
            tr = g.dropna(subset=["mean_fitness"])
            axes[0].plot(tr.generation, tr.mean_fitness, label=name)
            ev = g.dropna(subset=["eval_success_rate"])
            axes[1].plot(ev.generation, ev.eval_success_rate, marker="o", label=name)
        for summ in sorted(run.glob("eval_*/summary.json")):
            s = json.loads(summ.read_text())
            rows.append({"run": name, "eval": summ.parent.name.removeprefix("eval_")} | {c: s.get(c) for c in COLS})
    for extra in sorted(RESULTS.glob("eval_*/summary.json")):  # reference policies
        s = json.loads(extra.read_text())
        rows.append({"run": extra.parent.name.removeprefix("eval_"), "eval": "reference"} | {c: s.get(c) for c in COLS})

    axes[0].set(xlabel="generation", ylabel="mean training fitness (episode reward)", title="ES training")
    axes[1].set(xlabel="generation", ylabel="success rate (held-out seeds)", title="In-training evaluation", ylim=(-0.05, 1.05))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(RESULTS / "learning_curves.png", dpi=120)

    lines = ["| run | eval | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for r in rows:
        lines.append(f"| {r['run']} | {r['eval']} | " + " | ".join(fmt(r[c]) for c in COLS) + " |")
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
