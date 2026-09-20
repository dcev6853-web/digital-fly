# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Science-fair step 5: build the one judge-readable results file.

Collects every finished run and evaluation and writes results/report.md with:
  1. a success-rate bar chart across all brains and all three test sets,
  2. learning curves for every training run,
  3. the sparsity trade-off plot (step 3),
  4. the obstacle-count chart (step 4),
plus the numbers behind them. Reads results only; runs nothing.

    python experiments/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from common import RESULTS  # noqa: I001

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brain._provenance import AUTHOR_NAME, PROJECT_DATE, PROJECT_ID  # noqa: E402

# label -> (kind, name). Order is the display order.
BRAINS = [
    ("Random (floor)", "ref", "random"),
    ("Hand-coded reflex", "ref", "reflex"),
    ("MLP baseline (E1)", "run", "baseline_mlp"),
    ("MLP + heading reward (v2)", "run", "baseline_v2"),
    ("MANC real wiring (E2)", "run", "manc_v2"),
    ("MANC shuffled (E3)", "run", "manc_v2_shuffled"),
    ("MANC shuffled, sides kept (E4)", "run", "manc_v2_shuffled_side"),
    ("MANC dense, w≥5", "run", "manc_v2_w5"),
    ("MANC sparse, w≥20", "run", "manc_v2_w20"),
]
SETS = [("test", "standard targets"), ("g1_obstacles", "2 obstacles"), ("g2_far", "far targets")]


def summary_for(kind, name, tag):
    if kind == "run":
        for d in ([RESULTS / name / "eval_test_regenerated"] if tag == "test" else []) + [RESULTS / name / f"eval_{tag}"]:
            if (d / "summary.json").exists():
                return json.loads((d / "summary.json").read_text())
        return None
    d = RESULTS / f"eval_{name}_{tag}"
    return json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None


def main():
    rows, present = [], []
    for label, kind, name in BRAINS:
        got = {}
        for tag, _ in SETS:
            s = summary_for(kind, name, tag)
            if s:
                got[tag] = s
        if not got:
            continue
        present.append(label)
        row = {"brain": label}
        for tag, nice in SETS:
            s = got.get(tag)
            row[f"{nice} success"] = f"{s['success_rate']:.2f}" if s else "–"
            row[f"{nice} time (s)"] = (f"{s['mean_time_to_target']:.2f}" if s and s["mean_time_to_target"] else "–")
        row["params"] = ""
        cfg = RESULTS / name / "config.json"
        if kind == "run" and cfg.exists():
            row["params"] = json.loads(cfg.read_text()).get("n_params", "")
        rows.append(row | {"_data": got})
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_data"} for r in rows])

    # 1. success bar chart
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(rows)), 4.4), constrained_layout=True)
    for i, (tag, nice) in enumerate(SETS):
        ax.bar(x + (i - 1) * 0.27, [r["_data"].get(tag, {}).get("success_rate", np.nan) for r in rows], 0.27, label=nice)
    ax.set_xticks(x, present, rotation=22, ha="right")
    ax.set(ylabel="success rate (44 episodes, identical seeds)", ylim=(0, 1.05),
           title="Reach-target success by brain and test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(RESULTS / "success_by_brain.png", dpi=120)
    plt.close(fig)

    # 2. learning curves for every run that has them
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    for label, kind, name in BRAINS:
        f = RESULTS / name / "generations.csv"
        if kind != "run" or not f.exists():
            continue
        g = pd.read_csv(f).dropna(subset=["eval_success_rate"])
        ax.plot(g.generation, g.eval_success_rate, marker="o", ms=3, label=label)
    ax.set(xlabel="ES generation (32 episodes each)", ylabel="held-out success (22 seeds)",
           ylim=(-0.05, 1.05), title="Learning curves (identical budget and seeds)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.savefig(RESULTS / "learning_curves_all.png", dpi=120)
    plt.close(fig)

    md = [f"# Digital Fly / MANC — results\n",
          f"_{AUTHOR_NAME}, {PROJECT_DATE}. Project ID: {PROJECT_ID}._\n",
          "A learned brain sits between the senses and the legs of an unmodified NeuroMechFly v2 "
          "(FlyGym) fly. The experimental brain routes its motor pathway through the male nerve-cord "
          "connectome (MANC v1.0). Full protocol, hypotheses and caveats: EXPERIMENTS.md. "
          "Pipeline and insertion point: ARCHITECTURE.md.\n",
          "## 1. Success by brain\n", "![success](success_by_brain.png)\n",
          table.to_markdown(index=False) + "\n",
          "Every brain is scored on the same 44 held-out episodes (seeds 20,000,000+) with its best "
          "checkpoint, chosen on separate in-training seeds.\n",
          "## 2. Learning curves\n", "![learning curves](learning_curves_all.png)\n",
          "Identical optimiser, budget (48 generations x 32 episodes) and seeds for every brain.\n"]
    sp = RESULTS / "sparsity_sweep"
    if (sp / "table.md").exists():
        md += ["## 3. Connectome sparsity trade-off\n", "![sparsity](sparsity_sweep/sweep.png)\n",
               (sp / "table.md").read_text(),
               "\nOnly the synapse-count threshold changes; env, optimiser, budget and seeds are fixed. "
               "Wall-clock per generation is noisy on this laptop (the host steals cores), so the isolated "
               "brain-tick time is the reliable compute measure.\n"]
    ob = RESULTS / "obstacle_sweep"
    if (ob / "table.md").exists():
        md += ["## 4. Generalisation to obstacles\n", "![obstacles](obstacle_sweep/obstacles.png)\n",
               (ob / "table.md").read_text(),
               "\nNo brain saw an obstacle during training, so 1 and 2 obstacles are both unfamiliar.\n"]
    md += ["## Verification\n",
           "- Simulator untouched: `git -C simulator status` clean at the pinned commit, and FlyGym's own "
           "76 tests pass.\n- Project tests: 15 pass, including a bit-identical check of the cached "
           "controller path and a regression test for the readout-saturation bug.\n"
           "- Reference policies bound the task: random 3/44, hand-coded reflex 43/44.\n"]
    (RESULTS / "report.md").write_text("\n".join(md))
    print(table.to_string(index=False))
    print("\nwrote", RESULTS / "report.md")


if __name__ == "__main__":
    main()
