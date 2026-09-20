# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Science-fair step 3: MANC circuit sparsity (min_weight) vs success vs compute.

Reads (never writes) the training runs manc_v2_w5, manc_v2 (w10) and manc_v2_w20, plus
circuits.json (build time, size, brain tick cost; written once by the sweep prep), and
writes results/sparsity_sweep/{table.md, sweep.png}.

Only min_weight differs between the three runs: same env, ES settings, budget and seeds.

    python experiments/sparsity_sweep.py
"""

from __future__ import annotations

import json

from common import RESULTS  # noqa: I001

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = RESULTS / "sparsity_sweep"
RUNS = {5: "manc_v2_w5", 10: "manc_v2", 20: "manc_v2_w20"}


def main():
    circuits = {c["min_weight"]: c for c in json.loads((OUT / "circuits.json").read_text())}
    rows = []
    for w, run in RUNS.items():
        d = RESULTS / run
        row = dict(circuits[w]) | {"run": run}
        if (d / "generations.csv").exists():
            g = pd.read_csv(d / "generations.csv")
            es_only = g[g.eval_success_rate.isna()]  # generations without an eval pass
            row["median_wall_s_per_gen"] = float(es_only.wall_s.median()) if len(es_only) else None
            row["generations_done"] = int(g.generation.max())
        if (d / "summary.json").exists():
            row["best_heldout_success"] = json.loads((d / "summary.json").read_text())["best"]["eval_success_rate"]
        if (d / "eval_test" / "summary.json").exists():
            row["test_success"] = json.loads((d / "eval_test" / "summary.json").read_text())["success_rate"]
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["min_weight", "nodes", "edges", "density", "params", "build_s", "brain_tick_ms",
            "median_wall_s_per_gen", "best_heldout_success", "test_success", "run"]
    df = df[[c for c in cols if c in df.columns]]
    (OUT / "table.md").write_text(df.to_markdown(index=False, floatfmt=".4g") + "\n")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    if "test_success" in df:
        ax[0].plot(df.edges, df.test_success, "o-", label="test success (44 seeds)")
    if "best_heldout_success" in df:
        ax[0].plot(df.edges, df.best_heldout_success, "s--", label="best in-training held-out (22 seeds)")
    for _, r in df.iterrows():
        ax[0].annotate(f"w≥{r.min_weight}", (r.edges, 0.02), ha="center", fontsize=8)
    ax[0].set(xscale="log", xlabel="circuit edges (log)", ylabel="success rate", ylim=(0, 1.05), title="Sparsity vs success")
    ax[0].legend(fontsize=8)
    ax[1].plot(df.edges, df.brain_tick_ms, "o-", color="C2", label="brain tick (ms, isolated)")
    ax2 = ax[1].twinx()
    if "median_wall_s_per_gen" in df:
        ax2.plot(df.edges, df.median_wall_s_per_gen, "s--", color="C3", label="median wall s / generation")
    ax[1].set(xscale="log", xlabel="circuit edges (log)", ylabel="brain tick (ms)", title="Sparsity vs compute")
    ax2.set_ylabel("wall s per generation (whole ES step)")
    fig.legend(loc="lower right", fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.savefig(OUT / "sweep.png", dpi=120)
    print("wrote", OUT / "table.md", OUT / "sweep.png")


if __name__ == "__main__":
    main()
