# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Per-tick diagnosis of a brain's approach behaviour (read-only: no training).

Replays a saved checkpoint on given seeds (the sim is deterministic, so this reproduces
recorded episodes/videos exactly) and records, every brain tick: position, heading,
egocentric target bearing, distance, commanded (forward, turn), and yaw rate.

    python experiments/diagnose_looping.py --run experiments/results/baseline_mlp \
        --checkpoint experiments/results/baseline_mlp/theta_gen20_snapshot.npy \
        --seeds 20000000 20000001 20000002 --out experiments/results/diagnosis_looping/gen20

Writes ticks.csv, episodes.csv, summary.json and paths.png into --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config  # noqa: I001  (sets sys.path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brain.interface import run_episode  # noqa: E402
from brain.model import make_brain  # noqa: E402
from environment.fly_interface import EnvConfig, FlyInterface  # noqa: E402


def trace_episode(brain, fly, seed):
    rows = []

    def on_tick(obs, action, reward, info):
        xy = fly.thorax_xy()
        rows.append({
            "tick": fly.tick, "x": xy[0], "y": xy[1],
            "heading": float(np.arctan2(obs[11], obs[10])),
            "bearing": float(np.arctan2(obs[2], obs[1])),  # >0: target to the left
            "distance": info["distance"],
            "forward_cmd": float(action[0]), "turn_cmd": float(action[1]),  # turn>0: clockwise
            "yaw_rate": float(obs[9] * fly.config.yaw_rate_norm),
            "reward": reward,
        })

    # Record the pre-first-tick state too.
    ep = run_episode(brain, fly, seed=seed, on_tick=on_tick)
    df = pd.DataFrame(rows)
    df["seed"] = seed
    return ep, df, fly.target_xy.copy()


def analyse(ticks: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    """Numbers that distinguish looping mechanisms."""
    t = ticks
    # Correct steering: target left (bearing>0) -> counter-clockwise (turn<0).
    corr = float(np.corrcoef(t.bearing, t.turn_cmd)[0, 1])
    wrong_way = float(np.mean(np.sign(t.turn_cmd) == np.sign(t.bearing)) )  # turning away
    ahead = t[np.abs(t.bearing) < np.deg2rad(20)]
    # Net heading change per episode (sum of wrapped heading increments) = total rotation.
    rot = []
    for _, g in t.groupby("seed"):
        dh = np.angle(np.exp(1j * np.diff(g.heading.to_numpy())))
        rot.append(float(np.sum(dh)))
    # Command oscillation: fraction of consecutive ticks where turn command flips sign.
    flips = []
    for _, g in t.groupby("seed"):
        s = np.sign(g.turn_cmd.to_numpy())
        flips.append(float(np.mean(s[1:] != s[:-1])) if len(s) > 1 else 0.0)
    # Path length of the thorax after a 5-tick (100 ms) centred moving average, which removes
    # the within-step body sway (step cycle ~83 ms) that inflates the raw per-tick sum.
    smooth_len, backward, init_d = [], [], episodes.set_index("seed").initial_distance
    for s, g in t.groupby("seed"):
        xy = g[["x", "y"]].rolling(5, center=True, min_periods=1).mean().to_numpy()
        smooth_len.append(float(np.hypot(*np.diff(xy, axis=0).T).sum()))
        d = np.diff(g[["x", "y"]].to_numpy(), axis=0)
        h = g.heading.to_numpy()[1:]
        backward.append(float(np.mean(d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h) < 0)))
    eff = np.array(smooth_len) / init_d.loc[sorted(t.seed.unique())].to_numpy()
    succ = episodes.set_index("seed").success.loc[sorted(t.seed.unique())].to_numpy().astype(bool)
    return {
        "episodes": int(len(episodes)),
        "success_rate": float(episodes.success.mean()),
        "path_efficiency_raw_mean": float((episodes.distance_traveled / episodes.initial_distance).mean()),
        "path_efficiency_smoothed_mean": float(eff.mean()),
        "path_efficiency_smoothed_successes_mean": float(eff[succ].mean()) if succ.any() else None,
        "frac_ticks_moving_backward": float(np.mean(backward)),
        "frac_ticks_target_behind": float(np.mean(np.abs(t.bearing) > np.pi / 2)),
        "turn_vs_bearing_corr": corr,
        "frac_ticks_turning_away_from_target": wrong_way,
        "mean_turn_cmd": float(t.turn_cmd.mean()),
        "mean_turn_cmd_when_target_ahead_20deg": float(ahead.turn_cmd.mean()) if len(ahead) else None,
        "mean_abs_bearing_rad": float(np.abs(t.bearing).mean()),
        "net_rotation_per_episode_rad": rot,
        "turn_sign_flip_rate": float(np.mean(flips)),
        "mean_forward_cmd": float(t.forward_cmd.mean()),
    }


def plot(ticks, targets, out: Path, title: str):
    seeds = sorted(ticks.seed.unique())
    fig, axes = plt.subplots(2, len(seeds), figsize=(3.6 * len(seeds), 7), squeeze=False, constrained_layout=True)
    for i, s in enumerate(seeds):
        g = ticks[ticks.seed == s]
        ax = axes[0, i]
        ax.plot(g.x, g.y, "-", lw=1.5)
        ax.plot(0, 0, "ko", ms=4)
        ax.add_patch(plt.Circle(targets[s], 1.0, color="red", alpha=0.4))
        ax.plot(*targets[s], "r+")
        ax.set_aspect("equal")
        ax.set_title(f"seed {s}")
        ax.grid(alpha=0.3)
        ax2 = axes[1, i]
        tt = g.tick * 0.02
        ax2.plot(tt, np.rad2deg(g.bearing), label="target bearing (deg, +left)")
        ax2.plot(tt, g.turn_cmd * 90, label="turn cmd ×90 (+clockwise)")
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_xlabel("time (s)")
        ax2.grid(alpha=0.3)
        if i == 0:
            ax2.legend(fontsize=7)
    fig.suptitle(title)
    fig.savefig(out / "paths.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="training run dir (brain spec + env from config.json)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--brain", help="reference brain type instead of --run: reflex | random")
    ap.add_argument("--config", default="configs/reach_target.yaml")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--reanalyse", action="store_true", help="recompute summary.json from --out/ticks.csv (no simulation)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.reanalyse:
        out = Path(args.out)
        summary = analyse(pd.read_csv(out / "ticks.csv"), pd.read_csv(out / "episodes.csv"))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return
    if args.run:
        saved = json.loads((Path(args.run) / "config.json").read_text())
        env = EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved["env"].items()})
        spec = saved["brain"]
        params = np.load(args.checkpoint or Path(args.run) / "theta_best.npy")
    else:
        env, spec, params = load_config(args.config)["env"], {"type": args.brain}, None
    brain = make_brain(spec)
    if params is not None:
        brain.set_params(params)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fly = FlyInterface(env)
    eps, all_ticks, targets = [], [], {}
    for s in args.seeds:
        ep, df, target = trace_episode(brain, fly, s)
        eps.append(ep)
        all_ticks.append(df)
        targets[s] = target
        print(f"seed {s}: success {ep['success']} t2t {ep['time_to_target']} path {ep['distance_traveled']:.1f} mm "
              f"/ straight {ep['initial_distance']:.1f} mm", flush=True)
    fly.close()
    ticks, episodes = pd.concat(all_ticks), pd.DataFrame(eps)
    ticks.to_csv(out / "ticks.csv", index=False)
    episodes.to_csv(out / "episodes.csv", index=False)
    summary = analyse(ticks, episodes)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(ticks, targets, out, f"{args.run or args.brain} {Path(args.checkpoint).name if args.checkpoint else ''}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
