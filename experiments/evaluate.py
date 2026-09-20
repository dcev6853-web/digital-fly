# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Evaluate a brain on held-out episodes and log per-episode metrics.

    python experiments/evaluate.py --run experiments/results/<run>          # trained brain
    python experiments/evaluate.py --brain reflex                           # reference
    python experiments/evaluate.py --run ... --visual                       # + mp4 videos
    python experiments/evaluate.py --run ... --live                         # MuJoCo viewer

Test seeds start at 20,000,000, disjoint from training (gen*1000+k) and from the
in-training evaluation seeds (10,000,000+).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import RESULTS, load_config, load_params  # noqa: I001  (sets sys.path)

import numpy as np

from brain._provenance import PROJECT_ID
from brain.interface import EPISODE_FIELDS, run_episode
from brain.model import make_brain
from brain.trainer import CSVLog, EpisodePool, summarize
from environment.fly_interface import FlyInterface

TEST_SEED_OFFSET = 20_000_000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/reach_target.yaml")
    ap.add_argument("--run", help="training run dir (uses its config.json brain + theta_best.npy)")
    ap.add_argument("--checkpoint", help="explicit parameter file (.npy)")
    ap.add_argument("--brain", help="brain type override: mlp | reflex | random")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed-offset", type=int, default=TEST_SEED_OFFSET)
    ap.add_argument("--n-obstacles", type=int, help="override env.n_obstacles")
    ap.add_argument("--env", nargs="*", default=[], help="env overrides key=value (YAML values)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--visual", action="store_true", help="also render the first --videos episodes to mp4")
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--live", action="store_true", help="open the MuJoCo viewer on one episode")
    ap.add_argument("--out", help="output dir (default: <run>/eval_<tag> or results/eval_<brain>)")
    ap.add_argument("--tag", help="names the output dir eval_<tag>. Official sets use test / g1_obstacles / g2_far; "
                    "without --tag an ad-hoc run gets its own timestamped dir, so it never lands on official results")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing eval dir that has results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    env, brain_spec = cfg["env"], cfg["brain"]
    params = load_params(args.checkpoint)
    if args.run:
        run = Path(args.run)
        saved = json.loads((run / "config.json").read_text())
        brain_spec = saved["brain"]
        env = type(env)(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved["env"].items()})
        if params is None:
            params = np.load(run / "theta_best.npy")
    if args.brain:
        brain_spec = {"type": args.brain}
    if args.n_obstacles is not None:
        env.n_obstacles = args.n_obstacles
    import yaml
    for kv in args.env:
        k, v = kv.split("=", 1)
        v = yaml.safe_load(v)
        setattr(env, k, tuple(v) if isinstance(v, list) else v)

    tag = args.tag or time.strftime("adhoc_%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else (Path(args.run) / f"eval_{tag}" if args.run else RESULTS / f"eval_{brain_spec['type']}_{tag}")
    if (out / "summary.json").exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing results in {out} (pick another --tag/--out, or pass --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed_offset + i for i in range(args.episodes)]

    t0 = time.time()
    workers = args.workers or cfg["es"].workers
    pool = EpisodePool(env, brain_spec, workers=min(workers, args.episodes))
    episodes = pool.run([(params, s) for s in seeds])
    pool.close()

    log = CSVLog(out / "episodes.csv", list(EPISODE_FIELDS) + ["seed"])
    for e in episodes:
        log.write(e)
    log.close()
    summary = summarize(episodes) | {"brain": brain_spec, "env": env.__dict__, "wall_s": round(time.time() - t0, 1),
                                     "project_id": PROJECT_ID}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("env",)}, indent=2, default=float))
    print("wrote", out)

    if args.visual or args.live:
        fly = FlyInterface(env, render=args.visual)
        brain = make_brain(brain_spec)
        if params is not None and len(params):
            brain.set_params(params)
        if args.live:
            live_episode(fly, brain, seeds[0])
        if args.visual:
            for s in seeds[: args.videos]:
                res = run_episode(brain, fly, seed=s)
                vdir = out / "videos" / f"seed{s}_{'success' if res['success'] else 'fail'}"
                fly.save_video(vdir)
                print("video:", vdir, "| success", res["success"], "t2t", res["time_to_target"])
        fly.close()


def live_episode(fly, brain, seed):
    """Run one episode in MuJoCo's passive viewer (needs a display, e.g. WSLg)."""
    import mujoco.viewer

    with mujoco.viewer.launch_passive(fly.sim.mj_model, fly.sim.mj_data) as viewer:
        obs = fly.reset(seed=seed)
        brain.reset()
        done = False
        while viewer.is_running() and not done:
            t = time.time()
            fly.apply_action(brain.act(obs))
            obs, _, term, trunc, _ = fly.step()
            done = term or trunc
            viewer.sync()
            time.sleep(max(0.0, fly.config.decision_interval - (time.time() - t)))


if __name__ == "__main__":
    main()
