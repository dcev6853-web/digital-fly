# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Evolution-strategies trainer (numpy + multiprocessing, CPU only).

Why ES: it needs no gradients through the brain, so the same optimiser trains the MLP
baseline and the recurrent, sparse MANC-constrained model (fair comparison, no torch).
Episodes are independent, so they spread across CPU cores.

Algorithm (OpenAI-ES style): antithetic Gaussian perturbations of the flat parameter
vector, centred-rank fitness shaping, Adam on the estimated gradient. All candidates
in one generation see the same episode seeds (common random numbers).
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from brain.interface import EPISODE_FIELDS, run_episode

# ---------------------------------------------------------------- worker pool
_worker = {}


def _init_worker(env_config, brain_spec):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from environment.fly_interface import FlyInterface
    from brain.model import make_brain

    _worker["fly"] = FlyInterface(env_config)
    _worker["brain"] = make_brain(brain_spec)


def _run_job(job):
    params, seed = job
    brain = _worker["brain"]
    if params is not None and len(params):
        brain.set_params(params)
    return run_episode(brain, _worker["fly"], seed=int(seed))


class EpisodePool:
    """A pool of worker processes, each holding one compiled FlyInterface + brain."""

    def __init__(self, env_config, brain_spec: dict, workers: int):
        ctx = mp.get_context("spawn")  # no inherited MuJoCo/GL state
        self.pool = ctx.Pool(workers, initializer=_init_worker, initargs=(env_config, brain_spec))

    def run(self, jobs: list[tuple[np.ndarray | None, int]]) -> list[dict]:
        return self.pool.map(_run_job, jobs, chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()


# ---------------------------------------------------------------- ES
@dataclass
class ESConfig:
    generations: int = 50
    pairs: int = 8  # antithetic pairs -> population = 2 * pairs
    sigma: float = 0.1
    learning_rate: float = 0.05
    episodes_per_candidate: int = 2
    weight_decay: float = 0.005
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    eval_every: int = 5
    eval_episodes: int = 16
    eval_seed_offset: int = 10_000_000  # eval seeds never overlap training seeds
    seed: int = 0


class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, grad):
        """Return the parameter *increment* for an ascent step along ``grad``."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        return self.lr * mhat / (np.sqrt(vhat) + self.eps)


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to [-0.5, 0.5]. Ties get their average rank, so identical fitnesses
    contribute nothing to the ES gradient (with an ordinal argsort, ties were ranked by
    index and produced a spurious random-walk update). Without ties this is identical to
    the ordinal version."""
    from scipy.stats import rankdata

    ranks = rankdata(x, method="average") - 1
    return ranks / (len(x) - 1) - 0.5


def summarize(episodes: list[dict]) -> dict:
    succ = [e for e in episodes if e["success"]]
    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_steps_to_target": float(np.mean([e["steps_to_target"] for e in succ])) if succ else None,
        "mean_time_to_target": float(np.mean([e["time_to_target"] for e in succ])) if succ else None,
        "mean_distance_traveled": float(np.mean([e["distance_traveled"] for e in episodes])),
        "mean_collision_ticks": float(np.mean([e["collision_ticks"] for e in episodes])),
        "mean_collision_onsets": float(np.mean([e["collision_onsets"] for e in episodes])),
        "mean_reward": float(np.mean([e["reward"] for e in episodes])),
        "mean_final_distance": float(np.mean([e["final_distance"] for e in episodes])),
        "flipped": int(sum(e["flipped"] for e in episodes)),
    }


class CSVLog:
    def __init__(self, path: Path, fields: list[str]):
        self.fields = fields
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        self.w.writeheader()

    def write(self, row: dict):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


class ESTrainer:
    def __init__(self, brain, brain_spec: dict, env_config, es: ESConfig, out_dir: Path):
        self.brain, self.brain_spec, self.env_config, self.es = brain, brain_spec, env_config, es
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.theta = brain.get_params().astype(float)
        self.rng = np.random.default_rng(es.seed)

    def evaluate(self, pool: EpisodePool, theta, n: int, seed0: int) -> list[dict]:
        return pool.run([(theta, seed0 + i) for i in range(n)])

    def train(self) -> dict:
        es, n = self.es, len(self.theta)
        adam = Adam(n, es.learning_rate)
        ep_fields = ["generation", "candidate", "sign"] + list(EPISODE_FIELDS) + ["seed"]
        train_log = CSVLog(self.out_dir / "train_episodes.csv", ep_fields)
        eval_log = CSVLog(self.out_dir / "eval_episodes.csv", ["generation"] + list(EPISODE_FIELDS) + ["seed"])
        gen_log = CSVLog(
            self.out_dir / "generations.csv",
            ["generation", "wall_s", "mean_fitness", "max_fitness", "train_success_rate", "theta_norm",
             "eval_success_rate", "eval_mean_reward", "eval_mean_time_to_target", "eval_mean_collision_ticks"],
        )
        (self.out_dir / "config.json").write_text(json.dumps(
            {"es": asdict(es), "env": asdict(self.env_config), "brain": self.brain_spec, "n_params": n}, indent=2, default=float))

        pool = EpisodePool(self.env_config, self.brain_spec, es.workers)
        best = {"eval_success_rate": -1.0}
        t_start = time.time()
        try:
            for gen in range(es.generations + 1):
                t0 = time.time()
                row = {"generation": gen}
                # Periodic evaluation of the current mean parameters on held-out seeds.
                if gen % es.eval_every == 0 or gen == es.generations:
                    evals = self.evaluate(pool, self.theta, es.eval_episodes, es.eval_seed_offset)
                    for e in evals:
                        eval_log.write({"generation": gen} | e)
                    s = summarize(evals)
                    row |= {"eval_success_rate": s["success_rate"], "eval_mean_reward": s["mean_reward"],
                            "eval_mean_time_to_target": s["mean_time_to_target"],
                            "eval_mean_collision_ticks": s["mean_collision_ticks"]}
                    np.save(self.out_dir / "theta_last.npy", self.theta)
                    if s["success_rate"] > best["eval_success_rate"] or (
                        s["success_rate"] == best["eval_success_rate"] and s["mean_reward"] > best.get("eval_mean_reward", -np.inf)
                    ):
                        best = {"generation": gen, "eval_success_rate": s["success_rate"], "eval_mean_reward": s["mean_reward"]}
                        np.save(self.out_dir / "theta_best.npy", self.theta)
                    print(f"[gen {gen:3d}] EVAL success {s['success_rate']:.2f} reward {s['mean_reward']:.2f} "
                          f"t2t {s['mean_time_to_target']} coll {s['mean_collision_ticks']:.1f}", flush=True)
                if gen == es.generations:
                    gen_log.write(row)
                    break

                # One ES generation.
                eps = self.rng.standard_normal((es.pairs, n))
                seeds = [gen * 1000 + k for k in range(es.episodes_per_candidate)]
                jobs, meta = [], []
                for i in range(es.pairs):
                    for sign in (+1, -1):
                        cand = self.theta + sign * es.sigma * eps[i]
                        for s_ in seeds:
                            jobs.append((cand, s_))
                            meta.append((i, sign))
                results = pool.run(jobs)
                fitness = np.zeros((es.pairs, 2))
                for (i, sign), res in zip(meta, results):
                    fitness[i, 0 if sign > 0 else 1] += res["reward"] / es.episodes_per_candidate
                    train_log.write({"generation": gen, "candidate": i, "sign": sign} | res)
                if np.unique(fitness.round(12)).size == 1:
                    print(f"[gen {gen:3d}] WARNING degenerate generation: all {fitness.size} candidates scored "
                          f"{fitness.flat[0]:.4f} (no learning signal)", flush=True)
                ranks = centered_ranks(fitness.ravel()).reshape(fitness.shape)
                grad = ((ranks[:, 0] - ranks[:, 1])[:, None] * eps).sum(0) / (2 * es.pairs * es.sigma)
                self.theta = self.theta * (1 - es.weight_decay * es.learning_rate) + adam.step(grad)

                row |= {"wall_s": round(time.time() - t0, 1), "mean_fitness": float(fitness.mean()),
                        "max_fitness": float(fitness.max()),
                        "train_success_rate": float(np.mean([r["success"] for r in results])),
                        "theta_norm": float(np.linalg.norm(self.theta))}
                gen_log.write(row)
                print(f"[gen {gen:3d}] fit {fitness.mean():6.2f} (max {fitness.max():6.2f}) "
                      f"succ {row['train_success_rate']:.2f} {row['wall_s']}s", flush=True)
        finally:
            pool.close()
            for log in (train_log, eval_log, gen_log):
                log.close()
        summary = {"best": best, "wall_s": round(time.time() - t_start, 1), "n_params": n}
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
