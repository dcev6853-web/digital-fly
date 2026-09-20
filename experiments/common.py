# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Config loading shared by train.py and evaluate.py."""

from __future__ import annotations

import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from brain.trainer import ESConfig  # noqa: E402
from environment.fly_interface import EnvConfig  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def load_config(path: str | Path) -> dict:
    """Read a YAML config and resolve it into EnvConfig, ESConfig and a brain spec.

    Top-level keys named in the spec (``episodes``, ``episode_length``,
    ``learning_rate``, ``reward_scale``, ``env_seed``) override the sections.
    ``episodes`` is the total training-episode budget; ES generations are derived from
    it: generations = episodes / (2 * pairs * episodes_per_candidate).
    """
    raw = yaml.safe_load(Path(path).read_text())
    env_kw = dict(raw.get("env", {}))
    es_kw = dict(raw.get("es", {}))
    if "episode_length" in raw:
        env_kw["episode_length"] = float(raw["episode_length"])
    if "reward_scale" in raw:
        env_kw["reward_scale"] = float(raw["reward_scale"])
    if "learning_rate" in raw:
        es_kw["learning_rate"] = float(raw["learning_rate"])
    if "env_seed" in raw:
        es_kw["seed"] = int(raw["env_seed"])
    for key in ("target_distance_range", "target_bearing_range"):
        if key in env_kw:
            env_kw[key] = tuple(float(v) for v in env_kw[key])
    _check_keys(env_kw, EnvConfig, "env")
    _check_keys(es_kw, ESConfig, "es")
    es = ESConfig(**es_kw)
    if "episodes" in raw:
        per_gen = 2 * es.pairs * es.episodes_per_candidate
        es.generations = max(1, int(raw["episodes"]) // per_gen)
    return {
        "name": raw.get("name", Path(path).stem),
        "env": EnvConfig(**env_kw),
        "es": es,
        "brain": dict(raw.get("brain", {"type": "mlp"})),
        "env_seed": int(raw.get("env_seed", 0)),
        "raw": raw,
    }


def _check_keys(kw: dict, cls, section: str) -> None:
    known = {f.name for f in fields(cls)}
    unknown = set(kw) - known
    if unknown:
        raise ValueError(f"Unknown keys in '{section}': {sorted(unknown)}")


def load_params(path: str | Path | None) -> np.ndarray | None:
    return None if path is None else np.load(path)
