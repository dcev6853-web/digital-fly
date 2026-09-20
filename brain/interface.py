# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""The contract every brain implements, and the episode loop that drives one.

A brain maps the 12-dim observation from `environment.fly_interface.FlyInterface` to a
continuous action ``(forward, turn)`` in [-1, 1]^2. `FlyInterface.apply_action` turns
that into FlyGym's 2-D descending signal. Learnable brains expose their parameters as
one flat vector so that any brain can be trained by the same optimiser.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Brain(Protocol):
    n_params: int

    def reset(self) -> None:
        """Clear any internal (recurrent) state at the start of an episode."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return ``(forward, turn)`` in [-1, 1]^2 for one brain tick."""

    def get_params(self) -> np.ndarray: ...

    def set_params(self, flat: np.ndarray) -> None: ...


EPISODE_FIELDS = (
    "success",
    "steps_to_target",
    "time_to_target",
    "distance_traveled",
    "collision_ticks",
    "collision_onsets",
    "reward",
    "initial_distance",
    "final_distance",
    "flipped",
    "ticks",
)


def run_episode(brain: Brain, fly, seed: int, layout=None, on_tick=None) -> dict:
    """Run one episode of ``brain`` on a `FlyInterface`; return its episode metrics.

    Args:
        brain: Any object implementing `Brain`.
        fly: A `FlyInterface`.
        seed: Episode seed (layout and CPG initial phases).
        layout: Optional fixed ``(target_xy, obstacles_xy)``.
        on_tick: Optional callback ``f(obs, action, reward, info)`` per brain tick.
    """
    obs = fly.reset(seed=seed, layout=layout)
    brain.reset()
    while True:
        action = brain.act(obs)
        fly.apply_action(action)
        obs, reward, terminated, truncated, info = fly.step()
        if on_tick is not None:
            on_tick(obs, action, reward, info)
        if terminated or truncated:
            break
    return {k: fly.episode[k] for k in EPISODE_FIELDS} | {"seed": seed}
