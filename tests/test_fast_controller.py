"""The cached controller path must be bit-identical to FlyGym's upstream path."""

import numpy as np

from environment.fly_interface import EnvConfig, FlyInterface


def rollout(fast: bool, seed: int, ticks: int):
    fly = FlyInterface(EnvConfig(n_obstacles=1, fast_controller=fast))
    fly.reset(seed=seed, layout=([6.0, 3.0], [[3.0, 0.3]]))  # obstacle in the path -> stumbling rules fire
    rng = np.random.default_rng(seed)
    trace = []
    for _ in range(ticks):
        fly.apply_action(rng.uniform(-1, 1, 2))
        obs, r, *_ = fly.step()
        trace.append((fly.sim.mj_data.qpos.copy(), fly.sim.mj_data.ctrl.copy(), obs, r))
    fly.close()
    return trace


def test_bit_identical():
    for seed in (0, 1):
        a, b = rollout(False, seed, 30), rollout(True, seed, 30)
        for (qa, ca, oa, ra), (qb, cb, ob, rb) in zip(a, b):
            assert np.array_equal(ca, cb) and np.array_equal(qa, qb) and np.array_equal(oa, ob) and ra == rb
