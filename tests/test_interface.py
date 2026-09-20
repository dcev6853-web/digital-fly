"""Milestones 2-3: world + FlyInterface round trip (sensory in, motor out)."""

import numpy as np
import pytest

from environment.fly_interface import OBS_DIM, EnvConfig, FlyInterface


@pytest.fixture(scope="module")
def fly():
    f = FlyInterface(EnvConfig(n_obstacles=1, episode_length=1.0))
    yield f
    f.close()


def run(fly, action, ticks):
    fly.apply_action(action)
    out = None
    for _ in range(ticks):
        out = fly.step()
    return out


def yaw(obs):
    return np.arctan2(obs[11], obs[10])


def test_observation_shape_and_target_geometry(fly):
    obs = fly.reset(seed=0, layout=([6.0, 0.0], [[500.0, -500.0]]))
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
    # Target 6 mm straight ahead: distance ~0.6 (normalised), bearing ~0.
    assert abs(obs[0] - 0.6) < 0.1 and obs[1] > 0.95 and obs[3] == 1.0


def test_forward_makes_progress_and_positive_reward(fly):
    obs0 = fly.reset(seed=0, layout=([8.0, 0.0], [[500.0, -500.0]]))
    total = 0.0
    fly.apply_action("forward")
    for _ in range(25):  # 0.5 s
        obs, r, term, trunc, info = fly.step()
        total += r
    assert obs[0] < obs0[0] - 0.1, "distance to target should shrink"
    assert obs[7] > 0.1, "forward velocity should be positive"
    assert total > 0


@pytest.mark.parametrize("action,sign", [((0.3, 0.7), -1), ((0.3, -0.7), +1)])
def test_turn_sign(fly, action, sign):
    obs0 = fly.reset(seed=1, layout=([8.0, 0.0], [[500.0, -500.0]]))
    obs = run(fly, action, 25)[0]
    dyaw = np.angle(np.exp(1j * (yaw(obs) - yaw(obs0))))
    assert sign * dyaw > 0.2, f"turn {action} gave dyaw={dyaw:.2f}"


def test_obstacle_collision_detected(fly):
    fly.reset(seed=0, layout=([10.0, 0.0], [[3.5, 0.0]]))
    obs = fly.get_observation()
    assert obs[4] < 0.3 and obs[5] > 0.9, "obstacle ahead should be close and at bearing ~0"
    assert obs[3] == 0.0, "target behind the obstacle should be occluded"
    run(fly, "forward", 40)
    assert fly.episode["collision_ticks"] > 0


def test_heading_reward_off_by_default_and_signed_when_on():
    """heading_reward=0 must leave the reward untouched; when on, facing the target pays."""
    rewards = {}
    for c in (0.0, 0.02):
        f = FlyInterface(EnvConfig(heading_reward=c, episode_length=0.2))
        f.reset(seed=5, layout=([6.0, 0.0], []))  # target straight ahead
        f.apply_action("stop")
        rewards[c] = f.step()[1]
        f.close()
    assert abs((rewards[0.02] - rewards[0.0]) - 0.02) < 1e-3  # cos(bearing ~ 0) ~ 1
