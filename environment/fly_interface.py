# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""FlyInterface: the one adapter between FlyGym and a brain.

    obs = fly.reset(seed)
    while True:
        fly.apply_action(brain.act(obs))      # brain output -> descending signal (2,)
        obs, reward, terminated, truncated, info = fly.step()   # one brain tick
        if terminated or truncated: break

Everything below the descending signal is FlyGym's own code, unchanged:
``HybridTurningController`` (CPG + preprogrammed steps + stumbling/retraction rules)
-> ``apply_locomotion_action`` -> ``Simulation.step`` (MuJoCo, dt = 1e-4 s). See
ARCHITECTURE.md §3 for the insertion point and §4 for what this adapter adds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco as mj
import numpy as np

from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

from environment.fast_controller import CachedHybridTurningController, CachedObservationExtractor
from environment.reach_world import PARKED_XY, ReachTargetWorld

OBS_NAMES = (
    "target_distance",
    "target_dir_cos",
    "target_dir_sin",
    "target_visible",
    "obstacle_distance",
    "obstacle_dir_cos",
    "obstacle_dir_sin",
    "velocity_forward",
    "velocity_lateral",
    "yaw_rate",
    "orientation_cos",
    "orientation_sin",
)
OBS_DIM = len(OBS_NAMES)

# Discrete actions as (forward, turn); turn > 0 = clockwise (right).
DISCRETE_ACTIONS = {
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "turn_left": (0.35, -0.65),
    "turn_right": (0.35, 0.65),
    "stop": (0.0, 0.0),
}


@dataclass
class EnvConfig:
    # Timing
    decision_interval: float = 0.02  # s of sim time per brain tick
    episode_length: float = 4.0  # s of sim time
    warmup: float = 0.05  # s; FlyGym's default settle time
    # Motor mapping
    drive_max: float = 1.2  # max |descending signal| (range used upstream)
    # Task layout (sampled per episode from the seed)
    target_distance_range: tuple[float, float] = (6.0, 10.0)  # mm
    target_bearing_range: tuple[float, float] = (-np.pi, np.pi)  # rad, rel. to heading
    reach_radius: float = 1.0  # mm, thorax-to-target-centre (xy)
    n_obstacles: int = 0
    obstacle_radius: float = 1.0  # mm
    obstacle_clearance: float = 1.5  # mm min gap to fly spawn and to target
    # Reward
    reward_scale: float = 1.0  # per mm of progress
    reach_bonus: float = 10.0
    collision_penalty: float = 0.5  # per brain tick with obstacle contact
    time_cost: float = 0.01  # per brain tick
    # Design Iteration 1 (configs/reach_target_v2.yaml): + heading_reward * cos(target bearing)
    # per tick. 0 = off, and the reward is then exactly the original one.
    heading_reward: float = 0.0
    # Observation
    fov_half_angle: float = np.deg2rad(135)  # compound eyes see ~270 deg
    distance_norm: float = 10.0  # mm
    velocity_norm: float = 15.0  # mm/s
    yaw_rate_norm: float = 3.0  # rad/s
    # Termination
    flip_up_z: float = 0.0  # terminate if thorax z-axis . world z < this
    # Simulation plumbing: cached, bit-identical controller path (tests/test_fast_controller.py)
    fast_controller: bool = True


class FlyInterface:
    """Adapter exposing ``reset / get_observation / apply_action / step`` on FlyGym.

    Args:
        config: Task and timing parameters.
        render: If True, attach FlyGym's `Renderer` (top-down + follow cameras).
        seed: Seed for the layout RNG (per-episode seeds can be passed to `reset`).
    """

    def __init__(
        self, config: EnvConfig | None = None, *, render: bool = False, seed: int = 0
    ) -> None:
        self.config = cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)

        self.fly = make_locomotion_fly(name="fly", add_adhesion=True, colorize=render)
        self.follow_camera = self.fly.add_tracking_camera(
            name="follow", pos_offset=(0.0, 0.0, 14.0), rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 1, 0)), fovy=45.0
        )
        self.world = ReachTargetWorld(
            n_obstacles=cfg.n_obstacles, obstacle_radius=cfg.obstacle_radius
        )
        self.world.add_fly(
            self.fly,
            [0, 0, 0.8],
            Rotation3D("quat", [1, 0, 0, 0]),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=False,
        )
        self.sim = Simulation(self.world)
        self.renderer = None
        if render:
            self.renderer = self.sim.set_renderer(
                ["topdown", self.follow_camera],
                camera_res=(352, 352),
                playback_speed=0.25,
                output_fps=25,
            )

        self.steps = PreprogrammedSteps()
        self.dof_order = self.fly.get_actuated_jointdofs_order("position")
        controller_cls = CachedHybridTurningController if cfg.fast_controller else HybridTurningController
        self.controller = controller_cls(
            timestep=self.sim.timestep,
            preprogrammed_steps=self.steps,
            output_dof_order=self.dof_order,
        )
        if cfg.fast_controller:
            self._controller_obs = CachedObservationExtractor(self.sim, self.fly.name, self.controller.legs)
        else:
            self._controller_obs = lambda: HybridControllerObservation.from_sim(self.sim, self.fly.name)

        m = self.sim.mj_model
        self._thorax_idx = self.fly.get_bodysegs_order().index(BodySegment("c_thorax"))
        self._target_mocap = m.body_mocapid[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "target")]
        self._obstacle_mocap = np.array(
            [m.body_mocapid[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b.name)] for b in self.world.obstacle_bodies],
            dtype=int,
        )
        self._obstacle_geom_ids = np.array(
            [mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, g.name) for g in self.world.obstacle_geoms],
            dtype=np.int32,
        )
        self._physics_steps_per_tick = max(1, round(cfg.decision_interval / self.sim.timestep))
        self._max_ticks = int(round(cfg.episode_length / cfg.decision_interval))

        # Episode state (set in reset)
        self.target_xy = np.zeros(2)
        self.obstacles_xy = np.zeros((0, 2))
        self.descending = np.zeros(2)
        self.tick = 0
        self._prev_xy = np.zeros(2)
        self._prev_yaw = 0.0
        self._velocity = np.zeros(3)  # fwd, lateral, yaw rate
        self._prev_distance = 0.0
        self.episode = {}

    # ------------------------------------------------------------------ layout
    def _sample_layout(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        dist = self.rng.uniform(*cfg.target_distance_range)
        bearing = self.rng.uniform(*cfg.target_bearing_range)
        # Fly spawns at the origin facing +x, so bearing == world angle here.
        target = dist * np.array([np.cos(bearing), np.sin(bearing)])
        obstacles = []
        for _ in range(cfg.n_obstacles):
            for _attempt in range(100):
                # Bias obstacles to lie between fly and target so they matter.
                s = self.rng.uniform(0.25, 0.75)
                p = s * target + self.rng.normal(0, 1.5, size=2)
                clear = cfg.obstacle_radius + cfg.obstacle_clearance
                if (
                    np.linalg.norm(p) > clear + 1.0
                    and np.linalg.norm(p - target) > clear + cfg.reach_radius
                    and all(np.linalg.norm(p - q) > 2 * cfg.obstacle_radius + 0.5 for q in obstacles)
                ):
                    obstacles.append(p)
                    break
        return target, np.array(obstacles).reshape(-1, 2)

    def set_layout(self, target_xy, obstacles_xy=()) -> None:
        """Place target and obstacles (world xy, mm). Unused obstacle slots are parked."""
        d = self.sim.mj_data
        self.target_xy = np.asarray(target_xy, dtype=float)
        self.obstacles_xy = np.asarray(obstacles_xy, dtype=float).reshape(-1, 2)
        if len(self.obstacles_xy) > len(self._obstacle_mocap):
            raise ValueError("More obstacles than compiled slots (EnvConfig.n_obstacles).")
        d.mocap_pos[self._target_mocap] = [*self.target_xy, 0.0]
        for i, mocap_id in enumerate(self._obstacle_mocap):
            xy = self.obstacles_xy[i] if i < len(self.obstacles_xy) else PARKED_XY
            d.mocap_pos[mocap_id] = [*xy, 0.0]

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, layout=None) -> np.ndarray:
        """Reset physics, controller and layout; returns the first observation.

        Args:
            seed: If given, reseeds the layout RNG and the CPG initial phases.
            layout: Optional ``(target_xy, obstacles_xy)`` to bypass random sampling.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim.reset()
        self.controller.reset(seed=int(self.rng.integers(2**31)))
        target, obstacles = layout if layout is not None else self._sample_layout()
        self.set_layout(target, obstacles)

        apply_locomotion_action(
            self.sim,
            self.fly.name,
            LocomotionAction(
                joint_angles=self.steps.default_pose_by_dof_order(self.dof_order),
                adhesion_onoff=np.ones(6, dtype=bool),
            ),
        )
        self.sim.warmup(self.config.warmup)

        self.descending = np.zeros(2)
        self.tick = 0
        xy, yaw = self._thorax_xy_yaw()
        self._prev_xy, self._prev_yaw = xy, yaw
        self._velocity = np.zeros(3)
        self._prev_distance = float(np.linalg.norm(self.target_xy - xy))
        self.episode = {
            "success": False,
            "steps_to_target": None,
            "time_to_target": None,
            "distance_traveled": 0.0,
            "collision_ticks": 0,
            "collision_onsets": 0,
            "reward": 0.0,
            "initial_distance": self._prev_distance,
            "final_distance": self._prev_distance,
            "flipped": False,
            "ticks": 0,
        }
        self._in_contact = False
        return self.get_observation()

    def apply_action(self, action) -> np.ndarray:
        """Set the descending signal held for the next brain tick.

        Accepts a continuous ``(forward, turn)`` in [-1, 1]^2 or a discrete action
        name from ``DISCRETE_ACTIONS``. Returns the resulting descending signal (2,).
        """
        if isinstance(action, str):
            action = DISCRETE_ACTIONS[action]
        forward, turn = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        m = self.config.drive_max
        # turn > 0 -> left side stronger -> clockwise turn (verified in M1/M3).
        self.descending = np.clip(m * np.array([forward + turn, forward - turn]), -m, m)
        return self.descending

    def step(self):
        """Advance one brain tick (``decision_interval`` of sim time).

        Returns:
            ``(obs, reward, terminated, truncated, info)``.
        """
        cfg = self.config
        contact_this_tick = False
        for _ in range(self._physics_steps_per_tick):
            action = self.controller.step(self.descending, self._controller_obs())
            apply_locomotion_action(self.sim, self.fly.name, action)
            self.sim.step()
            if len(self._obstacle_geom_ids) and self._obstacle_contact():
                contact_this_tick = True
            if self.renderer is not None:
                self.sim.render_as_needed()
        self.tick += 1

        xy, yaw = self._thorax_xy_yaw()
        dt = self._physics_steps_per_tick * self.sim.timestep
        world_vel = (xy - self._prev_xy) / dt
        c, s = np.cos(yaw), np.sin(yaw)
        self._velocity = np.array(
            [c * world_vel[0] + s * world_vel[1], -s * world_vel[0] + c * world_vel[1], _wrap(yaw - self._prev_yaw) / dt]
        )
        step_len = float(np.linalg.norm(xy - self._prev_xy))
        self._prev_xy, self._prev_yaw = xy, yaw

        distance = float(np.linalg.norm(self.target_xy - xy))
        reached = distance < cfg.reach_radius
        flipped = self._thorax_up_z() < cfg.flip_up_z
        reward = (
            cfg.reward_scale * (self._prev_distance - distance)
            + (cfg.reach_bonus if reached else 0.0)
            - (cfg.collision_penalty if contact_this_tick else 0.0)
            - cfg.time_cost
        )
        if cfg.heading_reward:
            to_target = self.target_xy - xy
            reward += cfg.heading_reward * float(np.cos(np.arctan2(to_target[1], to_target[0]) - yaw))
        self._prev_distance = distance

        ep = self.episode
        ep["ticks"] = self.tick
        ep["distance_traveled"] += step_len
        ep["reward"] += reward
        ep["final_distance"] = distance
        if contact_this_tick:
            ep["collision_ticks"] += 1
            if not self._in_contact:
                ep["collision_onsets"] += 1
        self._in_contact = contact_this_tick
        if reached and not ep["success"]:
            ep["success"] = True
            ep["steps_to_target"] = self.tick
            ep["time_to_target"] = self.tick * dt
        ep["flipped"] = bool(flipped)

        terminated = bool(reached or flipped)
        truncated = bool(self.tick >= self._max_ticks and not terminated)
        info = {"distance": distance, "collision": contact_this_tick, "descending": self.descending.copy()}
        return self.get_observation(), float(reward), terminated, truncated, info

    def get_observation(self) -> np.ndarray:
        """12-dim egocentric observation; see ``OBS_NAMES`` and ARCHITECTURE.md §4."""
        cfg = self.config
        xy, yaw = self._thorax_xy_yaw()
        to_target = self.target_xy - xy
        t_dist = float(np.linalg.norm(to_target))
        t_bearing = _wrap(np.arctan2(to_target[1], to_target[0]) - yaw)

        if len(self.obstacles_xy):
            rel = self.obstacles_xy - xy
            centre_d = np.linalg.norm(rel, axis=1)
            i = int(np.argmin(centre_d))
            o_dist = max(0.0, float(centre_d[i]) - cfg.obstacle_radius)
            o_bearing = _wrap(np.arctan2(rel[i, 1], rel[i, 0]) - yaw)
            o_present = 1.0
        else:
            o_dist, o_bearing, o_present = cfg.distance_norm * 3, 0.0, 0.0

        visible = abs(t_bearing) <= cfg.fov_half_angle and not self._occluded(xy, self.target_xy)
        return np.array(
            [
                min(t_dist / cfg.distance_norm, 3.0),
                np.cos(t_bearing),
                np.sin(t_bearing),
                float(visible),
                min(o_dist / cfg.distance_norm, 3.0),
                o_present * np.cos(o_bearing),
                o_present * np.sin(o_bearing),
                self._velocity[0] / cfg.velocity_norm,
                self._velocity[1] / cfg.velocity_norm,
                self._velocity[2] / cfg.yaw_rate_norm,
                np.cos(yaw),
                np.sin(yaw),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ helpers
    def thorax_xy(self) -> np.ndarray:
        return self._thorax_xy_yaw()[0]

    def _thorax_xy_yaw(self) -> tuple[np.ndarray, float]:
        pos = self.sim.get_body_positions(self.fly.name)[self._thorax_idx]
        w, x, y, z = self.sim.get_body_rotations(self.fly.name)[self._thorax_idx]
        # Body x-axis in world frame (first column of the rotation matrix); this is the
        # same heading FlyGym's HybridControllerObservation uses.
        hx, hy = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
        return pos[:2].copy(), float(np.arctan2(hy, hx))

    def _thorax_up_z(self) -> float:
        w, x, y, z = self.sim.get_body_rotations(self.fly.name)[self._thorax_idx]
        return float(1 - 2 * (x * x + y * y))  # world-z component of body z-axis

    def _obstacle_contact(self) -> bool:
        d = self.sim.mj_data
        n = d.ncon
        if n == 0:
            return False
        c = d.contact
        ids = self._obstacle_geom_ids
        # Only fly<->obstacle pairs exist for obstacle geoms, so any penetrating contact
        # involving an obstacle geom is a fly collision.
        hit = (np.isin(c.geom1[:n], ids) | np.isin(c.geom2[:n], ids)) & (c.dist[:n] <= 0)
        return bool(hit.any())

    def _occluded(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True if the segment a->b passes within an obstacle radius of any obstacle."""
        if not len(self.obstacles_xy):
            return False
        ab = b - a
        L2 = float(ab @ ab)
        if L2 == 0:
            return False
        t = np.clip(((self.obstacles_xy - a) @ ab) / L2, 0.0, 1.0)
        closest = a + t[:, None] * ab
        return bool((np.linalg.norm(self.obstacles_xy - closest, axis=1) < self.config.obstacle_radius).any())

    def save_video(self, path: str | Path) -> None:
        if self.renderer is None:
            raise RuntimeError("FlyInterface was created with render=False.")
        self.renderer.save_video(path)

    def close(self) -> None:
        self.sim.close()


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
