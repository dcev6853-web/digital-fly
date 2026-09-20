"""Milestone 1: run the unmodified FlyGym turning-controller demo, headless.

This is the code from ``simulator/tutorials/4d_turning_controller.ipynb`` (FlyGym
v2.1.0, Apache-2.0, NeLy-EPFL) as a plain script, with only the output directory
changed. Nothing here is our work; it verifies that the simulator runs as-is on this
machine and measures its throughput.

    MUJOCO_GL=egl python experiments/m1_upstream_turning_demo.py
"""

import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

output_dir = Path(__file__).parent / "results" / "m1_upstream"
output_dir.mkdir(parents=True, exist_ok=True)

fly = make_locomotion_fly(name="turning_demo", add_adhesion=True, colorize=True)
body_cam = fly.add_tracking_camera(
    name="body_cam",
    pos_offset=(-0.5, -7.5, 0.0),
    rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
    fovy=35.0,
)
world = FlatGroundWorld()
world.add_fly(
    fly,
    [0, 0, 0.8],
    Rotation3D("quat", [1, 0, 0, 0]),
    bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
    add_ground_contact_sensors=False,
)
sim = Simulation(world)
renderer = sim.set_renderer(
    [body_cam], camera_res=(240, 320), playback_speed=0.1, output_fps=25
)

preprogrammed_steps = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")
controller = HybridTurningController(
    timestep=sim.timestep,
    preprogrammed_steps=preprogrammed_steps,
    output_dof_order=dof_order,
)
print("Actuated DoFs:", len(dof_order), "| timestep:", sim.timestep)

sim.reset()
controller.reset(seed=0)
apply_locomotion_action(
    sim,
    fly.name,
    LocomotionAction(
        joint_angles=preprogrammed_steps.default_pose_by_dof_order(dof_order),
        adhesion_onoff=np.ones(6, dtype=bool),
    ),
)
sim.warmup()

run_time = 2.0
nsteps = int(run_time / sim.timestep)
thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
thorax_positions = np.full((nsteps, 3), np.nan)

t0 = time.perf_counter()
for i in range(nsteps):
    signal = np.array([1.2, 0.4]) if i * sim.timestep < run_time / 2 else np.array([0.4, 1.2])
    obs = HybridControllerObservation.from_sim(sim, fly.name)
    action = controller.step(signal, obs)
    apply_locomotion_action(sim, fly.name, action)
    sim.step_with_profile()
    thorax_positions[i] = sim.get_body_positions(fly.name)[thorax_idx]
    sim.render_as_needed_with_profile()
wall = time.perf_counter() - t0

disp = thorax_positions[-1] - thorax_positions[0]
half = thorax_positions[nsteps // 2] - thorax_positions[0]
print(f"Displacement at t=1s: {np.round(half, 3)} mm; final: {np.round(disp, 3)} mm")
print(f"Wall time {wall:.1f}s for {run_time}s sim -> {run_time / wall:.2f}x real time (incl. rendering)")
sim.print_performance_report(show_in_notebook=False)
renderer.save_video(output_dir / "hybrid_turning_controller.mp4")
np.save(output_dir / "thorax_positions.npy", thorax_positions)
print("Saved", output_dir / "hybrid_turning_controller.mp4")
