# ARCHITECTURE — Digital Fly learning architecture

Status: Step 0 and Milestones 1–4 verified. Milestone 5 (baseline training) is in progress. See §8. Every
API named below was read in the source at the pinned commit. None of it is assumed.

## 1. Simulator choice

| | |
|---|---|
| Repo | [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym) (NeuroMechFly v2) |
| Pinned | `v2.1.0-7-g38c8ec6` (commit `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`, 2026-06-29) |
| License | Apache-2.0 (`simulator/LICENSE`, kept intact) |
| Location | `simulator/`, a plain clone that we never edit. `git -C simulator status` must stay clean. |

FlyGym 2.x is a full rewrite (March 2026) and does not keep the 1.x `gymnasium` API. The
1.x code lives on as `flygym-gymnasium`. We target 2.x and use no 1.x APIs.

## 2. Step 0 map (what the repo actually is)

| Concern | Where / how | Notes |
|---|---|---|
| Fly representation | `compose/fly/neuromechfly.py` `NeuroMechFly(BaseFly)`, built in code through MuJoCo `MjSpec` (`base_fly.py`: `add_joints`, `add_actuators`, `add_leg_adhesion`, `add_vision`, `add_tracking_camera`) | Mesh body from a micro-CT scan of a female fly. Units: mm, s. Gravity -9810 mm/s². |
| Connectome storage | **None.** `grep -ri "flywire\|fafb\|connectome\|flyvis\|manc"` over `simulator/src` returns nothing. | FlyGym is a biomechanics, sensor and controller framework with no neural data. There is no FAFB default to swap out (see §5). |
| Neuron sim method | None in core. The "VNC" is an abstract controller in `flygym_demo/complex_terrain/`: a CPG of 6 coupled phase-amplitude oscillators (`cpg_controller.py:30`, 12 Hz tripod), plus recorded step kinematics (`preprogrammed.py`), plus hybrid stumbling and retraction rules (`hybrid_controller.py:103`). | Rate/phase model, not spiking. |
| Sensory input path | `Simulation` getters (`simulation.py`): `get_body_positions` L181, `get_body_rotations` L194, `get_joint_angles/velocities` L155/168, `get_ground_contact_info` L223, `get_bodysegment_contact_forces` L261, `get_raw_vision` L408, `get_ommatidia_readouts` L463 (2 eyes × 721 ommatidia × 2 channels). Raw `mj_data` is public too. | No odour in 2.x core. Vision needs `fly.add_vision()` before compile. |
| Motor output path | `Simulation.set_actuator_inputs` L343 (42 leg position actuators = 7 DoF × 6 legs), `set_leg_adhesion_states` L365. The helper `apply_locomotion_action` (`common.py`) writes both from a `LocomotionAction`. | Actuators are position servos (kp 45). |
| Movement control | `HybridTurningController.step(descending_signal: (2,), obs)` (`turning_controller.py:18`). Left/right drive sets the CPG amplitude per side, and the sign sets the stepping direction. It returns joint angles plus adhesion. | **This 2-D descending signal is the brain→VNC interface.** |
| 3D environment | `compose/world/flat_ground.py` `FlatGroundWorld` (1 m² checker plane), `complex_terrain.py`, `tethered_world.py`. `BaseWorld.mjcf_root` is a raw `MjSpec`, so geoms can be added. | **All collisions are explicit contact pairs** (`base_world.py:293`, `add_pair`). Every geom is `contype=0, conaffinity=0`. A new obstacle is non-colliding until we add pairs. |
| Animation linkage | `rendering.py` `Renderer` records offscreen frames from any camera (`Simulation.set_renderer`), and `save_video` writes mp4. `launch_interactive_viewer` L401 wraps `mujoco.viewer`. | EGL works headless here, at 178 ms/frame. WSLg display `:0` is available for the viewer. |
| Entry point | Library only. The tutorials (`tutorials/4d_turning_controller.ipynb`) are the reference loop: build fly → build world → `Simulation(world)` → `reset` → `warmup` → per step {obs → controller → apply → `sim.step()`}. | |
| Deps / install | Python ≥3.12,<3.15, `mujoco>=3.9,<3.10`, numpy 2, numba, scipy. Installed with `pip install -e simulator[examples]` into `.venv` (Python 3.12, aarch64 wheels) → mujoco 3.9.0. | Upstream tests `test_simulation`, `test_rendering` and `test_complex_terrain_locomotion`: **76 passed**. |

### Measured performance (this machine: aarch64, 12 cores, WSL2, CPU only)

`experiments/m1_upstream_turning_demo.py`, 2 s of simulated time:

- Physics: 113 µs/step at dt = 1e-4 s, i.e. 0.89× real time.
- Upstream hybrid controller and observation extraction: about 430 µs/step of Python. **This dominates headless cost.**
- Headless total ≈ 0.55 ms/step, so **1 s simulated ≈ 5.5 s wall per core**.
- Rendering: 178 ms/frame. Use it for evaluation videos only, never while training.
- With 11 parallel workers, each episode runs about 3× slower than alone (the shared-CPU
  VM saturates), so throughput is roughly 1 simulated second per wall second for the whole pool.
- **Optimisation, done after the pipeline worked:** `environment/fast_controller.py`
  subclasses `HybridTurningController` and caches the DoF and body index maps. It reuses
  the upstream math (CPG, splines, reflex rules) in the same floating-point order.
  `tests/test_fast_controller.py` checks that joint commands, MuJoCo `qpos`,
  observations and rewards are **bit-identical** to the upstream path. Measured speedup
  is about 1.6× per core (8.4–9.2 → 5.2–5.8 s wall per simulated second under load).
  `EnvConfig.fast_controller=False` restores the upstream path.
- Behaviour: drive `[1.2, 0.4]` for 1 s moves the fly (1.5, −7.3) mm (a right turn), then `[0.4, 1.2]` turns it back left. Walking speed is about 8–15 mm/s.

## 3. Pipeline and insertion point

### Board version (the one-picture summary)

```
   ┌──────────────┐   12 numbers    ┌───────────────────────────┐  2 numbers   ┌───────────────┐
   │ WHAT THE FLY │ ──────────────► │       MY BRAIN            │ ───────────► │  FLY BODY     │
   │ SENSES       │  how far/which  │  learned encoder          │  drive the   │  FlyGym CPG   │
   │              │  way the target │            ↓              │  LEFT legs,  │  + reflexes   │
   │ target,      │  is, nearest    │  MANC nerve-cord circuit  │  drive the   │  → 42 leg     │
   │ obstacles,   │  obstacle, own  │  5,041 real neurons,      │  RIGHT legs  │    joints     │
   │ own motion   │  speed, heading │  136k fixed connections   │              │  → MuJoCo     │
   └──────────────┘                 │            ↓              │              └───────────────┘
          ▲                         │  left / right leg-muscle  │                      │
          │                         │  pools                    │                      │
          │                         └───────────────────────────┘                      │
          └───────────  the fly moves, the world changes, sense again  ◄────────────────┘

   everything on the left and right is FlyGym, unmodified. only the middle box is mine.
```

### Detailed version

```
 FlyGym (untouched)                         ours                               FlyGym (untouched)
┌──────────────────────┐   ┌──────────────────────────────────────────┐   ┌─────────────────────────────────┐
│ MuJoCo world + fly   │   │ environment/fly_interface.py             │   │ HybridTurningController.step    │
│  mj_data (1e-4 s)    │──▶│  get_observation(): thorax pos/quat/vel, │   │  (descending_signal[2], obs)    │
│  Simulation getters  │   │   target & obstacle geometry → obs[12]   │   │   → CPG amps/dirs per side      │
│  (+ ommatidia, M9)   │   │                                          │   │   → preprogrammed steps         │
└──────────────────────┘   │        ┌────────── brain/ ───────────┐   │   │   → stumbling/retraction rules  │
                           │ obs ──▶│ MY ARCHITECTURE             │   │   │   → 42 joint angles + adhesion  │
                           │        │  (baseline: MLP)            │──▶│──▶│ apply_locomotion_action         │
                           │        │  (M6: MANC-constrained VNC) │   │   │  → sim.set_actuator_inputs      │
                           │        └─────────────────────────────┘   │   │  → sim.step()  → Renderer       │
                           │  apply_action(a): a → descending[2]      │   └─────────────────────────────────┘
                           └──────────────────────────────────────────┘
```

**Insertion point:** the `descending_signal` argument of `HybridTurningController.step`
(`simulator/src/flygym_demo/complex_terrain/turning_controller.py:18`). Upstream, a fixed
array goes there. We replace that array with `brain(obs)`. Everything downstream stays
FlyGym's: the CPG, step kinematics, reflex rules, actuators and physics. This is the same
brain/VNC split that NeuroMechFly v2 itself proposes (descending ⇄ ascending
representations).

**Timing:** physics and the upstream controller step every 0.1 ms. The brain ticks every
`decision_interval` (default 20 ms = 200 physics steps, about 4 decisions per 12 Hz step
cycle) and holds its output in between. `FlyInterface.step()` means one brain tick.

## 4. Adapters: what the repo lacks and the smallest fix

| Missing | Smallest adapter (in `environment/`, no simulator edits) |
|---|---|
| No target or obstacle objects | `ReachTargetWorld(FlatGroundWorld)` adds a visual-only target disc and cylinder obstacles to `mjcf_root.worldbody` before `add_fly`. |
| New geoms never collide (explicit-pair design) | After `add_fly`, add `add_pair(fly_geom, obstacle_geom)` for thorax, head, abdomen and leg geoms, reusing the ground `ContactParams`. |
| No task observations or reward | Computed in `FlyInterface` from `get_body_positions/rotations` (thorax) and the known target and obstacle geometry. |
| No collision signal | Scan `mj_data.contact[:ncon]` for (fly geom, obstacle geom) pairs. We use MuJoCo's own contact list and fabricate no proxy. |
| No 2.x gym env for locomotion (`flygym_demo/muscle_imitation/env.py` is task-specific) | `FlyInterface` exposes `reset/get_observation/apply_action/step`. No gymnasium dependency. |

The observation vector is 12 dims, egocentric and normalised:
`target_distance, target_dir (cos, sin), target_visible, obstacle_distance, obstacle_dir
(cos, sin), velocity (fwd, lateral, yaw_rate), orientation (cos, sin of world heading)`.
`target_visible` is geometric: the target is within ±135° of the heading (the fly's
compound eyes cover roughly 270°) and no obstacle blocks the line of sight. These are
privileged geometric features. The real ommatidia input from `get_ommatidia_readouts`
is kept for Milestone 9.

The action is continuous `(forward, turn) ∈ [-1,1]²`, mapped to
`descending = clip(drive_max·[forward + turn, forward − turn], −drive_max, drive_max)`
with `drive_max = 1.2` (the range used upstream). "Turn right" (`turn > 0`) makes the left
side stronger, which turns the fly clockwise (verified in M1). The discrete actions
forward, backward, turn_left, turn_right and stop map to fixed `(forward, turn)` pairs.

Reward per brain tick:
`r = reward_scale · (d_prev − d_now) + R_reach·[d_now < r_reach] − c_coll·[contact with obstacle] − c_time`.

## 5. Connectome: MANC integration path

### What "male CNS / MANC" means, and what we use

The spec names "the male CNS connectome (MANC — Male Adult Nerve Cord, Janelia/FlyWire)".
Those names cover three different datasets, so the choice is stated here rather than made
silently:

| Dataset | Scope | Source | Access |
|---|---|---|---|
| **MANC v1.0** (used) | Male **ventral nerve cord only**, about 23k traced neurons | Janelia FlyEM + Google (not FlyWire) | Public bucket `gs://flyem-manc-exports/v1.0/`, **no token**, CC-BY 4.0 |
| MANC v1.2.1 | Same volume, newer proofreading | Janelia | neuPrint only (needs a personal API token) |
| MaleCNS v1.0 (June 2026) | Whole male CNS, brain + VNC, a different animal | Janelia FlyEM | Public bucket `gs://flyem-male-cns/v1.0/`, 1.1 GB weights, CC-BY 4.0 |
| FlyWire / FAFB | Female **brain** | Princeton/FlyWire | Not used |

**Decision:** we use MANC v1.0, the dataset the spec names by acronym, because it is the
only MANC version with token-free downloads. FlyGym ships no connectome, so nothing is
being replaced. If you intended the whole-CNS MaleCNS (which would also constrain the
brain side), the loader is written against the same schema family (bodyId, type,
class, predictedNt, pre/post weights). Switching is a data swap plus a class-name mapping,
flagged in `configs/`.

### Files (in `data/manc_v1.0/`, gitignored, fetched by `scripts/fetch_manc.sh`, SHA256 in `SHA256SUMS`)

- `traced-neurons.csv` (0.6 MB): bodyId, type, instance for 23,188 traced neurons.
- `traced-connections.csv` (75 MB): bodyId_pre, bodyId_post, weight (synapse count), 5.24M edges.
- `manc-v1.0-neuron-properties.feather` (17 MB): class, subclass (fl/ml/hl = front/middle/hind leg), somaSide/rootSide, predictedNt, and more.

### Verified contents (traced neurons)

1,328 descending neurons (478 types, 664 per side) and 369 leg motor neurons (fl 132,
ml 114, hl 123, split by soma side). Known steering and walking DNs are present: DNa01 and
DNa02 (steering), DNp09 (forward walking), MDN (4, backward walking) and DNp01 (the giant
fibre).

The DN → interneuron → leg-MN subgraph (DN-targeted interneurons that also synapse onto
leg MNs, plus direct DN→MN edges):

| synapse threshold | DNs | interneurons | leg MNs | nodes | edges |
|---|---|---|---|---|---|
| ≥3 | 1,312 | 6,269 | 369 | 7,950 | 678k |
| ≥5 | 1,281 | 5,153 | 369 | 6,803 | 358k |
| **≥10** | 1,169 | 3,503 | 369 | **5,041** | **136k** |

At ≥10 synapses, one tick of this circuit is a single 136k-nonzero sparse mat-vec, about
0.1 ms, which is negligible next to physics. The full MANC is not simulated.

### How MANC enters the architecture (Milestone 6)

```
obs[12] ──(learned encoder = "brain", unconstrained: MANC has no brain)──▶ DN layer (MANC DNs, by type×side)
DN layer ──▶ MANC VNC subgraph: leaky rate units, W_ij = sign(NT_pre) · log1p(syn_ij)/norm    [wiring fixed]
             learnable: per-(pre-class × NT) gains, per-neuron bias                            [magnitudes learned]
leg-MN pools (6 legs × side) ──(fixed anatomical readout: Σ left-leg MNs, Σ right-leg MNs; learnable gain/bias)──▶ descending[2]
```

Stated assumption: FlyGym's CPG plus reflex rules stand in for the VNC's rhythm generator,
which MANC's wiring alone cannot provide without cellular dynamics. The MANC circuit
supplies the **tonic, side-specific drive** instead: leg-MN pool recruitment on each side
sets that side's stepping amplitude. Sparsity and signs come from the connectome.
Magnitudes are learned. A shuffled-connectome control (same degree distribution, wiring
permuted) will test whether the MANC structure matters.

Attribution: MANC data © Janelia FlyEM, CC-BY 4.0. Cite Takemura et al. 2024 (eLife
RP97769), Marin et al. 2024 (eLife RP97766) and Cheong et al. 2024 (eLife RP96084). We
redistribute no connectome data: `data/` is gitignored and fetched from the official
bucket.

## 6. Learning

- `brain/model.py`: `MLPPolicy` (baseline: 12 → 32 → 32 → 2, tanh) and, at M6, `MANCPolicy`. Both expose a flat parameter vector.
- `brain/trainer.py`: antithetic **evolution strategies** (rank-shaped fitness, Adam) in pure numpy, with episodes spread over CPU cores through `multiprocessing`. We chose ES because it needs no gradients through the recurrent or sparse connectome dynamics, it gives the baseline and MANC policies the same optimiser (a fair comparison), it parallelises across the 12 cores, and it adds no torch dependency.
- Budget: 3 s episodes (the reflex needs 0.74 s on average). 32 episodes per generation ≈ 40–80 s on 11 workers, and failing (full-length) episodes dominate.

## 7. Minimum diff for a first working prototype

**Zero lines changed under `simulator/`.** New files:

```
environment/reach_world.py      ReachTargetWorld(FlatGroundWorld): target + obstacles + contact pairs
environment/fly_interface.py    FlyInterface: build fly/world/sim/controller, get_observation,
                                apply_action (→ descending[2]), step (= one brain tick), reset, render
brain/model.py  state.py        MLPPolicy (+ MANCPolicy at M6); recurrent state container
brain/trainer.py                ES trainer
brain/interface.py              Brain protocol: reset() / act(obs) -> action; params get/set
brain/connectome.py             MANC loader → sparse DN→IN→MN subgraph (M6)
experiments/train.py            CLI: --headless, --config
experiments/evaluate.py         CLI: --visual (mp4 via Renderer, or live viewer)
configs/reach_target.yaml       episodes, episode_length, learning_rate, reward_scale, env_seed, ...
```

The prototype's upstream calls are exactly these, all verified in M1:
`make_locomotion_fly`, `FlatGroundWorld.add_fly`, `Simulation(world)`, `sim.reset/warmup/step`,
`HybridControllerObservation.from_sim`, `HybridTurningController.step`,
`apply_locomotion_action`, `sim.get_body_positions/rotations`, `sim.set_renderer`.

## 8. Milestones

| # | Milestone | Verification |
|---|---|---|
| 1 | Run existing sim as-is | ✅ upstream tests 76/76. `m1_upstream_turning_demo.py` walks and turns, mp4 saved. |
| 2 | Controllable 3D env | ✅ target and obstacles repositioned per episode (mocap). Walking into an obstacle gives real contact (`obstacle_0_geom`↔`rf_tarsus5`) and deflects the fly. Top-down and follow cameras render. |
| 3 | Sensory in / motor out | ✅ `tests/test_interface.py` 5/5: obs geometry, forward progress with positive reward, both turn signs, collision and occlusion. |
| 4 | Baseline controller inserted | ✅ `MLPPolicy` drives `FlyInterface` end to end inside the ES pool. References (calibration run, 4 s episodes): `SteeringReflex` 22/22 successes (0.74 s), random 3/22. The 3 s re-runs are in EXPERIMENTS.md. |
| 5 | Baseline reaches target | ES-trained MLP beats the random policy and reports its success rate |
| 6 | Swap in MANC architecture | `MANCPolicy` behind the same `Brain` interface |
| 7 | Training loop | the same trainer and configs work for both |
| 8 | Quantitative results | per-episode CSV/JSON in `experiments/results/`, written up in `EXPERIMENTS.md` |
| 9 | Harder/unfamiliar envs | unseen obstacle layouts, distances and seeds. Optionally ommatidia input. |

— Henry Cao, 2026-09-19. Project ID: henrycao-2026-f003b4
