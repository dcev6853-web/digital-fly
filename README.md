# Digital Fly: a learning brain between FlyGym's senses and motors

This project puts a trainable brain into an unmodified NeuroMechFly v2 simulation
(FlyGym). The brain reads task observations and outputs the 2-D **descending
signal** that FlyGym's own locomotion controller (CPG, recorded steps, reflex
rules) turns into leg joint commands for the MuJoCo body.

```
FlyGym sim ─▶ FlyInterface.get_observation() ─▶ brain ─▶ descending[2] ─▶ FlyGym controller ─▶ MuJoCo body
```

The experimental brain (`MANCPolicy`) runs its motor pathway through the **male adult
nerve cord connectome (MANC v1.0)**. A learned encoder drives MANC descending neurons,
activity propagates through the real DN → interneuron → leg-motor-neuron wiring (fixed
sparsity and signs, learned gains), and left/right motor pools set the left/right drive.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the full map, the insertion point, and the
dataset choice (MANC vs MaleCNS vs FlyWire).

## Setup (CPU only, tested on aarch64 Linux / WSL2)

```bash
git clone https://github.com/NeLy-EPFL/flygym.git simulator
git -C simulator checkout 38c8ec61034cd59bc5ba0de20688d4a3c0000d60   # FlyGym v2.1.0+7
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/fetch_manc.sh                  # ~93 MB public MANC v1.0 tables, checksummed
export MUJOCO_GL=egl                   # headless rendering
```

## Run

```bash
.venv/bin/python experiments/train.py --headless                                   # MLP baseline
.venv/bin/python experiments/train.py --headless --config configs/reach_target_manc.yaml
.venv/bin/python experiments/evaluate.py --run experiments/results/<run>           # held-out test episodes
.venv/bin/python experiments/evaluate.py --run experiments/results/<run> --visual  # + mp4 (top-down and follow cams)
.venv/bin/python experiments/evaluate.py --run experiments/results/<run> --live    # MuJoCo viewer (needs a display)
.venv/bin/python experiments/evaluate.py --brain reflex                            # hand-coded reference
.venv/bin/python -m pytest tests                                                   # our tests
```

Every run writes to `experiments/results/<run>/`:

- `train_episodes.csv`: every training episode.
- `eval_episodes.csv`: held-out seeds, evaluated during training.
- `generations.csv`
- `theta_best.npy`
- `summary.json`

`evaluate.py` writes `episodes.csv` and `summary.json`, with success rate,
steps and time to target, distance traveled, collisions and reward per episode. Results
and their interpretation are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Layout

```
simulator/      FlyGym, cloned and untouched (git -C simulator status is clean)
environment/    reach_world.py (target, obstacles, contact pairs), fly_interface.py (the adapter),
                fast_controller.py (bit-identical cached controller path)
brain/          model.py (MLP, MANC, reference policies), connectome.py (MANC loader),
                state.py, interface.py (Brain protocol + episode loop), trainer.py (ES)
experiments/    train.py, evaluate.py, m1_upstream_turning_demo.py, results/
configs/        reach_target*.yaml
tests/          interface, fast-controller equivalence, brain tests
data/           MANC tables, fetched and not redistributed (gitignored)
```

## Attribution and licences

- **FlyGym / NeuroMechFly v2**: © NeLy lab, EPFL. Apache-2.0 (`simulator/LICENSE`).
  Wang-Chen et al., *Nature Methods* 2024. The body model, meshes, recorded step data,
  CPG, hybrid controller and renderer are all theirs. This project only adds adapters
  around them.
- **MANC v1.0 connectome**: Janelia FlyEM, CC-BY 4.0. Takemura et al. 2024 (eLife
  RP97769), Marin et al. 2024 (eLife RP97766), Cheong et al. 2024 (eLife RP96084).
  It is downloaded from the official bucket by `scripts/fetch_manc.sh` and not redistributed here.
- Our additions: the `environment/`, `brain/`, `experiments/`, `configs/` and `tests/` code.

— Henry Cao, 2026-09-19. Project ID: henrycao-2026-f003b4
