# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Brains. Each maps obs (12,) -> (forward, turn) in [-1, 1]^2.

- `MLPPolicy`: the Milestone 4 baseline, a small feed-forward network.
- `SteeringReflex`: a hand-written reference controller (not learned). It checks that
  the task is solvable within the episode budget and gives an upper reference for
  the learned brains.
- `RandomPolicy`: lower reference.
- The MANC-constrained architecture arrives at Milestone 6 behind the same interface.
"""

from __future__ import annotations

import numpy as np

from environment.fly_interface import OBS_DIM, OBS_NAMES

_IDX = {name: i for i, name in enumerate(OBS_NAMES)}


class MLPPolicy:
    """obs -> tanh(hidden) -> tanh(2). Parameters are one flat vector."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 16, act_dim: int = 2, seed: int = 0):
        self.shapes = [(hidden, obs_dim), (hidden,), (act_dim, hidden), (act_dim,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.n_params = sum(self.sizes)
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(obs_dim), (hidden, obs_dim))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1 / np.sqrt(hidden), (act_dim, hidden))
        self.b2 = np.zeros(act_dim)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ obs + self.b1)
        return np.tanh(self.W2 @ h + self.b2)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_params(self, flat: np.ndarray) -> None:
        parts = np.split(np.asarray(flat, dtype=float), np.cumsum(self.sizes)[:-1])
        self.W1, self.b1, self.W2, self.b2 = (p.reshape(s) for p, s in zip(parts, self.shapes))


class SteeringReflex:
    """Hand-coded reference: steer toward the target bearing, slow down to turn."""

    n_params = 0

    def __init__(self, gain: float = 1.5, forward: float = 1.0):
        self.gain, self.forward = gain, forward

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        cos_b, sin_b = obs[_IDX["target_dir_cos"]], obs[_IDX["target_dir_sin"]]
        bearing = np.arctan2(sin_b, cos_b)  # > 0 means target is to the left
        turn = np.clip(-self.gain * bearing, -1, 1)  # turn > 0 is clockwise
        forward = self.forward * max(0.3, cos_b)
        return np.array([forward, turn])

    def get_params(self) -> np.ndarray:
        return np.zeros(0)

    def set_params(self, flat: np.ndarray) -> None:
        pass


class RandomPolicy:
    n_params = 0

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1, 1, size=2)

    def get_params(self) -> np.ndarray:
        return np.zeros(0)

    def set_params(self, flat: np.ndarray) -> None:
        pass


# Observation entries that flip sign under left-right mirroring (see OBS_NAMES).
_MIRROR = np.array([1 if n not in ("target_dir_sin", "obstacle_dir_sin", "velocity_lateral", "yaw_rate", "orientation_sin") else -1
                    for n in OBS_NAMES], dtype=float)


class MANCPolicy:
    """Connectome-constrained brain: learned encoder -> MANC VNC (fixed wiring) -> readout.

    1. Encoder ("brain"; MANC has no brain, so this part is unconstrained): each of the
       T input DN types gets ``u_t = W_t . obs + b_t`` on its left-side neurons and
       ``W_t . mirror(obs) + b_t`` on its right-side neurons. The shared weights make the
       encoder exactly bilaterally symmetric.
    2. VNC: leaky rate units on the MANC DN/IN/MN subgraph (`brain.connectome`),
       ``r <- (1-a) r + a * tanh(relu(W r + b_class + u))``, ``substeps`` updates per brain
       tick. W's sparsity and signs come from MANC; only one gain per (presynaptic class,
       neurotransmitter) group is learned, plus one bias per class.
    3. Readout: the mean rate of left-leg and of right-leg MN pools gives the side drives
       (FlyGym's descending signal, left and right) through a fixed anatomical grouping.
       ``readout_mode="raw"`` (legacy, used by the failed `manc` run):
       ``D = tanh(g (pool - theta))`` with learnable g, theta. The threshold lives on the
       pool scale (~1e-3), so ES noise of 0.1 saturates it (see EXPERIMENTS.md).
       ``readout_mode="standardized"``: ``D = tanh(a * (pool - mu) / sd + b)``. The pool
       statistics mu and sd are fixed constants from a canonical calibration, and only
       log(a) and b (both O(1)) are learned. There are no class biases in this mode.

    Since the encoder is symmetric, any left/right difference in drive, and hence any
    steering, must come through the lateral structure of the MANC wiring.
    """

    def __init__(self, circuit_kwargs: dict | None = None, substeps: int = 4, leak: float = 0.5, seed: int = 0,
                 readout_mode: str = "raw"):
        from brain.connectome import load_manc_vnc
        from brain.state import BrainState

        self.circuit = c = load_manc_vnc(**(circuit_kwargs or {}))
        self.substeps, self.leak = substeps, leak
        if readout_mode not in ("raw", "standardized"):
            raise ValueError(f"Unknown readout_mode '{readout_mode}'")
        self.readout_mode = readout_mode
        self.T = len(c.input_types)
        self.n_groups = len(c.group_mats)
        n_class_bias = 3 if readout_mode == "raw" else 0
        self.shapes = [(self.T, OBS_DIM), (self.T,), (self.n_groups,), (n_class_bias,), (2,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.n_params = sum(self.sizes)
        self.state = BrainState(rates=c.n)

        rng = np.random.default_rng(seed)
        self.enc_W = rng.normal(0, 0.5, (self.T, OBS_DIM))
        self.enc_b = np.zeros(self.T)
        self.log_gains = np.zeros(self.n_groups)
        self.class_bias = np.zeros(n_class_bias)
        self.readout = np.zeros(2)  # raw: (log gain, theta); standardized: (log a, b)
        self._pool_mu, self._pool_sd = 0.0, 1.0  # standardized mode only (fixed, not learned)
        # Map each input type to its node indices once.
        self._left = [np.asarray(ix, dtype=int) for ix in c.input_nodes_left]
        self._right = [np.asarray(ix, dtype=int) for ix in c.input_nodes_right]
        self._bias_vec = np.zeros(c.n)
        self._W = None
        self._rebuild()
        if readout_mode == "raw":
            self._calibrate_readout(rng)
        else:
            self._calibrate_standardized()

    # -- parameters
    def get_params(self) -> np.ndarray:
        return np.concatenate([self.enc_W.ravel(), self.enc_b, self.log_gains, self.class_bias, self.readout])

    def set_params(self, flat: np.ndarray) -> None:
        parts = np.split(np.asarray(flat, dtype=float), np.cumsum(self.sizes)[:-1])
        self.enc_W, self.enc_b, self.log_gains, self.class_bias, self.readout = (
            p.reshape(s) for p, s in zip(parts, self.shapes)
        )
        self._rebuild()

    def _rebuild(self) -> None:
        self._W = self.circuit.weights(np.exp(self.log_gains))
        self._bias_vec = self.class_bias[self.circuit.node_class] if len(self.class_bias) else np.zeros(self.circuit.n)

    # -- dynamics
    def reset(self) -> None:
        self.state.reset()

    def _dn_input(self, obs: np.ndarray) -> np.ndarray:
        u = np.zeros(self.circuit.n)
        left = self.enc_W @ obs + self.enc_b
        right = self.enc_W @ (obs * _MIRROR) + self.enc_b
        for t in range(self.T):
            u[self._left[t]] = left[t]
            u[self._right[t]] = right[t]
        return u

    def _pools(self, obs: np.ndarray) -> np.ndarray:
        u = self._dn_input(obs) + self._bias_vec
        r = self.state.rates
        for _ in range(self.substeps):
            r = (1 - self.leak) * r + self.leak * np.tanh(np.maximum(self._W @ r + u, 0.0))
        self.state.rates = r
        c = self.circuit
        return np.array([r[c.mn_left].mean(), r[c.mn_right].mean()])

    def act(self, obs: np.ndarray) -> np.ndarray:
        pools = self._pools(np.asarray(obs, dtype=float))
        if self.readout_mode == "raw":
            drive = np.tanh(np.exp(self.readout[0]) * (pools - self.readout[1]))  # (left, right) in [-1, 1]
        else:
            drive = np.tanh(np.exp(self.readout[0]) * (pools - self._pool_mu) / self._pool_sd + self.readout[1])
        return np.array([(drive[0] + drive[1]) / 2, (drive[0] - drive[1]) / 2])

    def _calibrate_readout(self, rng) -> None:
        """Standardise the readout at init: gain = 1/std(pool), threshold at mean - 0.5 std.

        Uses random observations and a few ticks of dynamics, so the untrained policy
        walks forward at moderate drive and pool fluctuations map to O(1) drive changes.
        """
        samples = []
        for _ in range(8):
            self.reset()
            for _ in range(10):
                obs = np.concatenate([[rng.uniform(0.1, 1.0)], _unit(rng), [1.0], [rng.uniform(0.1, 1.0)], _unit(rng),
                                      rng.normal(0, 0.3, 3), _unit(rng)])
                samples.append(self._pools(obs))
        s = np.array(samples).ravel()
        std = max(float(s.std()), 1e-6)
        self.readout = np.array([np.log(1.0 / std), float(s.mean()) - 0.5 * std])
        self.reset()

    def _calibrate_standardized(self, calibration_seed: int = 12345) -> None:
        """Fix mu and sd of the pool activity from a canonical calibration.

        The calibration uses its own fixed seed and a canonical random encoder, so every
        process (trainer and each worker) computes identical constants whatever the
        policy's init seed. The learned readout starts at log a = 0 and b = 0.5, so the
        untrained policy walks forward at moderate drive (tanh 0.5 = 0.46).
        """
        own = (self.enc_W, self.enc_b)
        rng = np.random.default_rng(calibration_seed)
        self.enc_W, self.enc_b = rng.normal(0, 0.5, self.enc_W.shape), np.zeros(self.T)
        samples = []
        for _ in range(8):
            self.reset()
            for _ in range(10):
                obs = np.concatenate([[rng.uniform(0.1, 1.0)], _unit(rng), [1.0], [rng.uniform(0.1, 1.0)], _unit(rng),
                                      rng.normal(0, 0.3, 3), _unit(rng)])
                samples.append(self._pools(obs))
        s = np.array(samples).ravel()
        self._pool_mu, self._pool_sd = float(s.mean()), max(float(s.std()), 1e-9)
        self.enc_W, self.enc_b = own
        self.readout = np.array([0.0, 0.5])
        self.reset()


def _unit(rng) -> np.ndarray:
    a = rng.uniform(-np.pi, np.pi)
    return np.array([np.cos(a), np.sin(a)])


def make_brain(spec: dict, seed: int = 0):
    """Build a brain from a config dict like ``{"type": "mlp", "hidden": 16}``."""
    kind = spec.get("type", "mlp")
    if kind == "mlp":
        return MLPPolicy(hidden=int(spec.get("hidden", 16)), seed=seed)
    if kind == "manc":
        circuit_kwargs = {k: spec[k] for k in ("min_weight", "n_input_types", "shuffle_seed", "shuffle_mode") if k in spec}
        return MANCPolicy(circuit_kwargs, substeps=int(spec.get("substeps", 4)), leak=float(spec.get("leak", 0.5)), seed=seed,
                          readout_mode=spec.get("readout", "raw"))
    if kind == "reflex":
        return SteeringReflex(**{k: v for k, v in spec.items() if k != "type"})
    if kind == "random":
        return RandomPolicy(seed=seed)
    raise ValueError(f"Unknown brain type '{kind}'")
