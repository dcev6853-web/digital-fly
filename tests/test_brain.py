"""Brain components (no physics): parameter plumbing, MANC circuit, encoder symmetry."""

import numpy as np
import pytest

from brain.connectome import DATA_DIR, load_manc_vnc
from brain.model import _MIRROR, MANCPolicy, MLPPolicy, make_brain

needs_manc = pytest.mark.skipif(not (DATA_DIR / "traced-connections.csv").exists(), reason="run scripts/fetch_manc.sh")


def test_mlp_param_roundtrip():
    b = MLPPolicy(seed=1)
    theta = np.random.default_rng(0).normal(size=b.n_params)
    b.set_params(theta)
    assert np.array_equal(b.get_params(), theta)
    a = b.act(np.zeros(12))
    assert a.shape == (2,) and np.all(np.abs(a) <= 1)


@needs_manc
def test_manc_circuit_matches_manc():
    c = load_manc_vnc(min_weight=10, n_input_types=32)
    assert c.meta["n_by_class"] == {"DN": 1169, "IN": 3503, "MN": 369}
    assert c.meta["n_edges"] == 135717
    assert len(c.mn_left) + len(c.mn_right) == 369
    for t in ("DNa01", "DNa02", "DNp09", "MDN"):
        assert t in c.input_types
    # Each row of |W| sums to 1 (fraction of in-circuit input) for neurons with inputs.
    W = c.weights(np.ones(len(c.group_mats)))
    row = np.asarray(abs(W).sum(axis=1)).ravel()
    assert np.allclose(row[row > 0], 1.0)


@needs_manc
@pytest.mark.parametrize("mode", ["class", "class_side"])
def test_shuffle_preserves_class_structure(mode):
    real = load_manc_vnc()
    shuf = load_manc_vnc(shuffle_seed=0, shuffle_mode=mode)
    assert abs(shuf.meta["n_edges"] - real.meta["n_edges"]) < 0.01 * real.meta["n_edges"]  # self-loops dropped
    Wr = real.weights(np.ones(len(real.group_mats)))
    Ws = shuf.weights(np.ones(len(shuf.group_mats)))
    assert (Wr != Ws).nnz > 0.9 * Wr.nnz


@needs_manc
def test_manc_policy_roundtrip_and_symmetric_encoder():
    b = make_brain({"type": "manc"})
    assert isinstance(b, MANCPolicy)
    theta = b.get_params() + 0.01
    b.set_params(theta)
    assert np.array_equal(b.get_params(), theta)
    obs = np.random.default_rng(0).normal(size=12)
    u, u_m = b._dn_input(obs), b._dn_input(obs * _MIRROR)
    for L, R in zip(b._left, b._right):
        # left neurons under obs == right neurons under mirror(obs) (same type)
        assert np.allclose(u[L][:1], u_m[R][:1])
    b.reset()
    a = b.act(obs)
    assert a.shape == (2,) and np.isfinite(a).all()
    assert np.abs(b.state.rates).sum() > 0
    b.reset()
    assert np.abs(b.state.rates).sum() == 0


def test_centered_ranks_tie_aware_and_backward_compatible():
    from brain.trainer import centered_ranks

    x = np.random.default_rng(0).normal(size=32)  # no ties
    old = np.empty(32)
    old[np.argsort(x)] = np.arange(32)
    assert np.array_equal(centered_ranks(x), old / 31 - 0.5)
    assert np.array_equal(centered_ranks(np.full(16, -21.0)), np.zeros(16))  # degenerate -> no signal


@needs_manc
def test_manc_raw_readout_is_the_legacy_behaviour():
    b = make_brain({"type": "manc"})  # default readout = raw (the failed `manc` run)
    assert b.n_params == 430
    assert np.allclose(b.readout, [7.2218, 0.0016], atol=1e-3)


@needs_manc
def test_manc_standardized_readout_survives_es_noise():
    """Regression test for the `manc` failure: with ES noise sigma=0.1 on every parameter,
    the standardized readout must still respond to observations (not pinned at +-1)."""
    b = make_brain({"type": "manc", "readout": "standardized"})
    assert b.n_params == 427
    b2 = make_brain({"type": "manc", "readout": "standardized"}, seed=3)
    assert (b._pool_mu, b._pool_sd) == (b2._pool_mu, b2._pool_sd)  # identical in every process
    rng = np.random.default_rng(1)
    obs_set = [np.concatenate([[rng.uniform(0.2, 1)], [np.cos(a := rng.uniform(-3, 3)), np.sin(a)], [1, 3, 0, 0],
                               rng.normal(0, 0.3, 3), [1, 0]]) for _ in range(12)]
    theta0 = b.get_params()
    for k in range(6):
        b.set_params(theta0 + 0.1 * rng.standard_normal(theta0.size))
        acts = []
        for obs in obs_set:
            b.reset()
            for _ in range(3):
                a = b.act(obs)
            acts.append(a)
        acts = np.array(acts)
        assert acts[:, 1].std() > 1e-3, "turn command must depend on the observation"
        assert np.mean(np.abs(acts) > 0.999) < 0.5, "readout must not be saturated"
