# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""MANC v1.0 -> a fixed, signed, sparse DN -> interneuron -> leg-motor-neuron circuit.

Data: Janelia FlyEM MANC v1.0 (CC-BY 4.0), fetched by scripts/fetch_manc.sh into
data/manc_v1.0/. Takemura et al. 2024, eLife RP97769; Marin et al. 2024, eLife
RP97766; Cheong et al. 2024, eLife RP96084.

Node set (traced neurons only):
  DN  descending neurons that project onto the IN/MN set
  IN  interneurons that are both a DN target and a leg-MN input
  MN  leg motor neurons (subclass fl / ml / hl = front / middle / hind leg)
using connections with at least ``min_weight`` synapses.

Edges: w_ij = sign(NT of pre j) * syn_ij / sum_k syn_ik (fraction of neuron i's
in-circuit input), with ACh -> +1 and GABA, glutamate -> -1 (the usual assumption for
the fly CNS, where glutamate mostly acts through inhibitory GluCl). Edges are split
into groups by (presynaptic class, NT) so that a model can learn one gain per group
while the wiring and signs stay fixed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "manc_v1.0"
CLASSES = ("DN", "IN", "MN")
NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}
NTS = ("acetylcholine", "gaba", "glutamate")
LEG_OF_SUBCLASS = {"fl": "f", "ml": "m", "hl": "h"}
# DNs with known locomotor roles; always offered as input channels if present.
KNOWN_LOCOMOTOR_DNS = ("DNa01", "DNa02", "DNp09", "MDN", "DNg13", "DNb05", "DNb06")


@dataclass
class VNCCircuit:
    n: int
    node_class: np.ndarray  # (n,) 0=DN 1=IN 2=MN
    body_ids: np.ndarray  # (n,)
    side: np.ndarray  # (n,) 0=left 1=right -1=unknown/midline
    group_mats: list  # [csr (n,n)] one per (pre class, NT) edge group, rows = post
    group_names: list[str]
    input_types: list[str]  # DN types used as input channels
    input_nodes_left: list[np.ndarray]  # per input type: node indices on the left
    input_nodes_right: list[np.ndarray]
    mn_left: np.ndarray  # node indices of left-leg MNs
    mn_right: np.ndarray
    mn_leg: np.ndarray  # (n,) leg letter per node ('' for non-MN)
    meta: dict

    def weights(self, gains: np.ndarray) -> sp.csr_matrix:
        """Effective signed weight matrix sum_g gains[g] * W_g."""
        W = self.group_mats[0] * gains[0]
        for g, M in zip(gains[1:], self.group_mats[1:]):
            W = W + M * g
        return W.tocsr()


def load_manc_vnc(
    data_dir: Path = DATA_DIR,
    *,
    min_weight: int = 10,
    n_input_types: int = 32,
    shuffle_seed: int | None = None,
    shuffle_mode: str = "class",
    use_cache: bool = True,
) -> VNCCircuit:
    """Build (or load cached) `VNCCircuit`.

    Args:
        min_weight: Minimum synapse count for an edge.
        n_input_types: Number of DN types exposed as input channels, ranked by their
            total in-circuit output onto IN and MN (known locomotor DNs always included).
        shuffle_seed: If set, build the *shuffled-connectome control*: every edge keeps
            its presynaptic neuron, weight and sign, but its postsynaptic partner is
            replaced by a random neuron of the same class. This keeps class-to-class
            edge counts, out-degrees and the weight distribution but destroys the specific
            wiring (including laterality).
        shuffle_mode: ``"class"`` (as above) or ``"class_side"``: the new partner also
            has the same side (left/right) as the original one, so laterality survives
            and only the specific wiring within each side is destroyed.
    """
    tag = f"w{min_weight}_t{n_input_types}" + (f"_shuf-{shuffle_mode}{shuffle_seed}" if shuffle_seed is not None else "")
    cache = Path(data_dir) / f"vnc_circuit_{tag}.npz"
    if use_cache and cache.exists():
        return _from_npz(cache)
    circuit = _build(Path(data_dir), min_weight, n_input_types, shuffle_seed, shuffle_mode)
    if use_cache:
        _to_npz(circuit, cache)
    return circuit


def _build(
    data_dir: Path, min_weight: int, n_input_types: int, shuffle_seed: int | None, shuffle_mode: str
) -> VNCCircuit:
    import pandas as pd

    props = pd.read_feather(data_dir / "manc-v1.0-neuron-properties.feather")
    traced = pd.read_csv(data_dir / "traced-neurons.csv")
    conns = pd.read_csv(data_dir / "traced-connections.csv")
    props = props[props.bodyId.isin(traced.bodyId)].set_index("bodyId")
    conns = conns[conns.weight >= min_weight]

    is_dn = props["class"] == "descending neuron"
    is_legmn = (props["class"] == "motor neuron") & props.subclass.isin(list(LEG_OF_SUBCLASS))
    dns, mns = set(props.index[is_dn]), set(props.index[is_legmn])

    dn_targets = set(conns.bodyId_post[conns.bodyId_pre.isin(dns)])
    mn_inputs = set(conns.bodyId_pre[conns.bodyId_post.isin(mns)])
    ins = (dn_targets & mn_inputs) - dns - mns
    dn_used = set(conns.bodyId_pre[conns.bodyId_pre.isin(dns) & conns.bodyId_post.isin(ins | mns)])

    ids = np.array(sorted(dn_used) + sorted(ins) + sorted(mns))
    node_class = np.array([0] * len(dn_used) + [1] * len(ins) + [2] * len(mns), dtype=np.int8)
    index = {b: i for i, b in enumerate(ids)}
    n = len(ids)
    sub = props.loc[ids]
    side_str = sub.somaSide.fillna(sub.rootSide).fillna("")
    side = np.where(side_str == "LHS", 0, np.where(side_str == "RHS", 1, -1)).astype(np.int8)
    nt = sub.predictedNt.fillna("unknown").to_numpy()

    e = conns[conns.bodyId_pre.isin(index) & conns.bodyId_post.isin(index)]
    pre = e.bodyId_pre.map(index).to_numpy()
    post = e.bodyId_post.map(index).to_numpy()
    w = e.weight.to_numpy(dtype=float)

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        post = post.copy()
        if shuffle_mode not in ("class", "class_side"):
            raise ValueError(f"Unknown shuffle_mode '{shuffle_mode}'")
        sides = (-1, 0, 1) if shuffle_mode == "class_side" else (None,)
        orig_post = post.copy()
        for c in range(3):
            for sd in sides:
                in_group = (node_class == c) if sd is None else (node_class == c) & (side == sd)
                members = np.flatnonzero(in_group)
                sel = in_group[orig_post]
                if sel.any():
                    post[sel] = rng.choice(members, size=sel.sum())
        # Merge duplicate (pre, post) pairs created by the shuffle and drop self-loops.
        keep = pre != post
        pre, post, w = pre[keep], post[keep], w[keep]

    in_total = np.bincount(post, weights=w, minlength=n)
    frac = w / np.maximum(in_total[post], 1e-9)
    sign = np.array([NT_SIGN.get(t, 1.0) for t in nt])[pre]  # unknown NT -> excitatory

    group_mats, group_names = [], []
    pre_nt = nt[pre]
    for c, cname in enumerate(CLASSES):
        for t in NTS:
            sel = (node_class[pre] == c) & (pre_nt == t if t != "acetylcholine" else np.isin(pre_nt, ["acetylcholine", "unknown"]))
            if not sel.any():
                continue
            M = sp.csr_matrix((frac[sel] * sign[sel], (post[sel], pre[sel])), shape=(n, n))
            group_mats.append(M)
            group_names.append(f"{cname}:{t}")

    # Input channels: DN types ranked by total in-circuit output.
    dn_idx = np.flatnonzero(node_class == 0)
    out_w = np.bincount(pre, weights=w, minlength=n)
    dn_types = sub.type.fillna("untyped").to_numpy()
    type_rank = (
        pd.DataFrame({"type": dn_types[dn_idx], "out": out_w[dn_idx]}).groupby("type").out.sum().sort_values(ascending=False)
    )
    chosen = [t for t in KNOWN_LOCOMOTOR_DNS if t in type_rank.index]
    for t in type_rank.index:
        if len(chosen) >= n_input_types:
            break
        if t not in chosen and t != "untyped":
            chosen.append(t)
    left_nodes = [dn_idx[(dn_types[dn_idx] == t) & (side[dn_idx] == 0)] for t in chosen]
    right_nodes = [dn_idx[(dn_types[dn_idx] == t) & (side[dn_idx] == 1)] for t in chosen]

    mn_idx = np.flatnonzero(node_class == 2)
    mn_leg = np.array([""] * n, dtype="<U1")
    mn_leg[mn_idx] = [LEG_OF_SUBCLASS[s] for s in sub.subclass.to_numpy()[mn_idx]]

    meta = {
        "dataset": "MANC v1.0 (Janelia FlyEM, CC-BY 4.0)",
        "min_weight": min_weight,
        "shuffle_seed": shuffle_seed,
        "shuffle_mode": shuffle_mode if shuffle_seed is not None else None,
        "n_nodes": int(n),
        "n_edges": int(len(pre)),
        "n_by_class": {cname: int((node_class == c).sum()) for c, cname in enumerate(CLASSES)},
        "input_types": chosen,
    }
    return VNCCircuit(
        n=n,
        node_class=node_class,
        body_ids=ids,
        side=side,
        group_mats=group_mats,
        group_names=group_names,
        input_types=chosen,
        input_nodes_left=left_nodes,
        input_nodes_right=right_nodes,
        mn_left=mn_idx[side[mn_idx] == 0],
        mn_right=mn_idx[side[mn_idx] == 1],
        mn_leg=mn_leg,
        meta=meta,
    )


def _to_npz(c: VNCCircuit, path: Path) -> None:
    import json

    arrays = {
        "node_class": c.node_class, "body_ids": c.body_ids, "side": c.side,
        "mn_left": c.mn_left, "mn_right": c.mn_right, "mn_leg": c.mn_leg,
    }
    for i, M in enumerate(c.group_mats):
        M = M.tocoo()
        arrays[f"g{i}_row"], arrays[f"g{i}_col"], arrays[f"g{i}_val"] = M.row, M.col, M.data
    for i, (l, r) in enumerate(zip(c.input_nodes_left, c.input_nodes_right)):
        arrays[f"in{i}_l"], arrays[f"in{i}_r"] = l, r
    meta = c.meta | {"group_names": c.group_names, "n": c.n}
    # Write-then-rename so concurrent workers never see (or produce) a partial file.
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    os.replace(tmp, path)


def _from_npz(path: Path) -> VNCCircuit:
    import json

    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = meta.pop("n")
    names = meta.pop("group_names")
    mats = [sp.csr_matrix((z[f"g{i}_val"], (z[f"g{i}_row"], z[f"g{i}_col"])), shape=(n, n)) for i in range(len(names))]
    k = len(meta["input_types"])
    return VNCCircuit(
        n=n, node_class=z["node_class"], body_ids=z["body_ids"], side=z["side"],
        group_mats=mats, group_names=names, input_types=meta["input_types"],
        input_nodes_left=[z[f"in{i}_l"] for i in range(k)], input_nodes_right=[z[f"in{i}_r"] for i in range(k)],
        mn_left=z["mn_left"], mn_right=z["mn_right"], mn_leg=z["mn_leg"], meta=meta,
    )
