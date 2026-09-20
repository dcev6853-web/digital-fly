"""Cached, bit-identical drop-in for FlyGym's HybridTurningController + observation.

Why: profiling (ARCHITECTURE.md §2) shows that ~75% of each 0.1 ms physics step goes to
Python plumbing in the upstream controller. Every step it builds 42 `JointDOF`
dataclasses, hashes them into a dict and re-reads them in output order, and it finds
body segments with `list.index` on dataclasses. The math (CPG integration, recorded
step splines, correction rules, adhesion) is cheap.

What: a subclass that precomputes those index maps once and then runs the *same*
upstream math in the same floating-point order. The simulator is not modified. The
upstream methods are still called for the CPG (`cpg_network.step`), the step splines
(`preprogrammed_steps.get_joint_angles`), the retraction/stumbling rules and adhesion.
Only the dict round-trip is replaced by an index scatter.

Guarantee: tests/test_fast_controller.py checks that actions and the resulting MuJoCo
state are bit-identical to the upstream path over whole episodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flygym_demo.complex_terrain import HybridControllerObservation, HybridTurningController, LocomotionAction
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof, get_default_locomotion_dof_order
from flygym_demo.complex_terrain.hybrid_controller import (
    _CORRECTION_VECTORS,
    _DETECTED_STUMBLING_LINKS,
    _RIGHT_LEG_CORRECTION_SIGN,
    _step_phase_gain,
)


@dataclass
class CachedHybridTurningController(HybridTurningController):
    def __post_init__(self) -> None:
        super().__post_init__()
        order = self.output_dof_order if self.output_dof_order is not None else get_default_locomotion_dof_order()
        position = {dof: i for i, dof in enumerate(order)}
        self._n_out = len(order)
        self._out_idx = np.array(
            [[position[dof_spec_to_jointdof(leg, spec)] for spec in self.preprogrammed_steps.dofs_per_leg] for leg in self.legs]
        )
        if sorted(self._out_idx.ravel().tolist()) != list(range(self._n_out)):
            raise ValueError("output_dof_order must contain each controlled leg DoF exactly once")
        self._corr_vecs = []
        for leg in self.legs:
            v = _CORRECTION_VECTORS[leg[1]]
            if leg.startswith("r"):
                v = v * _RIGHT_LEG_CORRECTION_SIGN
            self._corr_vecs.append(v)

    def step(self, descending_signal: np.ndarray, obs: HybridControllerObservation) -> LocomotionAction:
        # --- identical to HybridTurningController.step
        descending_signal = np.asarray(descending_signal, dtype=float)
        if descending_signal.shape != (2,):
            raise ValueError("descending_signal must have shape (2,).")
        self.cpg_network.intrinsic_amps = np.repeat(np.abs(descending_signal[:, np.newaxis]), 3, axis=1).ravel()
        intrinsic_freqs = self._base_intrinsic_freqs.copy()
        intrinsic_freqs[:3] *= 1 if descending_signal[0] >= 0 else -1
        intrinsic_freqs[3:] *= 1 if descending_signal[1] >= 0 else -1
        self.cpg_network.intrinsic_freqs = intrinsic_freqs
        return self._hybrid_step(obs)

    def _hybrid_step(self, obs: HybridControllerObservation) -> LocomotionAction:
        # --- identical to HybridController.step except the output assembly
        leg_to_correct_retraction = self._select_retraction_leg(obs)
        if leg_to_correct_retraction is not None:
            if self.retraction_correction[leg_to_correct_retraction] > self.retraction_persistence_initiation_threshold:
                self.retraction_persistence_counter[leg_to_correct_retraction] = 1
        self._update_persistence_counter()
        stumbling_mask = self._get_stumbling_mask(obs)
        self.cpg_network.step()

        joint_angles = np.empty(self._n_out, dtype=float)
        adhesion_onoff = []
        net_corrections = np.zeros(6, dtype=float)
        for leg_idx, leg in enumerate(self.legs):
            self._update_retraction_correction(leg_idx, leg_to_correct_retraction)
            self._update_stumbling_correction(leg_idx, stumbling_mask[leg_idx])
            if self.retraction_correction[leg_idx] > 0:
                net_correction = self.retraction_correction[leg_idx]
                self.stumbling_correction[leg_idx] = 0
            else:
                net_correction = self.stumbling_correction[leg_idx]
            phase = self.cpg_network.curr_phases[leg_idx]
            magnitude = self.cpg_network.curr_magnitudes[leg_idx]
            leg_angles = self.preprogrammed_steps.get_joint_angles(leg, phase, magnitude)
            net_correction = np.clip(net_correction, 0, self.max_correction)
            phase_gain = _step_phase_gain(
                phase % (2 * np.pi), self.preprogrammed_steps.swing_period[leg], self.swing_extension
            )
            leg_angles = leg_angles + net_correction * phase_gain * self._corr_vecs[leg_idx]
            net_corrections[leg_idx] = net_correction * phase_gain
            joint_angles[self._out_idx[leg_idx]] = leg_angles
            adhesion_onoff.append(self._get_adhesion_onoff(leg, phase) if self.enable_adhesion else False)

        self.last_info = {
            "net_corrections": net_corrections.copy(),
            "retraction_correction": self.retraction_correction.copy(),
            "stumbling_correction": self.stumbling_correction.copy(),
            "stumbling_mask": stumbling_mask.copy(),
            "leg_to_correct_retraction": leg_to_correct_retraction,
        }
        return LocomotionAction(joint_angles=joint_angles, adhesion_onoff=np.array(adhesion_onoff, dtype=bool))


class CachedObservationExtractor:
    """Same values as `HybridControllerObservation.from_sim`, with body indices cached."""

    def __init__(self, sim, fly_name: str, legs: tuple[str, ...], stumbling_links=_DETECTED_STUMBLING_LINKS):
        fly = sim.world.fly_lookup[fly_name]
        seg = type(fly).BODY_SEGMENT_CLASS
        order = fly.get_bodysegs_order()
        self.sim, self.fly_name = sim, fly_name
        self.thorax_idx = order.index(seg("c_thorax"))
        self.tarsus_idx = np.array([order.index(seg(f"{leg}_tarsus5")) for leg in legs])
        self.detected = [seg(f"{leg}_{link}") for leg in legs for link in stumbling_links]
        self.shape = (len(legs), len(stumbling_links), 3)
        self.thorax_body_id = sim._internal_bodyids_by_fly[fly_name][self.thorax_idx]

    def __call__(self) -> HybridControllerObservation:
        sim = self.sim
        positions = sim.get_body_positions(self.fly_name)
        return HybridControllerObservation(
            thorax_z=float(positions[self.thorax_idx, 2]),
            tarsus5_z=np.array(positions[self.tarsus_idx, 2], dtype=float),
            stumbling_contact_forces=sim.get_bodysegment_contact_forces(
                self.fly_name, self.detected, ground_only=True
            ).reshape(self.shape),
            fly_heading=sim.mj_data.xmat[self.thorax_body_id].reshape(3, 3)[:, 0].copy(),
        )
