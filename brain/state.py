# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Recurrent state carried between brain ticks within an episode.

Feed-forward brains (the MLP baseline) have none. Recurrent brains (the MANC VNC model
at Milestone 6) keep their unit activations here so that `reset()` at episode start is
explicit and the state can be inspected or logged.
"""

from __future__ import annotations

import numpy as np


class BrainState:
    """Named float arrays with a common reset.

    Example:
        state = BrainState(rates=512, last_action=2)
        state.rates[:] = ...
        state.reset()
    """

    def __init__(self, **sizes: int) -> None:
        self._sizes = dict(sizes)
        self.reset()

    def reset(self) -> None:
        for name, size in self._sizes.items():
            setattr(self, name, np.zeros(size, dtype=np.float64))

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy() for name in self._sizes}
