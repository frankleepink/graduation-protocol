"""Collapse toy models: GRW hits on a wavepacket, Diosi-Penrose times.

Shows unitary evolution conserving norm, a GRW trajectory with
stochastic hits, and the DP collapse time for a range of masses.
Calling collapse_oracle() raises CollapseNotSimulable on purpose —
the collapse step itself is not simulated (see the module docstring).

Run:  python examples/collapse_toy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gph_qgp import collapse

x = np.linspace(-5e-7, 5e-7, 512)
psi0 = collapse.gaussian_wavepacket(x, x0=0.0, sigma=1e-7)
traj = collapse.grw_trajectory(psi0, x, dt=1e-4, mass_kg=1e-25,
                               lam_per_s=50.0, n_steps=200, seed=0)
print(f"GRW trajectory: {traj['n_hits']} hit(s); "
      f"norm range [{traj['norm_history'].min():.6f}, "
      f"{traj['norm_history'].max():.6f}]")

for mass in (1e-15, 1e-12, 1e-9):
    tau = collapse.dp_collapse_time(mass, 1e-6)
    print(f"DP tau for {mass:.0e} kg, 1 um sphere: {tau:.3e} s")

try:
    collapse.collapse_oracle()
except collapse.CollapseNotSimulable as exc:
    print(f"\ncollapse_oracle() raised as designed:\n{exc}")
