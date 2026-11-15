"""Spacetime-foam phase noise for interferometer arms of different lengths.

Compares the holographic and random-walk scalings, and simulates a
10 s phase-noise series for a 4 km arm at 1064 nm.

Run:  python examples/foam_noise.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gph_qgp import foam

for length in (1.0, 100.0, 4000.0):
    for model in foam.MODELS:
        dl = foam.fluctuation(length, model)
        ph = foam.phase_noise(length, 1064e-9, model)
        print(f"L = {length:>7.0f} m  {model:<13} delta_l = {dl:.3e} m  "
              f"sigma_phi = {ph:.3e} rad")

sim = foam.simulate_interferometer(4000.0, 1064e-9, model="holographic",
                                   seed=0)
print(f"\nSimulated 4 km holographic phase series: "
      f"sample sigma = {sim['sigma_phi_sample_rad']:.3e} rad "
      f"(theory {sim['sigma_phi_theory_rad']:.3e} rad)")
