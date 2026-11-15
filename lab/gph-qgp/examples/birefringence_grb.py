"""Birefringence: GRB polarisation rotation under helicity-dependent LIV.

Simulates a GRB polarisation measurement (100-500 keV band, 1 Gpc)
for three values of eta and shows the surviving polarisation
fraction in each case. At eta = 1e-16 the band mostly survives; at
the published bound (1e-15) this wide band is already substantially
scrambled — the Wei et al. bound comes from energy-resolved fits of
the rotation vs energy, not from band-averaged survival.

Run:  python examples/birefringence_grb.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gph_qgp import birefringence, constraints

energies_gev = np.linspace(0.1, 0.5, 25) / 1000.0  # 100-500 keV
for eta in (1e-16, 1e-15, 1e-14, 1.0):
    sim = birefringence.simulate_grb_polarimetry(
        energies_gev, 1000.0, eta, p0=0.6)
    print(f"eta = {eta:.0e}: rotation spread across the band = "
          f"{sim['rotation_spread_rad']:.3g} rad; surviving "
          f"polarisation = {sim['polarization_fraction']:.3f} "
          f"(intrinsic 0.6)")

res = constraints.check_birefringence(eta=1.0)
print(f"A Planck-scale birefringence (eta = 1) is {res['verdict']} "
      f"against the Wei et al. 2025 bound |eta| < 1e-15.")
print(f"An eta of 1e-16 would give a rotation of "
      f"{birefringence.rotation_angle(4e-4, 1000.0, 1e-16):.3g} rad "
      f"at 400 keV, 1 Gpc — just at the edge of detectability.")
