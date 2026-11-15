"""Neutrinos: SN1987A time-of-flight under LIV.

Simulates the SN1987A neutrino burst (7.5-40 MeV, 51.4 kpc) under
subluminal LIV at the Planck scale and shows the energy-dependent
spread, then checks a Planck-scale model against the published
neutrino limits.

Run:  python examples/sn1987a_neutrinos.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gph_qgp import constraints, neutrinos

energies_mev = np.linspace(7.5, 40.0, 10)
for order in (1, 2):
    sim = neutrinos.simulate_sn1987a(
        energies_mev, neutrinos.E_PLANCK_GEV, order=order)
    print(f"n={order}, M_QG = E_Pl: delay at 40 MeV = "
          f"{sim['delays_s'][-1]:.3g} s, at 7.5 MeV = "
          f"{sim['delays_s'][0]:.3g} s, spread = "
          f"{sim['spread_s']:.3g} s (observed burst lasted ~13 s)")

for order in (1, 2):
    res = constraints.check_neutrino(
        neutrinos.E_PLANCK_GEV, order=order, branch="subluminal")
    print(f"Planck-scale subluminal neutrino n={order}: "
          f"{res['verdict']}")
res_sup = constraints.check_neutrino(
    neutrinos.E_PLANCK_GEV, order=1, branch="superluminal")
print(f"Planck-scale superluminal neutrino n=1: {res_sup['verdict']} "
      f"(IceCube PeV bound M_QG > ~1e5 M_Pl)")
delta = neutrinos.speed_deviation(0.01, neutrinos.E_PLANCK_GEV, order=1)
res_speed = constraints.check_neutrino_speed(delta)
print(f"Speed deviation at 10 MeV, M_QG = E_Pl: delta = {delta:.3g} — "
      f"{res_speed['verdict']} vs the SN1987A bound 4e-9.")
