"""Superluminal branch: survival of Crab photons under photon decay.

Simulates photons from the Crab Nebula (2 kpc) under superluminal
LIV at the Planck scale, both orders, and shows why LHAASO's 1.42
PeV photon kills the n=1 model but not the n=2 model.

Run:  python examples/superluminal_photon.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gph_qgp import constraints, superluminal

energies = np.logspace(3, 6.5, 12)  # 1 TeV .. ~3 PeV
for order in (1, 2):
    sim = superluminal.simulate_crab_survival(
        energies, superluminal.E_PLANCK_GEV, order=order)
    surv_142 = float(superluminal.photon_survival_probability(
        1.42e6, 2.0 * superluminal.KPC_IN_M,
        superluminal.E_PLANCK_GEV, order=order))
    print(f"n={order}: pair-decay threshold at E_Pl = "
          f"{sim['pair_decay_threshold_gev']:.3g} GeV; "
          f"survival of the 1.42 PeV photon = {surv_142:.3g}")
    res = constraints.check_superluminal_photon(
        superluminal.E_PLANCK_GEV, order=order)
    print(f"  -> Planck-scale superluminal n={order} model: "
          f"{res['verdict']}")

th_e = superluminal.electron_cherenkov_threshold(9.4e25, order=1)
print(f"Vacuum-Cherenkov threshold at the published electron bound "
      f"(9.4e25 GeV): {th_e:.3g} GeV — the Crab's 2.3 PeV electron "
      f"sits just below it, which is the point.")
