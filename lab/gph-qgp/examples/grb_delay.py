"""GRB photon time-of-flight under a linear modified dispersion relation.

Simulates 40 photons (0.1-100 GeV) from a burst 1 Gpc away with
eta = 1, then fits eta back from the arrival times.

Run:  python examples/grb_delay.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gph_qgp import dispersion

energies = np.logspace(-1, 2, 40)  # GeV
sim = dispersion.simulate_grb(energies, distance_mpc=1000.0, eta=1.0,
                              emission_jitter_s=0.02, seed=0)
fit = dispersion.fit_eta(sim["energies_gev"], sim["arrival_s"], 1000.0)

print(f"True eta: 1.0")
print(f"Recovered eta: {fit['eta']:.3f} +/- {fit['eta_err']:.3f}")
print(f"r = {fit['r_value']:.3f}, p = {fit['p_value']:.2e}")
print(f"Delay of a 100 GeV photon: "
      f"{dispersion.time_delay(100.0, 1000.0, eta=1.0):.3f} s")
print(f"Delay of a 0.1 GeV photon: "
      f"{dispersion.time_delay(0.1, 1000.0, eta=1.0):.6f} s")
