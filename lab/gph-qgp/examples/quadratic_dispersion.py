"""Quadratic dispersion: GRB photon time-of-flight at order n=2.

Simulates 40 photons (0.1-100 GeV) from a burst 1 Gpc away with
eta = 1 at the Planck scale under the quadratic MDR, then fits eta
back from the arrival times. Compare with examples/grb_delay.py
(the n=1 version): the delay now grows with E^2, and the fitted
eta is recovered from a regression against E^2.

Run:  python examples/quadratic_dispersion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gph_qgp import constraints, dispersion

energies = np.logspace(-1, 2, 40)  # GeV
# At the Planck scale the n=2 delays (~1e-17 s) sit far below any
# realistic emission jitter, so the fit is demonstrated at a low
# scale (1e12 GeV) where the delays are measurable — the same
# regression a real analysis would run, just on a visible signal.
sim = dispersion.simulate_grb(energies, distance_mpc=1000.0, eta=1.0,
                              emission_jitter_s=1e-4, seed=0,
                              e_qg_gev=1e12, order=2)
fit = dispersion.fit_eta(sim["energies_gev"], sim["arrival_s"], 1000.0,
                         e_qg_gev=1e12, order=2)

print(f"True eta (n=2): 1.0")
print(f"Recovered eta: {fit['eta']:.3f} +/- {fit['eta_err']:.3f}")
print(f"r = {fit['r_value']:.3f}, p = {fit['p_value']:.2e}")
print(f"Delay of a 100 GeV photon: "
      f"{dispersion.time_delay(100.0, 1000.0, eta=1.0, order=2):.3e} s")
print(f"Delay of a 100 GeV photon at n=1 would be: "
      f"{dispersion.time_delay(100.0, 1000.0, eta=1.0, order=1):.3f} s")
res = constraints.check_dispersion(eta=1.0, order=2)
print(f"Against the real limits, a Planck-scale quadratic term is "
      f"{res['verdict']} (best limit 1.2e12 GeV, far below E_Pl).")
