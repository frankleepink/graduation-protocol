"""GUP: minimal length, oscillator levels, and black-hole temperature.

Evaluates the generalised uncertainty principle for a macroscopic
resonator (0.3 kg, 10 kHz, the Bushev et al. setup) and for black
holes, and compares beta_0 with the published resonator bound.

Run:  python examples/gup_oscillator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gph_qgp import constraints, gup

print(f"Minimal length at beta_0 = 1: {gup.minimal_length(1.0):.3g} m "
      f"(the Planck length)")
print(f"Minimal length at the bound beta_0 = 5.2e6: "
      f"{gup.minimal_length(5.2e6):.3g} m")

omega = 2 * np.pi * 1e4  # 10 kHz resonator
levels = gup.oscillator_levels(np.arange(0, 4), omega, 0.3, beta0=1.0)
print("Oscillator levels (J) at beta_0 = 1, n = 0..3:")
for n, e in enumerate(levels):
    print(f"  n={n}: {e:.6g}")
print(f"Frequency shift at beta_0 = 5.2e6, 1 nm amplitude: "
      f"{gup.frequency_shift_fraction(1e-9, 0.3, omega, 5.2e6):.3g}")

for mass in (1e12, 1e15):
    print(f"Black hole {mass:.0e} kg: T_H = "
          f"{gup.hawking_temperature_hbar(mass):.3g} K, "
          f"T_GUP(beta_0=1) = {gup.hawking_temperature(mass, 1.0):.3g} K")
print(f"Remnant mass at beta_0 = 1: {gup.remnant_mass(1.0):.3g} kg")

res = constraints.check_gup(beta0=1.0)
print(f"beta_0 = 1 is {res['verdict']} (bound 5.2e6).")
res2 = constraints.check_gup(beta0=1e7)
print(f"beta_0 = 1e7 is {res2['verdict']}.")
