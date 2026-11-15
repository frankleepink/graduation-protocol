"""Real limits (v0.2): check the toy models against published data.

Runs in ~20 lines, no install needed (adds src/ to the path).
Every number here was verified against its primary source — see
constraints.py for the citation next to each one.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gph_qgp import constraints  # noqa: E402

for entry in constraints.summary_table():
    print(f"{entry['id']}: {entry['limit_value']:.3g} {entry['limit_units']} "
          f"({entry['confidence']}) — {entry['source']}")

print()
print("Linear dispersion, eta = 1 at the Planck scale:",
      constraints.check_dispersion(eta=1.0, order=1)["verdict"])
print("Linear dispersion, eta = 0.05 (effective 20 E_Pl):",
      constraints.check_dispersion(eta=0.05, order=1)["verdict"])
print("Random-walk foam at LIGO:",
      constraints.check_foam("random_walk")["verdict"])
print("Holographic foam at LIGO:",
      constraints.check_foam("holographic")["verdict"])
print("Hogan shear-symmetry holographic noise:",
      constraints.check_holometer()["verdict"])
