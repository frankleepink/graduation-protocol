"""CSL: collapse rates and heating against the published bounds.

Evaluates the Continuous Spontaneous Localisation model at the
standard GRW values (lambda = 1e-16 /s, r_C = 1e-7 m) and shows
which published bounds it passes and which it fails. The collapse
step itself is not simulated — collapse_oracle() still raises.

Run:  python examples/csl_bounds.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gph_qgp import collapse, constraints

for mass in (1e-18, 1e-15, 1e-12):
    print(f"mass {mass:.0e} kg: CSL rate = "
          f"{collapse.csl_rate(mass):.3g} /s, heating = "
          f"{collapse.csl_heating_power(mass):.3g} W")

res = constraints.check_csl(1e-16, 1e-7)
print(f"\nStandard GRW values (lambda=1e-16 /s, r_C=1e-7 m): "
      f"{res['verdict']}")
for p in res["per_constraint"]:
    print(f"  {p['constraint_id']}: {p['verdict']} "
          f"(value {p['ratio_value']:.3g} vs limit {p['limit']:.3g} "
          f"{p['limit_units']})")

res2 = constraints.check_csl(1e-17, 1e-7)
print(f"\nlambda = 1e-17 /s at r_C = 1e-7 m: {res2['verdict']} "
      f"(lambda/r_C^2 = {res2['lambda_over_rc2']:.3g}, below the "
      f"XENONnT bound 3.0e-3)")

dp = constraints.check_dp_cutoff(0.0)
print(f"\nParameter-free Diosi-Penrose (R_0 = 0): {dp['verdict']} "
      f"(Majorana lower limit 2.54e-10 m)")
dp2 = constraints.check_dp_cutoff(1e-9)
print(f"R_0 = 1e-9 m: {dp2['verdict']}")

try:
    collapse.collapse_oracle()
except collapse.CollapseNotSimulable as exc:
    print(f"\nOracle intact: {type(exc).__name__} raised as expected.")
