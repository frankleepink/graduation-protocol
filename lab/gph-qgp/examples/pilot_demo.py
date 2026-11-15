"""GPH sober pilot pipeline on simulated data.

Simulates blinded observers (stored mode: a shared pattern in the
live-wall condition only), analyses agreement, and runs the
invariance demo. Writes observers.csv, summary.csv, summary.json
into ./pilot_demo_out/.

Run:  python examples/pilot_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gph_qgp import pilot

out = Path(__file__).resolve().parent / "pilot_demo_out"
summary = pilot.run_demo(str(out), seed=7)

for cond, d in summary["per_condition"].items():
    print(f"{cond}: report rate {d['report_rate']:.2f}, "
          f"pairwise agreement {d['mean_pairwise_agreement']:.3f}, "
          f"alpha {d['alpha_nominal']:.3f} "
          f"95% CI [{d['alpha_ci95'][0]:.3f}, {d['alpha_ci95'][1]:.3f}]")
comp = summary["comparison"]
print(f"W vs S: alpha diff {comp['alpha_diff_a_minus_b']:.3f}, "
      f"p = {comp['perm_p_two_sided']:.4f}")
inv = summary["invariance_demo"]
print(f"Invariance mean best-r (changed settings): "
      f"stored {inv['stored']['mean_best_r_changed']:.3f} vs "
      f"conventional {inv['conventional']['mean_best_r_changed']:.3f}")
print(f"\nWrote {out}/observers.csv, summary.csv, summary.json")
