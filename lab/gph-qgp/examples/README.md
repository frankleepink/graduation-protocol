# Examples

Each script runs standalone from the repo root (no install needed —
they add `src/` to the path themselves):

| Script | What it does |
|---|---|
| `grb_delay.py` | Simulate GRB photon arrivals under a linear MDR (eta = 1, 1 Gpc) and fit eta back. |
| `foam_noise.py` | Spacetime-foam fluctuations and phase noise for 1 m / 100 m / 4 km arms, both scalings. |
| `collapse_toy.py` | GRW trajectory on a toy wavepacket, Diosi–Penrose times, and the `collapse_oracle()` boundary. |
| `pilot_demo.py` | Full sober-pilot pipeline on simulated observers; writes CSV/JSON into `pilot_demo_out/`. |

```bash
python examples/grb_delay.py
python examples/foam_noise.py
python examples/collapse_toy.py
python examples/pilot_demo.py
```

For the full CLI (CSV + plots), see the root README:

```bash
pip install -e .
python -m gph_qgp run --experiment dispersion --out results/
```

All outputs are simulated data for teaching and pipeline testing — see
the honesty section in the root README.
