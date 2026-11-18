# The Lab — gph-qgp

A runnable Python package of toy quantum-gravity phenomenology models, built around the Graduation Protocol Hypothesis. Install it from PyPI:

```
pip install gph-qgp
```

Or run it with no install in the [Colab notebook](https://colab.research.google.com/github/frankleepink/graduation-protocol/blob/main/lab/gph-qgp/examples/gph-qgp-colab.ipynb).

## What's inside

- **Dispersion** — modified dispersion relations, orders n = 1 and n = 2; GRB photon time-of-flight simulator + fitter.
- **Foam** — spacetime-foam distance fluctuations (Ng & van Dam): holographic and random-walk.
- **Collapse** — GRW, Diósi–Penrose and CSL toy models. The collapse step itself is an explicit, labelled non-computable oracle boundary — it is never simulated.
- **Superluminal** — photon decay, photon splitting, vacuum Cherenkov.
- **Birefringence** — helicity-dependent photon speed, GRB polarimetry.
- **GUP** — generalised uncertainty principle, minimal length.
- **Neutrinos** — energy-dependent neutrino speed (SN1987A, TXS 0506+056, IceCube).
- **Pilot** — the sober blinded-pilot analysis pipeline (v3.2 §06).

## The honest scoreboard

`python -m gph_qgp compare` checks every toy model against real published limits, cited next to the numbers in the code:

- **EXCLUDED:** linear dispersion at the Planck scale, random-walk foam, superluminal photons n=1, superluminal electrons, birefringence η = 1, textbook GRW CSL values (by the XENONnT preprint bound only), parameter-free Diósi–Penrose, superluminal neutrinos n=1.
- **SURVIVES:** quadratic dispersion, holographic foam, GUP β₀ = 1, subluminal neutrinos.
- **UNCONSTRAINED:** holographic foam at LIGO sensitivity.

Toy models only, simulated data only — a positive result demonstrates the analysis pipeline, never proof of simulation or curriculum.

Source: [github.com/frankleepink/graduation-protocol/tree/main/lab/gph-qgp](https://github.com/frankleepink/graduation-protocol/tree/main/lab/gph-qgp) · PyPI: [pypi.org/project/gph-qgp](https://pypi.org/project/gph-qgp/)
