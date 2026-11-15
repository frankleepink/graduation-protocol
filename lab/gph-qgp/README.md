# gph-qgp — the GPH Quantum Gravity Phenomenology lab

A small Python package of **toy phenomenological models** for teaching
and exploration, built around the Graduation Protocol Hypothesis (GPH)
— Walter Hivoina's proposal that reality functions as a developmental
curriculum for consciousness.

The idea: pull the GPH paper, then *apply* modified mechanics to it.
Each module implements a real, standard toy model from quantum-gravity
phenomenology, so you can run the experiments yourself on simulated
data and see what the analysis looks like.

## What's inside

| Module | What it does |
|---|---|
| `dispersion` | Linear modified dispersion relation (Amelino-Camelia style): E² = p²c² + ηE³/E_P. GRB photon time-of-flight simulator + a fitter that recovers η from simulated arrivals. |
| `foam` | Spacetime-foam distance fluctuations (Ng & van Dam): holographic δl ~ l_P^(2/3)·L^(1/3) and random-walk δl ~ √(l_P·L). Interferometer phase-noise simulation. |
| `collapse` | GRW stochastic-hit toy model on a 1-D wavefunction (split-step unitary evolution + Poisson hits) and Diosi–Penrose collapse time τ ~ ħ/ΔE_grav. The collapse step itself is an explicit, labelled **non-computable oracle boundary** — see below. |
| `pilot` | The GPH sober blinded-pilot analysis pipeline (v3.2 §06): synthetic speckle generation, blinded-rater agreement (Cohen's κ, Fleiss' κ, Krippendorff's α), and camera-invariance checks. |

## Install

```bash
pip install -e .
```

Python ≥ 3.10. Dependencies: numpy, scipy, matplotlib only.

## Quickstart

```bash
# Run one experiment: writes CSV results + a PNG plot into the output dir
python -m gph_qgp run --experiment dispersion --out results/
python -m gph_qgp run --experiment foam --out results/
python -m gph_qgp run --experiment collapse --out results/
python -m gph_qgp run --experiment pilot --out results/
```

Or run the short standalone examples (no install needed):

```bash
python examples/grb_delay.py
python examples/foam_noise.py
python examples/collapse_toy.py
python examples/pilot_demo.py
```

Run the tests:

```bash
pip install pytest
python -m pytest
```

## The collapse oracle — read this

`collapse.py` simulates unitary evolution and applies GRW hits as a
phenomenological rule for what a collapsed wavefunction would look
like *after the fact*. It does **not** simulate collapse itself.
`collapse_oracle()` always raises `CollapseNotSimulable`, marking the
point where an algorithmic simulation stops — following Faizal et al.
(arXiv:2507.22950), who argue that collapse requires actualising a
Gödelian truth, which no algorithm can produce. Redden
(arXiv:2512.11807) counters that undecidability constrains provability,
not execution; the oracle is a labelled boundary for exploration, not a
settled claim. The point is that the boundary is explicit instead of
being silently papered over with a random-number draw.

## Honesty section

- **Toy models, not real experiments.** The dispersion relation, foam
  scalings, GRW hits and DP times are standard textbook toy models.
  None of them has been observed; the foam models disagree with each
  other by orders of magnitude.
- **Simulated data only.** Every number this package produces comes
  from synthetic data generated inside the package. It contains no real
  GRB data, no real interferometer data, and no real observer data.
- **Not evidence for GPH.** A "positive" result inside these
  simulations only demonstrates that the analysis pipeline works as
  designed — it can detect a pattern that was injected on purpose.
  It is never proof of simulation, of a curriculum, or of anything
  about the real world.
- **What it is for:** teaching the phenomenology, testing the pilot
  analysis pipeline before real data exists, and giving people a
  hands-on way to argue with the models.

## Links

- GPH paper v3.2: https://doi.org/10.5281/zenodo.24237333
- Repository: https://github.com/frankleepink/graduation-protocol
- Faizal et al., "Consequences of Undecidability in Physics on the Theory of Everything": https://arxiv.org/abs/2507.22950
- Redden, "Provability vs. Execution": https://arxiv.org/abs/2512.11807

## License

CC BY 4.0 — see `LICENSE`. Give credit to Walter Hivoina when you
share or adapt this work.
