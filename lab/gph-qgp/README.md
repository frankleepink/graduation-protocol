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
| `constraints` | Real published limits, verified against primary sources: GRB time-of-flight (Abdo 2009, Vasileiou 2013, LHAASO 2024, Yang 2024), LIGO O1 strain floor (Abbott 2016), Ng & van Dam 40-m bounds, Holometer (Chou 2017). Checks the toy models against them — see "Real limits" below. |
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

# Compare the toy models with real published limits (v0.2)
python -m gph_qgp compare --out results/
```

## Real limits (v0.2) — where the models already die

`python -m gph_qgp compare` checks the package's own toy models
against published observations. Every number is verified against its
primary source (citation next to the number in `constraints.py`).

| Model | Real limit | Verdict |
|---|---|---|
| Linear dispersion, η = 1 at E_Pl | GRB time-of-flight: E_QG,1 > 1.2 E_Pl (Abdo et al. 2009, Nature 462, 331); > 7.6 E_Pl (Vasileiou et al. 2013, PRD 87, 122001); > 10 E_Pl (LHAASO 2024, PRL 133, 071501); > 12.3 E_Pl (Yang et al. 2024, JCAP 04, 060) | **EXCLUDED** — the Planck-scale linear model is below every published limit |
| Quadratic dispersion | Best subluminal limit E_QG,2 > 1.2×10¹² GeV (Yang et al. 2024) | **UNCONSTRAINED** — the toy model has no quadratic term, so the limit does not test it |
| Random-walk foam, 4 km | LIGO O1 strain floor 10⁻²³/√Hz at 100 Hz (Abbott et al. 2016, PRL 116, 131103): predicted noise ~1.1×10⁻¹⁶ m/√Hz, ~2800× above the floor | **EXCLUDED** — LIGO would have seen it |
| Holographic foam, 4 km | Same LIGO floor: RMS fluctuation ~1.0×10⁻²² m, ~400× below the floor in a 1 Hz band | **UNCONSTRAINED** — below sensitivity, not tested |
| Hogan shear-symmetry holographic noise | Fermilab Holometer, final data ~10× below the Planck-scale prediction (Chou et al. 2017, CQG 34, 165005) | **EXCLUDED** — that specific model, not the holographic principle and not the generic holographic scaling |

The 40-m interferometer parameter bounds (Ng & van Dam, arXiv:gr-qc/9906003)
do not exclude either foam model at Planck scale — they only bound the
model length parameters. That distinction is kept explicit in the code:
a parameter bound is not an exclusion.

Or run the short standalone examples (no install needed):

```bash
python examples/grb_delay.py
python examples/foam_noise.py
python examples/collapse_toy.py
python examples/pilot_demo.py
python examples/real_limits.py
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
