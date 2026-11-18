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
| `dispersion` | Modified dispersion relations, orders n = 1 and n = 2 (Amelino-Camelia style): E² = p²c² + ηE^(n+2)/E_QG^n. GRB photon time-of-flight simulator + a fitter that recovers η from simulated arrivals at either order. |
| `foam` | Spacetime-foam distance fluctuations (Ng & van Dam): holographic δl ~ l_P^(2/3)·L^(1/3) and random-walk δl ~ √(l_P·L). Interferometer phase-noise simulation. |
| `collapse` | GRW stochastic-hit toy model on a 1-D wavefunction (split-step unitary evolution + Poisson hits), Diosi–Penrose collapse time τ ~ ħ/ΔE_grav, and CSL phenomenology (mass-proportional rate, spontaneous heating). The collapse step itself is an explicit, labelled **non-computable oracle boundary** — see below. |
| `superluminal` | Superluminal LIV branch: photon decay (γ → e⁺e⁻) thresholds, photon splitting (γ → 3γ) mean free path, vacuum Cherenkov for electrons, Crab photon survival. |
| `birefringence` | Helicity-dependent photon speed: polarisation rotation Δφ = ηE²D/(ħE_Pl·c), band depolarisation, GRB polarimetry simulation. |
| `gup` | Generalised uncertainty principle: minimal length, GUP-corrected harmonic-oscillator levels, amplitude-dependent frequency shift, GUP-corrected Hawking temperature and remnant mass. |
| `neutrinos` | Energy-dependent neutrino speed: SN1987A and TXS 0506+056 time-of-flight delays, IceCube-scale superluminal check. |
| `constraints` | Real published limits, verified against primary sources: GRB time-of-flight (Abdo 2009, Vasileiou 2013, LHAASO 2024, Yang 2024), LIGO O1 strain floor (Abbott 2016), Ng & van Dam 40-m bounds, Holometer (Chou 2017) — plus v0.3: LHAASO superluminal limits, GRB birefringence (Wei 2025), GUP resonator bound (Bushev 2019), CSL bounds (Carlesso 2016, Majorana 2022, XENONnT 2025), neutrino limits (SN1987A, TXS 0506+056, IceCube). Checks the toy models against them — see "Real limits" below. |
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
python -m gph_qgp run --experiment superluminal --out results/
python -m gph_qgp run --experiment birefringence --out results/
python -m gph_qgp run --experiment gup --out results/
python -m gph_qgp run --experiment csl --out results/
python -m gph_qgp run --experiment neutrinos --out results/

# Compare the toy models with real published limits (v0.3)
python -m gph_qgp compare --out results/
```

## Try it with no install

- **Web lab (no install):** the interactive browser version runs the same mechanics live — dispersion, photon decay, GUP, CSL, neutrinos, birefringence — with sliders, real-event presets (GRB 090510, LHAASO 1.4 PeV, IceCube-170922A, the Crab) and SURVIVES / EXCLUDED verdicts.
- **Colab notebook (no install):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frankleepink/graduation-protocol/blob/main/lab/gph-qgp/examples/gph-qgp-colab.ipynb) — `examples/gph-qgp-colab.ipynb` installs the lab from GitHub and runs dispersion, the real-limits comparison, and the pilot pipeline.
- **Install from PyPI:** `pip install gph-qgp` — v0.3.0 is live on PyPI: https://pypi.org/project/gph-qgp/0.3.0/

## What running it looks like

`python -m gph_qgp compare` (v0.3, 117 tests pass) checks the toy models against real published limits. Sample output is in `sample-outputs/comparison.csv`:

- Linear dispersion, η = 1 at E_Pl — **EXCLUDED** (GRB time-of-flight: E_QG,1 > 12.3 E_Pl)
- Quadratic dispersion — **SURVIVES** (best limit 1.2e12 GeV, four orders below E_Pl)
- Random-walk foam at 4 km — **EXCLUDED** (LIGO would have seen it, factor ~2770)
- Holographic foam — **UNCONSTRAINED** (below LIGO sensitivity)
- Superluminal photons n=1 — **EXCLUDED**; n=2 — **SURVIVES**
- Superluminal electrons — **EXCLUDED**; birefringence η = 1 — **EXCLUDED**
- GUP β₀ = 1 — **SURVIVES**; CSL textbook GRW values — **EXCLUDED** (by the XENONnT preprint bound only)
- Diósi–Penrose parameter-free — **EXCLUDED**; neutrinos subluminal — **SURVIVE**, superluminal n=1 — **EXCLUDED**

Toy models only, simulated data only — never evidence for the hypothesis. A positive result in these sims demonstrates the analysis pipeline, never proof of simulation or curriculum.

## Real limits (v0.3) — where the models already die

`python -m gph_qgp compare` checks the package's own toy models
against published observations. Every number is verified against its
primary source (citation next to the number in `constraints.py`).

| Model | Real limit | Verdict |
|---|---|---|
| Linear dispersion, η = 1 at E_Pl | GRB time-of-flight: E_QG,1 > 1.2 E_Pl (Abdo et al. 2009, Nature 462, 331); > 7.6 E_Pl (Vasileiou et al. 2013, PRD 87, 122001); > 10 E_Pl (LHAASO 2024, PRL 133, 071501); > 12.3 E_Pl (Yang et al. 2024, JCAP 04, 060) | **EXCLUDED** — the Planck-scale linear model is below every published limit |
| Quadratic dispersion, η = 1 at E_Pl | Best subluminal limit E_QG,2 > 1.2×10¹² GeV (Yang et al. 2024) | **SURVIVES** — v0.3 tests a real quadratic model; the limit is four orders below E_Pl, so a Planck-scale quadratic term is allowed (v0.2 reported UNCONSTRAINED because the model had no quadratic term) |
| Random-walk foam, 4 km | LIGO O1 strain floor 10⁻²³/√Hz at 100 Hz (Abbott et al. 2016, PRL 116, 131103): predicted noise ~1.1×10⁻¹⁶ m/√Hz, ~2800× above the floor | **EXCLUDED** — LIGO would have seen it |
| Holographic foam, 4 km | Same LIGO floor: RMS fluctuation ~1.0×10⁻²² m, ~400× below the floor in a 1 Hz band | **UNCONSTRAINED** — below sensitivity, not tested |
| Hogan shear-symmetry holographic noise | Fermilab Holometer, final data ~10× below the Planck-scale prediction (Chou et al. 2017, CQG 34, 165005) | **EXCLUDED** — that specific model, not the holographic principle and not the generic holographic scaling |

### v0.3 — six new mechanics, same treatment

| Model | Real limit | Verdict |
|---|---|---|
| Superluminal photons, n = 1, E_LV = E_Pl | LHAASO 1.42 PeV Crab photon: E_LV > 2.74×10²⁴ GeV (He & Ma 2023, PLB 835, 137536); collaboration's own number 1.42×10²⁴ GeV (LHAASO 2022, PRL 128, 051102) | **EXCLUDED** — the photon would have decayed (γ → e⁺e⁻) |
| Superluminal photons, n = 2, E_LV = E_Pl | Photon splitting: E_LV > 1×10⁻³ M_Pl = 1.22×10¹⁶ GeV (LHAASO 2022, PRL 128, 051102) | **SURVIVES** — the splitting bound is only 10⁻³ M_Pl, and the pair-decay threshold at E_Pl is ~112 PeV, above the observed photon |
| Superluminal electrons, n = 1, E_LV = E_Pl | Crab 2.3 PeV parent electron: E_LV > 9.4×10²⁵ GeV (He & Ma 2023) | **EXCLUDED** — the electron would have radiated vacuum Cherenkov down to threshold |
| Vacuum birefringence, η = 1 | GRB polarimetry: \|η\| < 10⁻¹⁵, 95% CL (Wei et al. 2025, ApJ 983, 164) | **EXCLUDED** — a Planck-scale birefringence would scramble every GRB band |
| GUP, β₀ = 1 (Planck-length minimal length) | Sapphire resonator: β₀ < 5.2×10⁶ (Bushev et al. 2019, PRD 100, 066020) | **SURVIVES** — the bound is six orders above a Planck-length minimal length; the experiment cannot see β₀ = 1 |
| CSL, standard GRW values (λ = 10⁻¹⁶ /s, r_C = 10⁻⁷ m) | X-ray: λ/r_C² < 3.0×10⁻³ s⁻¹m⁻², 90% CL (XENONnT 2025, arXiv:2506.05507); < 0.494 (Majorana 2022); LISA Pathfinder λ < 3×10⁻⁸ /s at r_C = 10⁻⁷ m (Carlesso et al. 2016, PRD 94, 124036) | **EXCLUDED** — by the newest XENONnT X-ray bound (λ/r_C² = 10⁻²); Majorana and LISA Pathfinder alone would pass it |
| Diosi–Penrose, parameter-free (R₀ = 0) | Majorana: R₀ > 2.54×10⁻¹⁰ m (arXiv:2202.01343) | **EXCLUDED** — a cutoff-free DP model is ruled out |
| Neutrinos, subluminal, M_QG = E_Pl | SN1987A: M₁ > 2.7×10¹⁰ GeV, M₂ > 4.6×10⁴ GeV; TXS 0506+056 flare: M₁ > 10¹⁴ GeV, M₂ > 10⁹ GeV (Bustamante et al. 2024, arXiv:2408.15949) | **SURVIVES** — the strongest subluminal neutrino limit is five orders below E_Pl |
| Neutrinos, superluminal, n = 1, M_QG = E_Pl | IceCube PeV neutrinos: M_QG > ~10⁵ M_Pl (Borriello et al. 2013, PRD 87, 116009) | **EXCLUDED** — a superluminal PeV neutrino would lose energy by pair bremsstrahlung and never arrive |

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
python examples/quadratic_dispersion.py
python examples/superluminal_photon.py
python examples/birefringence_grb.py
python examples/gup_oscillator.py
python examples/csl_bounds.py
python examples/sn1987a_neutrinos.py
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
