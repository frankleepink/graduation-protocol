"""Spacetime-foam distance fluctuations and interferometer phase noise.

Models (Ng & van Dam style)
---------------------------
A length L measured over a distance is uncertain by an amount that grows
with L. Two standard toy scalings are implemented:

* "holographic":  delta_l ~ l_P^(2/3) * L^(1/3)
* "random_walk":   delta_l ~ sqrt(l_P * L)

where l_P = 1.616e-35 m is the Planck length. The holographic scaling
comes from the holographic principle (the number of degrees of freedom
in a volume scales with its surface area); the random-walk scaling is
what you get if Planck-scale kicks accumulate like a random walk.
Both are *toy* scalings for a quantum-gravity effect that has never
been observed; they disagree with each other by many orders of
magnitude, which is itself the point — the models are not settled.

What is computed
----------------
* ``fluctuation(length_m, model)`` — the RMS distance uncertainty (m).
* ``phase_noise(...)`` — the RMS phase uncertainty (radians) that
  fluctuation produces in an interferometer of a given wavelength:
  sigma_phi = 2 * pi * delta_l / lambda.
* ``simulate_interferometer(...)`` — a synthetic time series of phase
  noise for an arm of a given length, so the scaling can be seen
  directly.

Assumptions and limits (plain language)
----------------------------------------
* These are order-of-magnitude phenomenological models, not derivations
  from a theory of quantum gravity. No experiment has detected spacetime
  foam.
* The fluctuation is treated as white Gaussian noise on the phase.
  Real foam, if it exists, need not be Gaussian or white.
* No detector noise, seismic noise, or thermal noise is included — a
  real interferometer is dominated by those, not by foam.
* Units: lengths in metres, wavelengths in metres, phase in radians.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
L_PLANCK_M = 1.616255e-35   # Planck length, m
C_M_PER_S = 299_792_458.0   # speed of light, m/s

MODELS = ("holographic", "random_walk")


# ------------------------------------------------------------------- model
def fluctuation(length_m, model="holographic"):
    """RMS distance uncertainty (m) over a length ``length_m`` (m).

    holographic : delta_l = l_P^(2/3) * L^(1/3)
    random_walk : delta_l = sqrt(l_P * L)
    """
    length_m = np.asarray(length_m, dtype=float)
    if np.any(length_m < 0):
        raise ValueError("length_m must be >= 0")
    if model == "holographic":
        return L_PLANCK_M ** (2.0 / 3.0) * length_m ** (1.0 / 3.0)
    if model == "random_walk":
        return np.sqrt(L_PLANCK_M * length_m)
    raise ValueError(f"unknown model {model!r}; choose from {MODELS}")


def phase_noise(length_m, wavelength_m, model="holographic"):
    """RMS interferometer phase noise (radians).

    sigma_phi = 2 * pi * delta_l / lambda, with delta_l from
    ``fluctuation``. ``length_m`` is the arm length, ``wavelength_m``
    the laser wavelength.
    """
    delta_l = fluctuation(length_m, model=model)
    return 2.0 * np.pi * delta_l / float(wavelength_m)


def fluctuation_vs_length(lengths_m, model="holographic"):
    """Table of delta_l for an array of lengths (m). Returns a dict
    with the input lengths and the fluctuation for each."""
    lengths = np.asarray(lengths_m, dtype=float)
    return {"length_m": lengths, "delta_l_m": fluctuation(lengths, model=model)}


# -------------------------------------------------------------- simulation
def simulate_interferometer(arm_length_m, wavelength_m, model="holographic",
                            n_samples=10000, sample_rate_hz=1000.0, seed=0):
    """Simulate a phase-noise time series for one interferometer arm.

    The phase at each sample is drawn as Gaussian noise with standard
    deviation ``phase_noise(arm_length_m, wavelength_m, model)`` — a
    white-noise stand-in for foam-induced jitter. Returns a dict with
    the time axis (s), the phase series (rad), and the theoretical
    sigma_phi for comparison with the realised sample std.

    This is a demonstration of the scaling, not a detector simulation.
    """
    rng = np.random.default_rng(seed)
    sigma_phi = float(phase_noise(arm_length_m, wavelength_m, model=model))
    times = np.arange(n_samples) / float(sample_rate_hz)
    phase = rng.normal(0.0, sigma_phi, size=n_samples)
    return {
        "time_s": times,
        "phase_rad": phase,
        "sigma_phi_theory_rad": sigma_phi,
        "sigma_phi_sample_rad": float(np.std(phase, ddof=1)),
        "arm_length_m": float(arm_length_m),
        "wavelength_m": float(wavelength_m),
        "model": model,
    }
