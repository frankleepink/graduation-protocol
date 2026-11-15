"""Generalised Uncertainty Principle (GUP) minimal-length physics (v0.3).

The GUP is the most widely studied phenomenological form of a minimal
length: the Heisenberg commutator is deformed so that position cannot
be known more sharply than about the Planck length,

    [x, p] = i hbar (1 + beta p^2),     beta = beta_0 / (M_Pl c)^2

with beta_0 a dimensionless number expected to be of order 1 if the
minimal length is the Planck length, and larger if quantum-gravity
effects set in earlier. Consequences implemented here:

    minimal length:   Delta_x_min = hbar sqrt(beta) = l_P sqrt(beta_0)
    oscillator:       E_n = hbar w (n+1/2)
                        + (beta m hbar^2 w^2 / 4) (2n^2 + 2n + 1)
                      (first-order p^4 perturbation, Kempf, Mangano &
                      Mann 1995)
    amplitude-frequency effect: a classical oscillator's frequency
                      shifts with its amplitude,
                      Delta_w / w_0 = (beta_0 / 2)
                        * (m w_0 A / (M_Pl c))^2
                      derived from the levels above via w = dE/dJ
                      (J = hbar n, E ~ (1/2) m w^2 A^2). This is the
                      effect macroscopic-resonator experiments measure.
    black hole:       T_GUP = T_H * (1 + sqrt(1 - beta_0 (M_Pl/M)^2)) / 2
                      (Adler, Chen & Santiago 2001), defined only for
                      M >= sqrt(beta_0) M_Pl — below that the black hole
                      stops evaporating and leaves a remnant.

Verified limit (see constraints.py for the full citation)
--------------------------------------------------------
* beta_0 < 5.2e6, from a 0.3 kg sapphire split-bar resonator
  (Bushev et al., Phys. Rev. D 100, 066020 (2019), arXiv:1903.03346).
  A quartz-resonator estimate of beta_0 < 4e4 in the same paper is an
  estimate, not a firm bound, and is recorded as such in the notes.

Honesty note: the GUP is a phenomenological deformation, not a theory
of quantum gravity, and the oscillator/black-hole numbers here are toy
evaluations of its formulas. The check against the real bound is
constraints.check_gup.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
HBAR_J_S = 1.054571817e-34           # reduced Planck constant, J s
C_M_PER_S = 299_792_458.0            # speed of light, m/s
G_SI = 6.67430e-11                   # Newton constant, m^3 kg^-1 s^-2
K_B_J_PER_K = 1.380649e-23           # Boltzmann constant, J/K
M_PLANCK_KG = 2.176434e-8            # Planck mass, kg
L_PLANCK_M = 1.616255e-35            # Planck length, m

# Strongest verified bound on beta_0 (Bushev et al. 2019, sapphire).
BETA0_LIMIT = 5.2e6                  # dimensionless


def beta_si(beta0):
    """GUP parameter beta in SI units, 1/(kg m/s)^2.

    beta = beta_0 / (M_Pl c)^2. beta_0 = 1 means a minimal length of
    exactly the Planck length.
    """
    return float(beta0) / (M_PLANCK_KG * C_M_PER_S) ** 2


def minimal_length(beta0):
    """Minimal position uncertainty (m): Delta_x_min = l_P * sqrt(beta_0)."""
    if beta0 < 0:
        raise ValueError("beta0 must be >= 0")
    return L_PLANCK_M * float(np.sqrt(beta0))


def oscillator_level_shift(n, omega_rad_s, mass_kg, beta0):
    """GUP correction (J) to the n-th harmonic-oscillator level.

    (beta m hbar^2 w^2 / 4) * (2n^2 + 2n + 1), the first-order p^4
    perturbation of the deformed Hamiltonian. n = 0, 1, 2, ...
    """
    n = np.asarray(n, dtype=float)
    beta = beta_si(beta0)
    return (beta * float(mass_kg) * HBAR_J_S**2 * float(omega_rad_s)**2
            / 4.0) * (2.0 * n**2 + 2.0 * n + 1.0)


def oscillator_levels(n, omega_rad_s, mass_kg, beta0):
    """GUP-corrected harmonic-oscillator energy levels (J).

    E_n = hbar w (n + 1/2) + the correction from oscillator_level_shift.
    At beta_0 = 0 this reduces to the ordinary levels exactly.
    """
    n = np.asarray(n, dtype=float)
    return HBAR_J_S * float(omega_rad_s) * (n + 0.5) + \
        oscillator_level_shift(n, omega_rad_s, mass_kg, beta0)


def frequency_shift_fraction(amplitude_m, mass_kg, omega_rad_s, beta0):
    """Fractional GUP frequency shift of a classical oscillator.

    Delta_w / w_0 = (beta_0 / 2) * (m w_0 A / (M_Pl c))^2, derived
    from the level correction above (see module docstring). This is
    the quantity a resonator experiment measures: the frequency
    depends on the oscillation amplitude, which ordinary quantum
    mechanics forbids. A is the oscillation amplitude in metres.
    """
    if amplitude_m < 0:
        raise ValueError("amplitude_m must be >= 0")
    ratio = float(mass_kg) * float(omega_rad_s) * float(amplitude_m) / (
        M_PLANCK_KG * C_M_PER_S)
    return (float(beta0) / 2.0) * ratio**2


def hawking_temperature_hbar(mass_kg):
    """Ordinary Hawking temperature (K) of a black hole of ``mass_kg``.

    T_H = hbar c^3 / (8 pi G k_B M).
    """
    return HBAR_J_S * C_M_PER_S**3 / (
        8.0 * np.pi * G_SI * K_B_J_PER_K * float(mass_kg))


def remnant_mass(beta0):
    """Black-hole remnant mass (kg): M = sqrt(beta_0) * M_Pl.

    Below this mass the GUP-corrected temperature has no real value;
    evaporation stops.
    """
    if beta0 < 0:
        raise ValueError("beta0 must be >= 0")
    return float(np.sqrt(beta0)) * M_PLANCK_KG


def hawking_temperature(mass_kg, beta0):
    """GUP-corrected Hawking temperature (K).

    T_GUP = T_H * (1 + sqrt(1 - beta_0 (M_Pl/M)^2)) / 2 (Adler, Chen &
    Santiago 2001). Returns None for M < sqrt(beta_0) M_Pl, where the
    black hole has evaporated down to its remnant and the temperature
    is no longer defined. At beta_0 = 0 this is exactly T_H.
    """
    mass_kg = float(mass_kg)
    if mass_kg <= 0:
        raise ValueError("mass_kg must be > 0")
    if mass_kg < remnant_mass(beta0):
        return None
    correction = (1.0 + float(np.sqrt(
        1.0 - float(beta0) * (M_PLANCK_KG / mass_kg)**2))) / 2.0
    return float(hawking_temperature_hbar(mass_kg) * correction)
