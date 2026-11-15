"""Vacuum birefringence: helicity-dependent photon speed (v0.3).

If Lorentz violation is helicity-dependent, left- and right-circularly
polarised photons travel at different speeds, and the plane of linear
polarisation rotates as the photon propagates. The rotation grows with
the square of the energy and with the distance, so a distant gamma-ray
burst (GRB) observed in polarised light is a sensitive probe: if the
rotation varied strongly across the observed energy band, the net
polarisation would wash out — and it does not.

Model (Toma et al. 2012; Wei et al. 2025 convention)
-----------------------------------------------------
    Delta_phi(E) = eta * E^2 * D / (hbar * E_Pl * c)

where eta is the dimensionless birefringence parameter, E the photon
energy, D the source distance, and E_Pl the Planck energy. The two
helicities travel at v/c = 1 +/- eta * E / E_Pl, so the phase difference
between them accumulates as the square of the energy. (O(1) convention
factors differ between authors — some write xi instead of eta, or
absorb a factor of 2 — so the parameter below is the Wei et al. 2025
definition, and the bound quoted is in that definition.)

Verified limit (see constraints.py for the full citation)
--------------------------------------------------------
* |eta| < O(1e-15 - 1e-16), 95% CL, from energy-resolved polarimetry
  of five GRBs (Wei et al., Astrophys. J. 983, 164 (2025),
  arXiv:2503.18277). The module uses the conservative end,
  |eta| < 1e-15, as the bound value.

Honesty note: the GRBs in the simulator are synthetic. The check
against the real bound is constraints.check_birefringence; a simulated
polarisation measurement that survives is a pipeline demonstration,
not a measurement of eta.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
E_PLANCK_GEV = 1.22e19               # Planck energy, GeV
HBAR_GEV_S = 6.582119569e-25         # reduced Planck constant, GeV s
C_M_PER_S = 299_792_458.0            # speed of light, m/s
MPC_IN_M = 3.085677581491367e22       # one megaparsec, m

# Conservative end of the Wei et al. (2025) bound, |eta| < O(1e-15-1e-16).
ETA_LIMIT = 1.0e-15                  # dimensionless, 95% CL


def rotation_angle(energy_gev, distance_mpc, eta):
    """Polarisation rotation angle (radians) for a photon.

    Delta_phi = eta * E^2 * D / (hbar * E_Pl * c). Sign follows eta;
    the magnitude is what observations constrain. Works on scalars
    and arrays.
    """
    energy_gev = np.asarray(energy_gev, dtype=float)
    distance_m = float(distance_mpc) * MPC_IN_M
    return float(eta) * energy_gev**2 * distance_m / (
        HBAR_GEV_S * E_PLANCK_GEV * C_M_PER_S)


def rotation_difference(energy_low_gev, energy_high_gev, distance_mpc,
                        eta):
    """Rotation difference (radians) across an energy band.

    Delta_phi(E_high) - Delta_phi(E_low). A difference of order pi/2
    or more across the band washes out the net polarisation; that is
    what the GRB polarimetry limits are really measuring.
    """
    return float(rotation_angle(energy_high_gev, distance_mpc, eta)
                 - rotation_angle(energy_low_gev, distance_mpc, eta))


def band_polarization(energies_gev, distance_mpc, eta, p0=1.0):
    """Observed polarisation fraction after rotation across a band.

    Each energy in the band rotates by a different angle, so the
    polarisation vectors partially cancel. The surviving fraction is
    p0 * |mean(exp(2i * Delta_phi))| over the band — 1 if nothing
    rotates, ~0 if the band is fully scrambled. p0 is the intrinsic
    (source) polarisation fraction.
    """
    energies_gev = np.asarray(energies_gev, dtype=float)
    angles = rotation_angle(energies_gev, distance_mpc, eta)
    phasors = np.exp(2j * angles)
    return float(p0 * abs(np.mean(phasors)))


def eta_limit(rotation_max_rad, energy_gev, distance_mpc):
    """Invert the rotation formula: the eta that would produce a given
    rotation at a given energy and distance. Used to translate an
    observed "no rotation larger than X" into a bound on eta."""
    energy_gev = float(energy_gev)
    distance_m = float(distance_mpc) * MPC_IN_M
    if energy_gev <= 0 or distance_m <= 0:
        raise ValueError("energy and distance must be > 0")
    return float(rotation_max_rad) * HBAR_GEV_S * E_PLANCK_GEV * C_M_PER_S \
        / (energy_gev**2 * distance_m)


def simulate_grb_polarimetry(energies_gev, distance_mpc, eta, p0=1.0):
    """Simulate a GRB polarisation measurement under birefringence.

    ``energies_gev``: the observed energy band. Returns the rotation
    per energy, the rotation spread across the band, and the surviving
    polarisation fraction. A real GRB that is observed polarised with
    p ~ 0.3-0.8 in a band like this rules out any eta that would
    scramble it.
    """
    energies_gev = np.asarray(energies_gev, dtype=float)
    rotations = rotation_angle(energies_gev, distance_mpc, eta)
    return {
        "energies_gev": energies_gev,
        "distance_mpc": float(distance_mpc),
        "eta": float(eta),
        "rotation_rad": rotations,
        "rotation_spread_rad": float(rotations.max() - rotations.min()),
        "polarization_fraction": band_polarization(
            energies_gev, distance_mpc, eta, p0=p0),
        "intrinsic_polarization": float(p0),
    }
