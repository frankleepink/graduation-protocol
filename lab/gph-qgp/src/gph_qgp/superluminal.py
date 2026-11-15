"""Superluminal branch of Lorentz-invariance violation (v0.3).

The other side of the modified dispersion relation. Where dispersion.py
covers the subluminal branch (high-energy photons arrive late), this
module covers the superluminal branch (high-energy photons travel
faster than c) — and the decays that branch makes possible:

    photon decay      gamma -> e+ e-   (allowed above a threshold energy)
    photon splitting  gamma -> 3 gamma (n = 2, no threshold, rate grows
                                        steeply with energy)
    vacuum Cherenkov  e -> e + gamma   (superluminal electrons radiate
                                        until they drop below threshold)

Model
-----
Superluminal MDR (natural units, c = 1):

    E^2 = p^2 + m^2 + E^(n+2) / E_LV^n        (LIV term with + sign)

so the particle carries an effective mass squared
m_eff^2 = E^(n+2) / E_LV^n. A photon can decay into e+ e- once
m_eff >= 2 m_e, which gives the pair-decay threshold

    E_th = (4 m_e^2)^(1/(n+2)) * E_LV^(n/(n+2))

(n = 1: E_th = (4 m_e^2 E_LV)^(1/3); n = 2: E_th = (4 m_e^2)^(1/4) *
sqrt(E_LV)). An electron radiates vacuum Cherenkov once its LIV term
exceeds its rest mass, E_th,e = (m_e^2)^(1/(n+2)) * E_LV^(n/(n+2)).

Above the pair-decay threshold the decay is electromagnetic-strength,
so the mean free path is microscopic compared with astrophysical
distances: a photon above threshold simply does not arrive. That is
why the survival of LHAASO's 1.42 PeV Crab photons sets a hard bound
on the superluminal scale (see constraints.py).

Photon splitting (gamma -> 3 gamma, n = 2) has no threshold; its rate
scales as E^5 / E_LV^4 (Jacobson, Liberati & Mattingly 2006; Maccione
et al. 2008), so the mean free path scales as E_LV^4 / E^5. The
normalisation is calibrated so that the mean free path equals the
Crab distance (2 kpc) for a 1 PeV photon at the published bound scale
E_LV,2 = 1e-3 M_Pl — i.e. the toy reproduces the published bound by
construction, and is a teaching model of *why* the bound is what it
is, not an independent derivation of it.

Verified limits (see constraints.py for the full citations)
----------------------------------------------------------
* Photon superluminal, n = 1: E_LV > 2.74e24 GeV (He & Ma, Phys. Lett.
  B 835, 137536 (2023), from the LHAASO 1.42 PeV photon); the LHAASO
  Collaboration's own number is E_LV > 1.42e24 GeV (PRL 128, 051102
  (2022)). Both 95% CL.
* Photon superluminal, n = 2: E_LV > 1e-3 M_Pl = 1.22e16 GeV (LHAASO
  Collaboration, PRL 128, 051102 (2022), 95% CL; photon splitting).
* Electron superluminal, n = 1: E_LV > 9.4e25 GeV (He & Ma 2023, from
  the Crab 2.3 PeV parent electron).

Honesty note: the decays above are real physics of the LIV model, but
the photons in the simulator are synthetic. Survival fractions are
what the model predicts for a given scale; the comparison with the
real LHAASO photons is done by constraints.check_superluminal.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
ELECTRON_MASS_GEV = 5.1099895e-4     # electron rest mass, GeV/c^2
E_PLANCK_GEV = 1.22e19               # Planck energy, GeV
KPC_IN_M = 3.085677581491367e19       # one kiloparsec, m
CRAB_DISTANCE_KPC = 2.0              # Crab Nebula distance, ~2 kpc

# Photon-splitting mean-free-path calibration (see module docstring):
# lambda = CRAB_DISTANCE at E = 1 PeV when E_LV,2 = 1e-3 M_Pl.
SPLIT_CAL_ENERGY_GEV = 1.0e6         # 1 PeV
SPLIT_CAL_E_LV_GEV = 1.0e-3 * E_PLANCK_GEV
SPLIT_CAL_DISTANCE_M = CRAB_DISTANCE_KPC * KPC_IN_M


def _threshold(mass_term_gev2, e_lv_gev, order):
    """Generic LIV threshold: E^(n+2)/E_LV^n = mass_term (GeV^2)."""
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if e_lv_gev <= 0:
        raise ValueError("e_lv_gev must be > 0")
    return float(mass_term_gev2)**(1.0 / (order + 2)) * float(
        e_lv_gev)**(order / (order + 2))


def pair_decay_threshold(e_lv_gev, order=1):
    """Threshold energy (GeV) for gamma -> e+ e- under superluminal LIV.

    E_th = (4 m_e^2)^(1/(n+2)) * E_LV^(n/(n+2)). A photon above this
    energy decays almost immediately; one below it cannot decay at all.
    """
    return _threshold(4.0 * ELECTRON_MASS_GEV**2, e_lv_gev, order)


def electron_cherenkov_threshold(e_lv_e_gev, order=1):
    """Threshold energy (GeV) for vacuum Cherenkov, e -> e + gamma.

    E_th,e = (m_e^2)^(1/(n+2)) * E_LV^(n/(n+2)). A superluminal
    electron above this energy radiates photons until it drops below
    threshold, so this is also the maximum energy an electron can
    keep over an astrophysical distance.
    """
    return _threshold(ELECTRON_MASS_GEV**2, e_lv_e_gev, order)


def electron_max_energy(e_lv_e_gev, order=1):
    """Maximum electron energy (GeV) survivable over astrophysical
    distances: identical to the vacuum-Cherenkov threshold, because
    an electron above threshold radiates itself down to it."""
    return electron_cherenkov_threshold(e_lv_e_gev, order=order)


def splitting_mean_free_path(energy_gev, e_lv_gev):
    """Mean free path (m) for photon splitting gamma -> 3 gamma (n = 2).

    Rate scales as E^5 / E_LV^4, so lambda scales as E_LV^4 / E^5,
    normalised so lambda = 2 kpc for a 1 PeV photon at the published
    bound scale E_LV,2 = 1e-3 M_Pl (see module docstring). n = 1 has
    no splitting channel in the standard treatment.
    """
    energy_gev = np.asarray(energy_gev, dtype=float)
    if np.any(energy_gev <= 0):
        raise ValueError("energy_gev must be > 0")
    if e_lv_gev <= 0:
        raise ValueError("e_lv_gev must be > 0")
    return SPLIT_CAL_DISTANCE_M * (float(e_lv_gev) / SPLIT_CAL_E_LV_GEV)**4 \
        * (SPLIT_CAL_ENERGY_GEV / energy_gev)**5


def photon_survival_probability(energy_gev, distance_m, e_lv_gev,
                                order=1):
    """Probability a superluminal-LIV photon survives a distance.

    * n = 1: 1 below the pair-decay threshold, 0 above it (the decay
      is electromagnetic-strength, so the mean free path above
      threshold is microscopic next to astrophysical distances).
    * n = 2: pair decay above threshold (0), plus photon splitting
      below threshold with survival exp(-D / lambda_split).
    Returns 0..1 for scalars and arrays.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    energy_gev = np.asarray(energy_gev, dtype=float)
    distance_m = float(distance_m)
    e_th = pair_decay_threshold(e_lv_gev, order=order)
    survives = np.where(energy_gev >= e_th, 0.0, 1.0)
    if order == 2:
        lam = splitting_mean_free_path(energy_gev, e_lv_gev)
        split_survival = np.exp(-distance_m / lam)
        survives = np.where(energy_gev >= e_th, 0.0, split_survival)
    return survives


def simulate_crab_survival(energies_gev, e_lv_gev, order=1,
                           distance_kpc=CRAB_DISTANCE_KPC):
    """Survival of Crab Nebula photons over ``distance_kpc`` (default 2).

    The Crab is the source of the 1.42 PeV photon that sets the
    superluminal bounds: any model in which that photon would not
    survive the trip is excluded. Returns a dict with the energies,
    the pair-decay threshold, and the survival probability per energy.
    """
    energies_gev = np.asarray(energies_gev, dtype=float)
    distance_m = float(distance_kpc) * KPC_IN_M
    survival = photon_survival_probability(energies_gev, distance_m,
                                           e_lv_gev, order=order)
    return {
        "energies_gev": energies_gev,
        "distance_kpc": float(distance_kpc),
        "e_lv_gev": float(e_lv_gev),
        "order": int(order),
        "pair_decay_threshold_gev": pair_decay_threshold(e_lv_gev,
                                                         order=order),
        "survival_probability": survival,
    }
