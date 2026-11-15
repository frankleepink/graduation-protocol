"""Modified dispersion relations and GRB photon time-of-flight (v0.3).

Model
-----
Lorentz-invariance violation (LIV), Amelino-Camelia style. The photon
dispersion relation is modified at order n in E / E_QG::

    E^2 = p^2 c^2 + eta * E^(n+2) / E_QG^n        n = 1 or 2

where ``eta`` is a dimensionless coefficient (eta = +1 is the usual
benchmark), and E_QG is the quantum-gravity energy scale, taken equal to
the Planck energy E_P ~ 1.22e19 GeV unless the caller says otherwise.
v0.1/v0.2 had n = 1 only; v0.3 adds n = 2 (quadratic dispersion).

Consequences used here
-----------------------
* Group velocity (first order):  v_g / c ~ 1 - ((n+1)/2) * eta * (E/E_QG)^n.
  Positive eta makes high-energy photons travel *slower* (they arrive
  later); negative eta makes them arrive earlier.
* Arrival-time delay over a distance D (flat space, no cosmology)::

      delta_t ~ ((n+1)/2) * eta * (E / E_QG)^n * D / c

  The (n+1)/2 factor is the standard Vasileiou et al. (PRD 87, 122001,
  2013) form. It equals 1 at n = 1, so v0.1/v0.2 behaviour is unchanged,
  and 3/2 at n = 2.

Assumptions and limits (plain language)
----------------------------------------
* Flat, non-expanding space. Real gamma-ray bursts are cosmological:
  a proper treatment integrates over redshift with the expansion history.
  The delays here are order-of-magnitude teaching numbers, not fits to
  real GRB data.
* First order in the LIV term only. Terms of higher order are dropped.
* No intrinsic source effects: a real burst emits photons over a finite
  time with an energy-dependent light curve, which can mimic or hide a
  propagation delay. The simulator adds an optional Gaussian emission
  jitter to stand in for that, but it is a placeholder, not a source model.
* Energies are in GeV, distances in Mpc, times in seconds. E_QG in GeV.

Units: energy GeV, distance Mpc, time s, eta dimensionless.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- constants
C_M_PER_S = 299_792_458.0            # speed of light, m/s
MPC_IN_M = 3.0856775814913673e22      # one megaparsec, m
E_PLANCK_GEV = 1.22e19               # Planck energy, GeV (~1.22 x 10^19 GeV)
GEV_IN_J = 1.602176634e-10           # 1 GeV in joules


# ------------------------------------------------------------------- model
def group_velocity_fraction(energy_gev, eta=1.0, e_qg_gev=E_PLANCK_GEV,
                            order=1):
    """v_g / c for a photon of ``energy_gev`` under the order-n MDR.

    First-order result v_g/c ~ 1 - ((n+1)/2) * eta * (E / E_QG)^n.
    Positive eta means subluminal (slower than c); negative eta means
    superluminal. At n = 1 this is exactly the v0.1 formula.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    energy_gev = np.asarray(energy_gev, dtype=float)
    prefactor = (order + 1) / 2.0
    return 1.0 - prefactor * eta * (
        energy_gev / float(e_qg_gev))**order


def time_delay(energy_gev, distance_mpc, eta=1.0, e_qg_gev=E_PLANCK_GEV,
               order=1):
    """Propagation delay (seconds) of a photon relative to a c-speed photon.

    delta_t = ((n+1)/2) * eta * (E / E_QG)^n * D / c (Vasileiou et al.
    2013 form). Sign follows eta: eta > 0 delays the photon, eta < 0
    advances it. Works on scalars and arrays. At n = 1 the prefactor
    is 1, so this is identical to the v0.1 formula.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    energy_gev = np.asarray(energy_gev, dtype=float)
    distance_m = float(distance_mpc) * MPC_IN_M
    prefactor = (order + 1) / 2.0
    return prefactor * eta * (
        energy_gev / float(e_qg_gev))**order * distance_m / C_M_PER_S


def dispersion_relation(energy_gev, momentum_gev_c, eta=1.0,
                        e_qg_gev=E_PLANCK_GEV, order=1):
    """Left-hand side minus right-hand side of E^2 = p^2 c^2 + eta E^(n+2)/E_QG^n.

    Returns E^2 - p^2 c^2 - eta E^(n+2) / E_QG^n in GeV^2; zero on the
    modified mass shell. (Here p is given in GeV/c so p^2 c^2 is in
    GeV^2.) At n = 1 this is exactly the v0.1 form.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    e = np.asarray(energy_gev, dtype=float)
    p = np.asarray(momentum_gev_c, dtype=float)
    return e**2 - p**2 - eta * e**(order + 2) / float(e_qg_gev)**order


# -------------------------------------------------------------- simulation
def simulate_grb(energies_gev, distance_mpc, eta=1.0,
                 e_qg_gev=E_PLANCK_GEV, emission_jitter_s=0.0, seed=0,
                 order=1):
    """Simulate observed GRB photon arrival times.

    Each photon is emitted at t = 0 (plus optional Gaussian emission
    jitter standing in for intrinsic source spread), then delayed by
    ``time_delay``. Returns a dict with the energies, the true delays,
    and the observed arrival times (all in seconds).

    Parameters
    ----------
    energies_gev : array of photon energies (GeV)
    distance_mpc : source distance (Mpc)
    eta : LIV coefficient used to generate the data
    emission_jitter_s : std of Gaussian intrinsic emission spread (s)
    seed : RNG seed (int)
    order : MDR order, 1 (linear) or 2 (quadratic); new in v0.3
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    rng = np.random.default_rng(seed)
    energies = np.asarray(energies_gev, dtype=float)
    delays = time_delay(energies, distance_mpc, eta=eta,
                        e_qg_gev=e_qg_gev, order=order)
    jitter = rng.normal(0.0, emission_jitter_s, size=energies.shape) \
        if emission_jitter_s > 0 else np.zeros_like(energies)
    return {
        "energies_gev": energies,
        "true_delay_s": delays,
        "emission_jitter_s": jitter,
        "arrival_s": delays + jitter,
        "order": int(order),
        "e_qg_gev": float(e_qg_gev),
    }


def fit_eta(energies_gev, arrival_s, distance_mpc, e_qg_gev=E_PLANCK_GEV,
            order=1):
    """Recover eta from simulated arrival times by linear regression.

    Fits arrival_s = a + b * E^order and converts the slope b into eta
    via b = ((order+1)/2) * eta * D / (c * E_QG^order). Returns a dict
    with eta, its standard error, the intercept, and the fit's r-value.
    This is the same regression a real GRB time-of-flight analysis does
    at first order; it cannot separate a propagation delay from an
    intrinsic energy-dependent emission lag in real data. At order 1
    this is identical to the v0.1 fit.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    energies = np.asarray(energies_gev, dtype=float)
    arrival = np.asarray(arrival_s, dtype=float)
    if energies.size < 3:
        raise ValueError("need at least 3 photons to fit eta")
    res = stats.linregress(energies**order, arrival)
    distance_m = float(distance_mpc) * MPC_IN_M
    prefactor = (order + 1) / 2.0
    slope_to_eta = prefactor * distance_m / (
        C_M_PER_S * float(e_qg_gev)**order)
    return {
        "eta": float(res.slope / slope_to_eta),
        "eta_err": float(res.stderr / slope_to_eta),
        "intercept_s": float(res.intercept),
        "r_value": float(res.rvalue),
        "p_value": float(res.pvalue),
        "n_photons": int(energies.size),
        "order": int(order),
    }
