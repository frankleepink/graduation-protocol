"""Modified dispersion relations and GRB photon time-of-flight.

Model
-----
Linear Lorentz-invariance violation (LIV), Amelino-Camelia style. The
photon dispersion relation is modified at first order in E / E_QG::

    E^2 = p^2 c^2 + eta * E^3 / E_QG        (photons, m = 0)

where ``eta`` is a dimensionless coefficient (eta = +1 is the usual
benchmark), and E_QG is the quantum-gravity energy scale, taken equal to
the Planck energy E_P ~ 1.22e19 GeV unless the caller says otherwise.

Consequences used here
-----------------------
* Group velocity (first order):  v_g / c ~ 1 - eta * E / E_QG.
  Positive eta makes high-energy photons travel *slower* (they arrive
  later); negative eta makes them arrive earlier.
* Arrival-time delay over a distance D (flat space, no cosmology)::

      delta_t ~ eta * (E / E_QG) * D / c

Assumptions and limits (plain language)
----------------------------------------
* Flat, non-expanding space. Real gamma-ray bursts are cosmological:
  a proper treatment integrates over redshift with the expansion history.
  The delays here are order-of-magnitude teaching numbers, not fits to
  real GRB data.
* First order in E / E_QG only. Terms of order (E / E_QG)^2 are dropped.
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
def group_velocity_fraction(energy_gev, eta=1.0, e_qg_gev=E_PLANCK_GEV):
    """v_g / c for a photon of ``energy_gev`` under the linear MDR.

    First-order result v_g/c ~ 1 - eta * E / E_QG. Positive eta means
    subluminal (slower than c); negative eta means superluminal.
    """
    energy_gev = np.asarray(energy_gev, dtype=float)
    return 1.0 - eta * energy_gev / float(e_qg_gev)


def time_delay(energy_gev, distance_mpc, eta=1.0, e_qg_gev=E_PLANCK_GEV):
    """Propagation delay (seconds) of a photon relative to a c-speed photon.

    delta_t = eta * (E / E_QG) * D / c. Sign follows eta: eta > 0 delays
    the photon, eta < 0 advances it. Works on scalars and arrays.
    """
    energy_gev = np.asarray(energy_gev, dtype=float)
    distance_m = float(distance_mpc) * MPC_IN_M
    return eta * (energy_gev / float(e_qg_gev)) * distance_m / C_M_PER_S


def dispersion_relation(energy_gev, momentum_gev_c, eta=1.0,
                        e_qg_gev=E_PLANCK_GEV):
    """Left-hand side minus right-hand side of E^2 = p^2 c^2 + eta E^3/E_QG.

    Returns E^2 - p^2 c^2 - eta E^3 / E_QG in GeV^2; zero on the modified
    mass shell. (Here p is given in GeV/c so p^2 c^2 is in GeV^2.)
    """
    e = np.asarray(energy_gev, dtype=float)
    p = np.asarray(momentum_gev_c, dtype=float)
    return e**2 - p**2 - eta * e**3 / float(e_qg_gev)


# -------------------------------------------------------------- simulation
def simulate_grb(energies_gev, distance_mpc, eta=1.0,
                 e_qg_gev=E_PLANCK_GEV, emission_jitter_s=0.0, seed=0):
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
    """
    rng = np.random.default_rng(seed)
    energies = np.asarray(energies_gev, dtype=float)
    delays = time_delay(energies, distance_mpc, eta=eta, e_qg_gev=e_qg_gev)
    jitter = rng.normal(0.0, emission_jitter_s, size=energies.shape) \
        if emission_jitter_s > 0 else np.zeros_like(energies)
    return {
        "energies_gev": energies,
        "true_delay_s": delays,
        "emission_jitter_s": jitter,
        "arrival_s": delays + jitter,
    }


def fit_eta(energies_gev, arrival_s, distance_mpc, e_qg_gev=E_PLANCK_GEV):
    """Recover eta from simulated arrival times by linear regression.

    Fits arrival_s = a + b * E and converts the slope b into eta via
    b = eta * D / (c * E_QG). Returns a dict with eta, its standard
    error, the intercept, and the fit's r-value. This is the same
    regression a real GRB time-of-flight analysis does at first order;
    it cannot separate a propagation delay from an intrinsic
    energy-dependent emission lag in real data.
    """
    energies = np.asarray(energies_gev, dtype=float)
    arrival = np.asarray(arrival_s, dtype=float)
    if energies.size < 3:
        raise ValueError("need at least 3 photons to fit eta")
    res = stats.linregress(energies, arrival)
    distance_m = float(distance_mpc) * MPC_IN_M
    slope_to_eta = distance_m / (C_M_PER_S * float(e_qg_gev))
    return {
        "eta": float(res.slope / slope_to_eta),
        "eta_err": float(res.stderr / slope_to_eta),
        "intercept_s": float(res.intercept),
        "r_value": float(res.rvalue),
        "p_value": float(res.pvalue),
        "n_photons": int(energies.size),
    }
