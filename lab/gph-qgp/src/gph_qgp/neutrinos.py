"""Neutrino time-of-flight under Lorentz-invariance violation (v0.3).

Neutrinos are the other long-baseline clock. SN1987A gave us a dozen
neutrinos from 51.4 kpc arriving within seconds of each other and of
the light — a bound on any energy-dependent neutrino speed that has
stood for forty years — and IceCube's PeV neutrinos extend the same
test to energies a hundred million times higher.

Model
-----
Energy-dependent neutrino speed (subluminal branch, velocity-difference
convention used by the neutrino papers):

    v / c = 1 - (E / M_QG)^n ,        delta_t = (D / c) * (E / M_QG)^n

with M_QG the LIV scale for neutrinos and n = 1 or 2. Note the
convention: the photon GRB papers (dispersion.py) carry a (n+1)/2
factor in the delay; the neutrino bounds below are quoted in this
velocity-difference convention, and the checks compare like with
like. For the superluminal branch the same formulas apply with the
sign flipped; the IceCube PeV bound is a superluminal one (a
superluminal neutrino would lose energy by electron-positron pair
bremsstrahlung, nu -> nu e+ e-, and never arrive at PeV energies).

Verified limits (see constraints.py for the full citations)
----------------------------------------------------------
* SN1987A (subluminal): M_1 > 2.7e10 GeV, M_2 > 4.6e4 GeV
  (Bustamante, Ellis, Konoplich & Sakharov, arXiv:2408.15949,
  summarising the Longo 1988 / Stodolsky 1988 SN1987A analyses).
* SN1987A absolute speed: |v - c| / c < 4e-9 (Longo 1988; Stodolsky
  1988; as summarised in Borriello et al., PRD 87, 116009 (2013)).
* IceCube PeV (superluminal): delta < O(1e-18), i.e. M_QG > ~1e5 M_Pl
  at n = 1 (Borriello, Chakraborty & Mirizzi, PRD 87, 116009 (2013),
  arXiv:1303.5843).
* TXS 0506+056 flare (subluminal): M_1 > 1e14 GeV, M_2 > 1e9 GeV
  (Bustamante et al., arXiv:2408.15949).

Honesty note: the neutrinos in the simulator are synthetic, and the
SN1987A energies used here (7.5-40 MeV) are the published observed
range. The check against the real bounds is constraints.check_neutrino.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
C_M_PER_S = 299_792_458.0            # speed of light, m/s
MPC_IN_M = 3.085677581491367e22       # one megaparsec, m
KPC_IN_M = 3.085677581491367e19       # one kiloparsec, m
E_PLANCK_GEV = 1.22e19               # Planck energy, GeV

SN1987A_DISTANCE_KPC = 51.4          # Large Magellanic Cloud, kpc
TXS0506_DISTANCE_MPC = 1300.0       # TXS 0506+056, z = 0.3365, ~1.3 Gpc
SN1987A_ENERGY_RANGE_MEV = (7.5, 40.0)  # observed neutrino energies


def speed_fraction(energy_gev, m_qg_gev, order=1):
    """Neutrino speed v / c under subluminal LIV.

    v/c = 1 - (E / M_QG)^n. Works on scalars and arrays.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    energy_gev = np.asarray(energy_gev, dtype=float)
    return 1.0 - (energy_gev / float(m_qg_gev))**order


def speed_deviation(energy_gev, m_qg_gev, order=1):
    """Fractional speed deviation (c - v) / c = (E / M_QG)^n.

    Positive for subluminal. Compare with the SN1987A absolute bound
    |delta| < 4e-9: any model predicting a larger deviation at
    SN1987A energies is excluded. Computed directly as (E/M)^n, not
    as 1 - speed_fraction — the latter loses the answer to floating
    point cancellation when the deviation is below ~1e-16.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    energy_gev = np.asarray(energy_gev, dtype=float)
    return (energy_gev / float(m_qg_gev))**order


def time_delay(energy_gev, distance_mpc, m_qg_gev, order=1):
    """LIV time delay (s) of a neutrino relative to a light-speed one.

    delta_t = (D / c) * (E / M_QG)^n. Flat space; for cosmological
    distances (TXS 0506+056) this is an order-of-magnitude number,
    not a redshift-integrated fit.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    energy_gev = np.asarray(energy_gev, dtype=float)
    distance_m = float(distance_mpc) * MPC_IN_M
    return distance_m / C_M_PER_S * (
        energy_gev / float(m_qg_gev))**order


def sn1987a_delay(energy_mev, m_qg_gev, order=1):
    """LIV delay (s) of an SN1987A neutrino of ``energy_mev`` (MeV).

    Distance 51.4 kpc. The observed burst lasted ~13 s across
    7.5-40 MeV; a model predicting a much larger energy-dependent
    spread than that is excluded.
    """
    energy_gev = np.asarray(energy_mev, dtype=float) / 1000.0
    distance_mpc = SN1987A_DISTANCE_KPC * 1.0e-3
    return time_delay(energy_gev, distance_mpc, m_qg_gev, order=order)


def txs0506_delay(energy_gev, m_qg_gev, order=1):
    """LIV delay (s) of a TXS 0506+056 neutrino, distance ~1.3 Gpc."""
    return time_delay(energy_gev, TXS0506_DISTANCE_MPC, m_qg_gev,
                      order=order)


def simulate_sn1987a(energies_mev, m_qg_gev, order=1):
    """Simulate SN1987A neutrino arrival delays under LIV.

    Returns the energies, the per-energy delay, and the total
    energy-dependent spread (max delay minus min delay) across the
    sample — the quantity the SN1987A bound is really about.
    """
    energies_mev = np.asarray(energies_mev, dtype=float)
    delays = sn1987a_delay(energies_mev, m_qg_gev, order=order)
    return {
        "energies_mev": energies_mev,
        "delays_s": delays,
        "spread_s": float(delays.max() - delays.min()),
        "m_qg_gev": float(m_qg_gev),
        "order": int(order),
        "distance_kpc": SN1987A_DISTANCE_KPC,
    }
