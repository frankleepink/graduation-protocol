"""Tests for gph_qgp.dispersion."""

import numpy as np

from gph_qgp import dispersion as d


def test_delay_sign_positive_eta():
    # Positive eta delays high-energy photons; delay grows with energy.
    e = np.array([0.1, 1.0, 10.0, 100.0])
    delays = d.time_delay(e, 1000.0, eta=1.0)
    assert np.all(delays > 0)
    assert np.all(np.diff(delays) > 0)


def test_delay_sign_negative_eta():
    delays = d.time_delay(np.array([1.0, 10.0]), 1000.0, eta=-1.0)
    assert np.all(delays < 0)


def test_delay_zero_eta_or_zero_distance():
    assert d.time_delay(10.0, 1000.0, eta=0.0) == 0.0
    assert d.time_delay(10.0, 0.0, eta=1.0) == 0.0


def test_delay_magnitude_order():
    # 10 GeV photon, 1 Gpc, eta=1: ~(10/1.22e19) * (3.09e25/3e8) ~ 0.08 s
    delay = float(d.time_delay(10.0, 1000.0, eta=1.0))
    assert 0.01 < delay < 1.0


def test_group_velocity_subluminal_positive_eta():
    v = d.group_velocity_fraction(1e17, eta=1.0)
    assert 0.0 < v < 1.0


def test_dispersion_relation_on_shell():
    # On the modified mass shell the residual is ~0 for the consistent p.
    e = 10.0
    # p from E^2 = p^2 c^2 + eta E^3/E_QG  ->  p = sqrt(E^2 - eta E^3/E_QG)
    p = np.sqrt(e**2 - 1.0 * e**3 / d.E_PLANCK_GEV)
    assert abs(d.dispersion_relation(e, p, eta=1.0)) < 1e-6


def test_eta_recovery_within_tolerance():
    energies = np.logspace(-1, 2, 40)
    sim = d.simulate_grb(energies, 1000.0, eta=1.0,
                         emission_jitter_s=0.02, seed=0)
    fit = d.fit_eta(sim["energies_gev"], sim["arrival_s"], 1000.0)
    assert abs(fit["eta"] - 1.0) < 0.2  # within 20% of the true value
    assert fit["eta_err"] > 0
    assert fit["r_value"] > 0.9


def test_eta_recovery_negative_eta():
    energies = np.logspace(-1, 2, 40)
    sim = d.simulate_grb(energies, 1000.0, eta=-0.5,
                         emission_jitter_s=0.01, seed=3)
    fit = d.fit_eta(sim["energies_gev"], sim["arrival_s"], 1000.0)
    assert abs(fit["eta"] - (-0.5)) < 0.2
