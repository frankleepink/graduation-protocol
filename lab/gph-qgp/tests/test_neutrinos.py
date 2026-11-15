"""Tests for neutrino time-of-flight (v0.3)."""

import numpy as np
import pytest

from gph_qgp import constraints, neutrinos as nu


def test_sn1987a_delay_at_published_bound():
    # 10 MeV neutrino, M_1 = 2.7e10 GeV (the SN1987A bound): ~2 s.
    delay = float(nu.sn1987a_delay(10.0, 2.7e10, order=1))
    assert delay == pytest.approx(1.96, rel=0.05)


def test_sn1987a_delay_scales_with_energy():
    d1 = float(nu.sn1987a_delay(10.0, 2.7e10, order=1))
    d2 = float(nu.sn1987a_delay(20.0, 2.7e10, order=1))
    assert d2 / d1 == pytest.approx(2.0)
    d1_q = float(nu.sn1987a_delay(10.0, 4.6e4, order=2))
    d2_q = float(nu.sn1987a_delay(20.0, 4.6e4, order=2))
    assert d2_q / d1_q == pytest.approx(4.0)


def test_sn1987a_spread_at_bound_within_observed_burst():
    # At the published M_1 bound the 7.5-40 MeV spread is ~6 s,
    # inside the observed ~13 s burst — consistent, not excluded.
    sim = nu.simulate_sn1987a(np.array([7.5, 40.0]), 2.7e10, order=1)
    assert sim["spread_s"] == pytest.approx(6.4, rel=0.05)
    assert sim["spread_s"] < 13.0


def test_speed_deviation_matches_sn1987a_bound():
    # At the M_1 bound the 10 MeV deviation is ~3.7e-13, far below 4e-9.
    delta = float(nu.speed_deviation(0.01, 2.7e10, order=1))
    assert delta == pytest.approx(3.7e-13, rel=0.05)
    assert nu.speed_fraction(0.01, 2.7e10, order=1) < 1.0


def test_planck_scale_spread_is_negligible():
    sim = nu.simulate_sn1987a(np.linspace(7.5, 40.0, 10),
                               nu.E_PLANCK_GEV, order=1)
    assert sim["spread_s"] < 1e-6


def test_txs_delay_scales_with_distance():
    # Same energy and scale, TXS (1.3 Gpc) delays far more than
    # SN1987A (51.4 kpc).
    d_sn = float(nu.sn1987a_delay(10.0, 1e14, order=1))
    d_txs = float(nu.txs0506_delay(0.01, 1e14, order=1))
    assert d_txs / d_sn > 1e4


def test_check_neutrino_subluminal_verdicts():
    assert constraints.check_neutrino(
        nu.E_PLANCK_GEV, order=1, branch="subluminal")["verdict"] == "SURVIVES"
    assert constraints.check_neutrino(
        nu.E_PLANCK_GEV, order=2, branch="subluminal")["verdict"] == "SURVIVES"
    # Below the TXS n=1 bound (1e14 GeV): excluded.
    assert constraints.check_neutrino(
        1e13, order=1, branch="subluminal")["verdict"] == "EXCLUDED"
    # Between SN1987A (2.7e10) and TXS (1e14): still excluded by TXS.
    assert constraints.check_neutrino(
        1e12, order=1, branch="subluminal")["verdict"] == "EXCLUDED"
    # Below the TXS n=2 bound (1e9 GeV): excluded.
    assert constraints.check_neutrino(
        1e8, order=2, branch="subluminal")["verdict"] == "EXCLUDED"


def test_check_neutrino_superluminal_verdicts():
    assert constraints.check_neutrino(
        nu.E_PLANCK_GEV, order=1,
        branch="superluminal")["verdict"] == "EXCLUDED"
    assert constraints.check_neutrino(
        1e25, order=1, branch="superluminal")["verdict"] == "SURVIVES"
    assert constraints.check_neutrino(
        1e25, order=2, branch="superluminal")["verdict"] == "UNCONSTRAINED"


def test_check_neutrino_speed_verdicts():
    assert constraints.check_neutrino_speed(1e-8)["verdict"] == "EXCLUDED"
    assert constraints.check_neutrino_speed(1e-9)["verdict"] == "SURVIVES"
    assert constraints.check_neutrino_speed(-5e-9)["verdict"] == "EXCLUDED"
