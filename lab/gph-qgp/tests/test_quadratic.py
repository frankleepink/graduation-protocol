"""Tests for quadratic dispersion (v0.3) and the dispersion checks."""

import numpy as np
import pytest

from gph_qgp import constraints, dispersion as d


def test_delay_n2_scales_as_energy_squared():
    delays = d.time_delay(np.array([1.0, 10.0]), 1000.0, eta=1.0, order=2)
    assert delays[1] / delays[0] == pytest.approx(100.0)


def test_delay_n2_much_smaller_than_n1_at_geV():
    # At 100 GeV the quadratic term is suppressed by another E/E_QG.
    d2 = float(d.time_delay(100.0, 1000.0, eta=1.0, order=2))
    d1 = float(d.time_delay(100.0, 1000.0, eta=1.0, order=1))
    assert d2 < 1e-15
    assert d2 / d1 == pytest.approx(1.5 * 100.0 / d.E_PLANCK_GEV)


def test_delay_n2_prefactor_is_3_over_2():
    # delta_t = (3/2) * eta * (E/E_QG)^2 * D/c at n=2.
    expected = 1.5 * (10.0 / d.E_PLANCK_GEV) ** 2 * 1000.0 * d.MPC_IN_M \
        / d.C_M_PER_S
    assert float(d.time_delay(10.0, 1000.0, eta=1.0, order=2)) == \
        pytest.approx(expected)


def test_group_velocity_n2():
    v = float(d.group_velocity_fraction(1e17, eta=1.0, order=2))
    expected = 1.0 - 1.5 * (1e17 / d.E_PLANCK_GEV) ** 2
    assert v == pytest.approx(expected)
    assert 0.0 < v < 1.0


def test_dispersion_relation_on_shell_n2():
    e = 10.0
    p = np.sqrt(e**2 - e**4 / d.E_PLANCK_GEV**2)
    assert abs(d.dispersion_relation(e, p, eta=1.0, order=2)) < 1e-6


def test_eta_recovery_n2_with_low_scale():
    # At the Planck scale the n=2 delays are far below any jitter, so
    # use a low scale (1e12 GeV) where the delays are measurable.
    energies = np.logspace(-1, 2, 40)
    sim = d.simulate_grb(energies, 1000.0, eta=1.0, e_qg_gev=1e12,
                         emission_jitter_s=1e-4, seed=0, order=2)
    fit = d.fit_eta(sim["energies_gev"], sim["arrival_s"], 1000.0,
                    e_qg_gev=1e12, order=2)
    assert abs(fit["eta"] - 1.0) < 0.2
    assert fit["r_value"] > 0.9


def test_bad_order_raises():
    with pytest.raises(ValueError):
        d.time_delay(10.0, 1000.0, order=3)
    with pytest.raises(ValueError):
        d.fit_eta(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]),
                  1000.0, order=0)


def test_dispersion_quadratic_planck_scale_survives():
    """v0.3: the quadratic model is real now, and a Planck-scale term
    survives — the best quadratic limit (1.2e12 GeV) is four orders
    below E_Pl."""
    res = constraints.check_dispersion(eta=1.0, order=2)
    assert res["verdict"] == "SURVIVES"
    assert res["effective_scale_e_pl"] == pytest.approx(1.0)
    assert len(res["per_constraint"]) == 3
    assert all(p["verdict"] == "SURVIVES" for p in res["per_constraint"])


def test_dispersion_quadratic_low_scale_excluded():
    res = constraints.check_dispersion(eta=1.0, e_qg_gev=1e11, order=2)
    assert res["verdict"] == "EXCLUDED"
    assert res["effective_scale_gev"] == pytest.approx(1e11)
