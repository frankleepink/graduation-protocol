"""Tests for the GUP minimal-length module (v0.3)."""

import numpy as np
import pytest

from gph_qgp import constraints, gup


def test_minimal_length_planck():
    assert gup.minimal_length(1.0) == pytest.approx(gup.L_PLANCK_M)
    assert gup.minimal_length(0.0) == 0.0
    assert gup.minimal_length(4.0) == pytest.approx(2.0 * gup.L_PLANCK_M)
    with pytest.raises(ValueError):
        gup.minimal_length(-1.0)


def test_beta_si():
    # beta = beta_0 / (M_Pl c)^2; at beta_0 = 1 it is ~0.023 (kg m/s)^-2.
    assert gup.beta_si(1.0) == pytest.approx(0.0234, rel=0.01)
    assert gup.beta_si(2.0) == pytest.approx(2.0 * gup.beta_si(1.0))


def test_oscillator_levels_reduce_to_ordinary():
    omega = 2 * np.pi * 1e3
    ns = np.arange(0, 4)
    levels = gup.oscillator_levels(ns, omega, 1e-3, beta0=0.0)
    expected = gup.HBAR_J_S * omega * (ns + 0.5)
    assert np.allclose(levels, expected)


def test_level_shift_positive_and_grows_with_n():
    omega = 2 * np.pi * 1e3
    shifts = gup.oscillator_level_shift(np.arange(0, 4), omega, 1e-3,
                                        beta0=1.0)
    assert np.all(shifts > 0)
    assert np.all(np.diff(shifts) > 0)
    # Ground-state shift: (beta m hbar^2 w^2 / 4) * 1.
    expected = gup.beta_si(1.0) * 1e-3 * gup.HBAR_J_S**2 * omega**2 / 4.0
    assert shifts[0] == pytest.approx(expected)


def test_frequency_shift_scales_as_amplitude_squared():
    omega = 2 * np.pi * 1e4
    # 1 nm amplitude at the bound: a measurable ~2e-5 shift. (At a
    # 1 um amplitude the perturbative formula gives > 1, which is
    # outside its regime — the resonator experiments run at nm scale.)
    s1 = gup.frequency_shift_fraction(1e-9, 0.3, omega, beta0=5.2e6)
    s2 = gup.frequency_shift_fraction(2e-9, 0.3, omega, beta0=5.2e6)
    assert s2 / s1 == pytest.approx(4.0)
    assert gup.frequency_shift_fraction(0.0, 0.3, omega, 5.2e6) == 0.0
    assert 0.0 < s1 < 1.0
    assert s1 == pytest.approx(2.2e-5, rel=0.05)


def test_hawking_temperature():
    t_h = gup.hawking_temperature_hbar(1e12)
    assert gup.hawking_temperature(1e12, beta0=0.0) == pytest.approx(t_h)
    # GUP lowers the temperature; at beta_0 = 1 and 1e12 kg the
    # correction is negligible (M >> M_Pl).
    assert gup.hawking_temperature(1e12, beta0=1.0) == pytest.approx(
        t_h, rel=1e-6)
    # Below the remnant mass there is no temperature.
    assert gup.hawking_temperature(1e-9, beta0=1.0) is None
    assert gup.remnant_mass(1.0) == pytest.approx(gup.M_PLANCK_KG)


def test_check_gup_verdicts():
    assert constraints.check_gup(beta0=1.0)["verdict"] == "SURVIVES"
    assert constraints.check_gup(beta0=5.2e6)["verdict"] == "SURVIVES"
    assert constraints.check_gup(beta0=1e7)["verdict"] == "EXCLUDED"
