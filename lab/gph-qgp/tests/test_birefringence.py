"""Tests for vacuum birefringence (v0.3)."""

import numpy as np
import pytest

from gph_qgp import birefringence as b, constraints


def test_rotation_angle_magnitude():
    # 400 keV photon, 1 Gpc, eta = 1e-15: ~2 rad (Wei et al. regime).
    angle = float(b.rotation_angle(4e-4, 1000.0, 1e-15))
    assert angle == pytest.approx(2.05, rel=0.05)


def test_rotation_scales_as_energy_squared():
    a1 = float(b.rotation_angle(2e-4, 1000.0, 1e-15))
    a2 = float(b.rotation_angle(4e-4, 1000.0, 1e-15))
    assert a2 / a1 == pytest.approx(4.0)


def test_rotation_scales_with_distance_and_eta():
    a1 = float(b.rotation_angle(4e-4, 1000.0, 1e-15))
    a2 = float(b.rotation_angle(4e-4, 2000.0, 2e-15))
    assert a2 / a1 == pytest.approx(4.0)


def test_rotation_difference_scrambles_band_at_bound():
    # 100-500 keV band, 1 Gpc, eta = 1e-15: spread > pi/2.
    spread = float(b.rotation_difference(1e-4, 5e-4, 1000.0, 1e-15))
    assert spread > np.pi / 2
    assert spread == pytest.approx(3.07, rel=0.05)


def test_band_polarization_limits():
    energies = np.linspace(0.1, 0.5, 25) / 1000.0
    assert b.band_polarization(energies, 1000.0, 0.0, p0=0.6) == \
        pytest.approx(0.6)
    # At eta = 1e-16 the wide band mostly survives; at the published
    # bound (1e-15) it is already substantially scrambled (0.13) —
    # the Wei et al. bound comes from energy-resolved fits of the
    # rotation vs energy, not from band-averaged survival, which is
    # why the bound can sit where the wide band is already washed out.
    assert b.band_polarization(energies, 1000.0, 1e-16, p0=0.6) > 0.5
    assert b.band_polarization(energies, 1000.0, 1e-15, p0=0.6) < 0.2
    assert b.band_polarization(energies, 1000.0, 1e-14, p0=0.6) < 0.1


def test_eta_limit_inverts_rotation():
    eta = b.eta_limit(2.05, 4e-4, 1000.0)
    assert eta == pytest.approx(1e-15, rel=0.05)


def test_simulate_grb_polarimetry_shape():
    energies = np.linspace(0.1, 0.5, 10) / 1000.0
    sim = b.simulate_grb_polarimetry(energies, 1000.0, 1e-15, p0=0.6)
    assert sim["rotation_rad"].shape == (10,)
    assert sim["rotation_spread_rad"] > 0
    assert 0.0 <= sim["polarization_fraction"] <= 0.6


def test_check_birefringence_verdicts():
    assert constraints.check_birefringence(eta=1.0)["verdict"] == "EXCLUDED"
    assert constraints.check_birefringence(eta=1e-14)["verdict"] == "EXCLUDED"
    assert constraints.check_birefringence(eta=1e-15)["verdict"] == "SURVIVES"
    assert constraints.check_birefringence(eta=1e-16)["verdict"] == "SURVIVES"
    assert constraints.check_birefringence(eta=-1e-14)["verdict"] == "EXCLUDED"
