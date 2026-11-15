"""Tests for the superluminal branch (v0.3)."""

import numpy as np
import pytest

from gph_qgp import constraints, superluminal as s


def test_pair_decay_threshold_n1_matches_lhaaso_bound():
    # At the published bound E_LV = 2.74e24 GeV the threshold is the
    # observed 1.42 PeV photon: the bound and the formula must agree.
    th = s.pair_decay_threshold(2.74e24, order=1)
    assert th == pytest.approx(1.42e6, rel=0.01)


def test_pair_decay_threshold_n2():
    th = s.pair_decay_threshold(1.22e16, order=2)
    assert th == pytest.approx(3.5e6, rel=0.02)


def test_threshold_grows_with_scale():
    assert s.pair_decay_threshold(1e25, order=1) > \
        s.pair_decay_threshold(1e24, order=1)
    assert s.pair_decay_threshold(1e17, order=2) > \
        s.pair_decay_threshold(1e16, order=2)


def test_electron_cherenkov_threshold():
    # At the published electron bound the threshold is ~2.9 PeV, just
    # above the Crab's 2.3 PeV parent electron.
    th = s.electron_cherenkov_threshold(9.4e25, order=1)
    assert 2.3e6 < th < 3.5e6
    assert s.electron_max_energy(9.4e25, order=1) == pytest.approx(th)


def test_survival_below_threshold_is_one_n1():
    # At E_Pl the n=1 threshold is ~23 TeV; a 10 TeV photon is below
    # it and survives.
    surv = s.photon_survival_probability(
        1e4, 2.0 * s.KPC_IN_M, s.E_PLANCK_GEV, order=1)
    assert float(surv) == pytest.approx(1.0)


def test_survival_above_threshold_is_zero_n1():
    # At E_Pl the n=1 threshold is ~23 TeV; a 1.42 PeV photon decays.
    surv = s.photon_survival_probability(
        1.42e6, 2.0 * s.KPC_IN_M, s.E_PLANCK_GEV, order=1)
    assert float(surv) == 0.0


def test_survival_n2_planck_scale_pev_photon_survives():
    # n=2 threshold at E_Pl is ~112 PeV, and splitting is negligible
    # at 1.42 PeV for a Planck-scale E_LV.
    surv = s.photon_survival_probability(
        1.42e6, 2.0 * s.KPC_IN_M, s.E_PLANCK_GEV, order=2)
    assert float(surv) > 0.999


def test_splitting_mean_free_path_scaling():
    lam1 = float(s.splitting_mean_free_path(1e6, 1.22e16))
    lam2 = float(s.splitting_mean_free_path(2e6, 1.22e16))
    # lambda ~ E^-5: doubling the energy cuts lambda by 2^5.
    assert lam2 / lam1 == pytest.approx(2.0 ** -5)
    # Calibration: 2 kpc at 1 PeV for the bound scale.
    assert lam1 == pytest.approx(2.0 * s.KPC_IN_M)


def test_simulate_crab_survival_shape():
    sim = s.simulate_crab_survival(np.logspace(3, 6, 10),
                                   s.E_PLANCK_GEV, order=1)
    assert sim["pair_decay_threshold_gev"] == pytest.approx(
        s.pair_decay_threshold(s.E_PLANCK_GEV, order=1))
    assert sim["survival_probability"].shape == (10,)
    assert np.all((sim["survival_probability"] >= 0)
                  & (sim["survival_probability"] <= 1))


def test_check_superluminal_photon_verdicts():
    # Planck-scale n=1 is killed by the 1.42 PeV photon; n=2 survives.
    assert constraints.check_superluminal_photon(
        s.E_PLANCK_GEV, order=1)["verdict"] == "EXCLUDED"
    assert constraints.check_superluminal_photon(
        s.E_PLANCK_GEV, order=2)["verdict"] == "SURVIVES"
    # Above the strictest n=1 bound, the model survives.
    assert constraints.check_superluminal_photon(
        1e25, order=1)["verdict"] == "SURVIVES"
    # Below the n=2 splitting bound, it is excluded.
    assert constraints.check_superluminal_photon(
        1e15, order=2)["verdict"] == "EXCLUDED"


def test_check_superluminal_electron_verdicts():
    assert constraints.check_superluminal_electron(
        s.E_PLANCK_GEV, order=1)["verdict"] == "EXCLUDED"
    assert constraints.check_superluminal_electron(
        1e26, order=1)["verdict"] == "SURVIVES"
    assert constraints.check_superluminal_electron(
        1e26, order=2)["verdict"] == "UNCONSTRAINED"


def test_bad_order_raises():
    with pytest.raises(ValueError):
        s.pair_decay_threshold(1e24, order=3)
