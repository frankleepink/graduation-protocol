"""Tests for the trans-Planckian sweep (trans_planckian module)."""

import numpy as np
import pytest

from gph_qgp import trans_planckian as tp


def test_effective_scale():
    assert tp.effective_scale_gev(1e19, eta=1.0, order=1) == pytest.approx(1e19)
    # eta=8, n=2 -> E_QG / sqrt(8)
    assert tp.effective_scale_gev(8e18, eta=8.0, order=2) == pytest.approx(
        8e18 / np.sqrt(8.0)
    )
    with pytest.raises(ValueError):
        tp.effective_scale_gev(-1.0)
    with pytest.raises(ValueError):
        tp.effective_scale_gev(1e19, eta=0.0)


def test_classify_boundaries():
    # eff scale below the bound -> EXCLUDED
    assert tp.classify_scale(10.0, 1.0)["verdict"] == "EXCLUDED"
    # eff scale far above the bound -> UNCONSTRAINED (ratio < 1e-3 floor)
    assert tp.classify_scale(1.0, 1e6)["verdict"] == "UNCONSTRAINED"
    # eff scale just above the bound -> SURVIVES
    assert tp.classify_scale(1.0, 2.0)["verdict"] == "SURVIVES"


def test_dispersion_delay_sign_and_scale():
    # superluminal (eta=+1) at z=0.9, 100 GeV, E_QG = Planck -> small positive delay
    dt = tp.dispersion_delay_s(100.0, z=0.9, e_qg_gev=tp.E_PLANCK_GEV, order=1)
    assert dt > 0
    # delay grows with energy and falls with E_QG
    assert tp.dispersion_delay_s(200.0, z=0.9, e_qg_gev=tp.E_PLANCK_GEV, order=1) > dt
    assert (
        tp.dispersion_delay_s(100.0, z=0.9, e_qg_gev=10 * tp.E_PLANCK_GEV, order=1)
        < dt
    )


def test_photon_decay_threshold():
    # higher suppression scale -> higher decay threshold
    lo = tp.photon_decay_threshold_gev(order=1, scale_gev=tp.E_PLANCK_GEV)
    hi = tp.photon_decay_threshold_gev(order=1, scale_gev=100 * tp.E_PLANCK_GEV)
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo
    # no superluminal branch -> no decay
    assert tp.photon_decay_threshold_gev(order=1, scale_gev=tp.E_PLANCK_GEV, eta_super=0.0) == float("inf")


def test_study_sweep_shape_and_extremes():
    rows = tp.run_trans_planckian_study()
    assert len(rows) == 135  # 45 scales x 3 bounds
    first = [r for r in rows if r["index"] == 0]
    last = [r for r in rows if r["index"] == 44]
    assert all(r["verdict"] == "EXCLUDED" for r in first)
    assert {r["constraint"]: r["verdict"] for r in last} == {
        "Linear dispersion (n=1)": "SURVIVES",
        "Birefringence (n=1)": "SURVIVES",
        "Superluminal decay (n=1)": "UNCONSTRAINED",
    }


def test_fit_recovers_true_scale():
    energies = np.geomspace(1.0, 1000.0, 12)
    sim = tp.simulate_dispersion_arrival_times(energies, e_qg_true_gev=2e17)
    fit = tp.fit_dispersion_log_scale(sim["energies_gev"], sim["t_obs_s"], sim["sigma_s"])
    assert fit["log10_e_qg_fit_gev"] == pytest.approx(np.log10(2e17), abs=0.5)
    assert fit["reduced_chi2"] < 5.0
    verdict = tp.evaluate_fitted_eqg_verdict(
        fit["log10_e_qg_fit_gev"], fit["log10_e_qg_err"]
    )
    assert verdict["center_verdict"] == "EXCLUDED"


def test_obs_at_emax_matches_workbench():
    # Values from the QGP Study Workbench export (qgp_raw_sweep_n1.csv),
    # reproduced exactly by this package's own physics at Emax=5000 GeV, z=0.9.
    rows = tp.run_trans_planckian_study()
    first = {r["constraint"]: r for r in rows if r["index"] == 0}
    assert first["Linear dispersion (n=1)"]["obs_at_emax"] == pytest.approx(1870.05, rel=1e-4)
    assert first["Birefringence (n=1)"]["obs_at_emax"] == pytest.approx(4.069584e32, rel=1e-4)
    assert first["Superluminal decay (n=1)"]["obs_at_emax"] == pytest.approx(10844.07, rel=1e-4)


def test_lever_arm_n1_uncertainty_shrinks():
    rows = tp.lever_arm_analysis()
    n1 = [r for r in rows if r["order"] == 1]
    errs = [r["log10_e_qg_err"] for r in n1]
    assert all(np.isfinite(errs))
    assert errs[0] > 0.5  # ~0.8 dex at 100 GeV: near the noise floor
    assert errs[-1] < 0.01  # ~0.002 dex at 10 TeV: lever arm works
    assert all(b < a for a, b in zip(errs, errs[1:]))  # strictly shrinking
    assert n1[-1]["log10_e_qg_fit_gev"] == pytest.approx(21.0, abs=0.05)


def test_lever_arm_n2_unidentifiable():
    rows = tp.lever_arm_analysis()
    n2 = [r for r in rows if r["order"] == 2]
    # quadratic signal is far below the 0.05 s noise at every GeV Emax:
    # uncertainty is infinite throughout -- the honest finding, not a bug.
    assert all(r["log10_e_qg_err"] == float("inf") for r in n2)
