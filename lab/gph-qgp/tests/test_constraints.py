"""Tests for the v0.3 real-limits constraints module and compare CLI."""

import os
import subprocess
import sys

import pytest

from gph_qgp import constraints


def test_every_entry_has_value_units_source():
    """Every constraint must carry a value, units and a verifiable link."""
    table = constraints.summary_table()
    assert len(table) >= 9
    for entry in table:
        assert entry["limit_value"] is not None, entry["id"]
        assert entry["limit_units"], entry["id"]
        assert entry["source"], entry["id"]
        assert entry["source_link"].startswith("http"), entry["id"]
        assert entry["confidence"], entry["id"]


def test_summary_table_is_a_copy():
    table = constraints.summary_table()
    table[0]["limit_value"] = -1.0
    assert constraints.summary_table()[0]["limit_value"] != -1.0


def test_get_constraint_unknown_raises():
    with pytest.raises(KeyError):
        constraints.get_constraint("no_such_limit")


# ------------------------------------------------------- dispersion checks
def test_dispersion_eta1_excluded_by_all_linear_limits():
    """eta = 1 at the Planck scale: effective 1 E_Pl, below every limit."""
    res = constraints.check_dispersion(eta=1.0, order=1)
    assert res["verdict"] == "EXCLUDED"
    assert res["effective_scale_e_pl"] == pytest.approx(1.0)
    assert len(res["per_constraint"]) == 4
    assert all(p["verdict"] == "EXCLUDED" for p in res["per_constraint"])


def test_dispersion_small_eta_survives():
    """eta = 0.05 -> effective 20 E_Pl, above the strongest limit (12.3)."""
    res = constraints.check_dispersion(eta=0.05, order=1)
    assert res["verdict"] == "SURVIVES"
    assert res["effective_scale_e_pl"] == pytest.approx(20.0)


def test_dispersion_eta_zero_survives():
    """eta = 0 means no effect at all: nothing to exclude."""
    res = constraints.check_dispersion(eta=0.0, order=1)
    assert res["verdict"] == "SURVIVES"


def test_dispersion_quadratic_planck_scale_survives():
    """v0.3: the quadratic model is real now. eta = 1 at the Planck
    scale survives — the best quadratic limit (1.2e12 GeV) is four
    orders below E_Pl. (v0.2 returned UNCONSTRAINED here because the
    model had no quadratic term; that is no longer true.)"""
    res = constraints.check_dispersion(eta=1.0, order=2)
    assert res["verdict"] == "SURVIVES"
    assert len(res["per_constraint"]) == 3
    assert all(p["verdict"] == "SURVIVES" for p in res["per_constraint"])


def test_dispersion_superluminal_unconstrained():
    res = constraints.check_dispersion(eta=-1.0, order=1)
    assert res["verdict"] == "UNCONSTRAINED"


def test_dispersion_bad_order_raises():
    with pytest.raises(ValueError):
        constraints.check_dispersion(eta=1.0, order=3)


# ------------------------------------------------------------- foam checks
def test_foam_random_walk_excluded_by_ligo():
    res = constraints.check_foam("random_walk")
    assert res["verdict"] == "EXCLUDED"
    # Predicted noise ~1e-16 m/sqrt(Hz) vs floor ~4e-20: ratio > 100.
    assert res["ratio"] > 100


def test_foam_holographic_unconstrained_by_ligo():
    res = constraints.check_foam("holographic")
    assert res["verdict"] == "UNCONSTRAINED"
    # RMS ~1e-22 m vs 1 Hz-band floor 4e-20 m: ratio well below 1.
    assert res["ratio"] < 0.1


def test_foam_unknown_model_raises():
    with pytest.raises(ValueError):
        constraints.check_foam("no_such_model")


def test_foam_40m_both_pass():
    """40-m data bounded parameters; Planck-scale models pass both."""
    for model in ("random_walk", "holographic"):
        res = constraints.check_foam_40m(model)
        assert res["verdict"] == "SURVIVES"
        assert res["model_length_scale_m"] < res["bound_m"]


def test_holometer_excludes_hogan_shear():
    res = constraints.check_holometer()
    assert res["verdict"] == "EXCLUDED"
    assert res["model"] == "hogan_shear_holographic"
    assert "Hogan" in res["note"]


def test_ligo_floor_value():
    # 1e-23 /sqrt(Hz) strain x 4000 m arm = 4e-20 m/sqrt(Hz).
    assert constraints.ligo_displacement_floor() == pytest.approx(4e-20)
    assert constraints.ligo_displacement_floor(1000.0) == pytest.approx(1e-20)


# -------------------------------------------------------------- CLI compare
def test_cli_compare(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src")
    result = subprocess.run(
        [sys.executable, "-m", "gph_qgp", "compare", "--out", str(tmp_path)],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "comparison.png").exists()
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    assert "EXCLUDED" in result.stdout
    assert "UNCONSTRAINED" in result.stdout
    assert "Random-walk foam" in result.stdout
    csv_text = (tmp_path / "comparison.csv").read_text()
    assert "dispersion_linear" in csv_text
    assert "foam_ligo_psd" in csv_text
    assert "holometer_shear" in csv_text
