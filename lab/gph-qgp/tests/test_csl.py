"""Tests for CSL collapse phenomenology (v0.3)."""

import pytest

from gph_qgp import collapse, constraints


def test_csl_rate_coherent_scaling():
    # 1e-12 kg body, smaller than r_C: Lambda = lambda * N^2.
    rate = collapse.csl_rate(1e-12)
    n = 1e-12 / collapse.NUCLEON_MASS_KG
    assert rate == pytest.approx(1e-16 * n**2, rel=0.01)
    # Doubling the mass quadruples the rate (coherent N^2).
    assert collapse.csl_rate(2e-12) == pytest.approx(4.0 * rate)


def test_csl_rate_large_body_reduced():
    # A 1 um body at r_C = 1e-7 m has (1e-6/1e-7)^3 = 1000 cells,
    # so the rate is 1000x smaller than the fully coherent value.
    rate_small = collapse.csl_rate(1e-12)
    rate_large = collapse.csl_rate(1e-12, size_m=1e-6)
    assert rate_large == pytest.approx(rate_small / 1000.0)


def test_csl_heating_power():
    # dE/dt = (3 lambda hbar^2 / (4 m_0 r_C^2)) * N.
    power = collapse.csl_heating_power(1e-12)
    assert power == pytest.approx(3.0e-29, rel=0.05)
    assert collapse.csl_heating_power(2e-12) == pytest.approx(2.0 * power)


def test_check_csl_grw_values_excluded_by_xenonnt():
    """The newest X-ray bound kills the textbook GRW values, while
    Majorana and LISA Pathfinder alone would pass them."""
    res = constraints.check_csl(1e-16, 1e-7)
    assert res["verdict"] == "EXCLUDED"
    by_id = {p["constraint_id"]: p["verdict"]
             for p in res["per_constraint"]}
    assert by_id["csl_xray_xenonnt2025"] == "EXCLUDED"
    assert by_id["csl_xray_majorana2022"] == "SURVIVES"
    assert by_id["csl_lisa_pathfinder_carlesso2016"] == "SURVIVES"
    assert res["lambda_over_rc2"] == pytest.approx(1e-2)


def test_check_csl_smaller_lambda_survives():
    res = constraints.check_csl(1e-17, 1e-7)
    assert res["verdict"] == "SURVIVES"
    assert res["lambda_over_rc2"] == pytest.approx(1e-3)


def test_check_csl_lpf_only_at_reference_rc():
    # At r_C = 1e-6 m the LISA Pathfinder bound is not applied.
    res = constraints.check_csl(1e-16, 1e-6)
    by_id = {p["constraint_id"]: p["verdict"]
             for p in res["per_constraint"]}
    assert by_id["csl_lisa_pathfinder_carlesso2016"] == "UNCONSTRAINED"
    assert res["verdict"] == "SURVIVES"


def test_check_csl_adler_values_excluded():
    # Adler's lambda ~ 1e-8 /s at r_C = 1e-7 m is excluded by both
    # the X-ray bounds and LISA Pathfinder.
    res = constraints.check_csl(1e-8, 1e-7)
    assert res["verdict"] == "EXCLUDED"


def test_check_dp_cutoff():
    # Parameter-free DP (R_0 = 0) is excluded; R_0 above the Majorana
    # lower limit survives.
    assert constraints.check_dp_cutoff(0.0)["verdict"] == "EXCLUDED"
    assert constraints.check_dp_cutoff(1e-11)["verdict"] == "EXCLUDED"
    assert constraints.check_dp_cutoff(2.54e-10)["verdict"] == "SURVIVES"
    assert constraints.check_dp_cutoff(1e-9)["verdict"] == "SURVIVES"


def test_oracle_still_intact():
    """The CSL phenomenology does not touch the oracle boundary."""
    with pytest.raises(collapse.CollapseNotSimulable):
        collapse.collapse_oracle()
