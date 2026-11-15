"""Tests for gph_qgp.pilot."""

import numpy as np
import pytest

from gph_qgp import pilot as p


def test_synth_speckle_shape_and_mean():
    rng = np.random.default_rng(0)
    img = p.synth_speckle(128, 4.0, rng)
    assert img.shape == (128, 128)
    assert img.mean() == pytest.approx(1.0, rel=1e-9)
    assert np.all(img >= 0)


def test_synth_speckle_contrast_near_one():
    rng = np.random.default_rng(0)
    img = p.synth_speckle(256, 4.0, rng)
    assert 0.8 < p.contrast(img) < 1.2  # fully developed speckle


def test_synth_speckle_fwhm_matches_target():
    rng = np.random.default_rng(1)
    img = p.synth_speckle(256, 4.0, rng)
    assert abs(p.estimate_fwhm(img) - 4.0) < 1.0


def test_synth_speckle_bad_fwhm_raises():
    with pytest.raises(ValueError):
        p.synth_speckle(64, 1000.0, np.random.default_rng(0))


def test_structure_strength_needs_structure():
    with pytest.raises(ValueError):
        p.synth_speckle(64, 4.0, np.random.default_rng(0),
                         structure_strength=0.5)


def test_structure_blend_raises_correlation():
    rng = np.random.default_rng(2)
    fixed = p.fixed_speckle(128, 4.0, rng)
    a = p.synth_speckle(128, 4.0, np.random.default_rng(10),
                        structure_strength=0.8, structure=fixed)
    b = p.synth_speckle(128, 4.0, np.random.default_rng(11),
                        structure_strength=0.8, structure=fixed)
    c = p.synth_speckle(128, 4.0, np.random.default_rng(12))
    d = p.synth_speckle(128, 4.0, np.random.default_rng(13))
    assert p.pearson(a, b) > 0.5
    assert abs(p.pearson(c, d)) < 0.2


def test_cohen_kappa_perfect_and_chance():
    assert p.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    # Independent random ratings: kappa near 0 on average; exact check
    # with a balanced disagreement case.
    assert p.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_cohen_kappa_in_range():
    rng = np.random.default_rng(0)
    cats = ["none", "texture only", "geometric structure"]
    a = [cats[i] for i in rng.integers(0, 3, 50)]
    b = [cats[i] for i in rng.integers(0, 3, 50)]
    assert -1.0 <= p.cohen_kappa(a, b) <= 1.0


def test_fleiss_kappa_perfect_agreement():
    ratings = np.array([["a", "a", "a"], ["b", "b", "b"], ["a", "a", "a"]])
    assert p.fleiss_kappa(ratings) == pytest.approx(1.0)


def test_fleiss_kappa_in_range():
    rng = np.random.default_rng(0)
    cats = np.array(["none", "texture only", "geometric structure"])
    ratings = cats[rng.integers(0, 3, size=(20, 4))]
    assert -1.0 <= p.fleiss_kappa(ratings) <= 1.0


def test_krippendorff_alpha_perfect_agreement():
    units = {("W", "L00"): ["grid", "grid", "grid"],
             ("W", "L01"): ["dots", "dots", "dots"]}
    assert p.krippendorff_alpha_nominal(units) == pytest.approx(1.0)


def test_krippendorff_alpha_in_range():
    rows = p.simulate_observers(n_observers=6, n_locations=10, mode="null",
                                seed=0)
    from collections import defaultdict
    units = defaultdict(list)
    for r in rows:
        units[(r["condition"], r["location_id"])].append(r["response"])
    alpha = p.krippendorff_alpha_nominal(dict(units))
    assert -1.0 <= alpha <= 1.0


def test_pairwise_agreement_bounds():
    assert p.pairwise_agreement([["a", "a"], ["a", "b"]]) == pytest.approx(0.5)
    assert np.isnan(p.pairwise_agreement([["a"]]))


def test_simulate_observers_null_has_no_shared_signal():
    rows = p.simulate_observers(n_observers=12, n_locations=30, mode="null",
                                seed=7)
    res = p.analyse_observers(rows, n_perm=200, seed=1)
    # Under the null, W and S alphas should both be near zero.
    assert abs(res["per_condition"]["W"]["alpha_nominal"]) < 0.15
    assert abs(res["per_condition"]["S"]["alpha_nominal"]) < 0.15


def test_simulate_observers_stored_shows_w_signal():
    rows = p.simulate_observers(n_observers=12, n_locations=30, mode="stored",
                                q=0.8, seed=7)
    res = p.analyse_observers(rows, n_perm=200, seed=1)
    assert res["per_condition"]["W"]["alpha_nominal"] > \
        res["per_condition"]["S"]["alpha_nominal"] + 0.1


def test_invariance_check_stored_vs_conventional(tmp_path):
    rng = np.random.default_rng(0)
    fixed = p.fixed_speckle(128, 4.0, rng)
    stored = [p.synth_speckle(128, 4.0, np.random.default_rng([0, i]),
                              structure_strength=0.7, structure=fixed)
              for i in range(3)]
    conv = [p.synth_speckle(128, 4.0, np.random.default_rng([1, i]))
            for i in range(3)]
    rs = p.invariance_check(stored, ["a", "b", "c"], null_draws=10, seed=0)
    rc = p.invariance_check(conv, ["a", "b", "c"], null_draws=10, seed=0)
    mean_stored = float(np.mean([r["best_r"] for r in rs]))
    mean_conv = float(np.mean([r["best_r"] for r in rc]))
    assert mean_stored > mean_conv + 0.3


def test_run_demo_end_to_end(tmp_path):
    summary = p.run_demo(str(tmp_path), seed=7)
    assert (tmp_path / "observers.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert set(summary["per_condition"]) == {"W", "R", "S"}
    for cond, d in summary["per_condition"].items():
        assert -1.0 <= d["alpha_nominal"] <= 1.0
        assert d["units"] == 30
    assert "comparison" in summary
    assert "invariance_demo" in summary
