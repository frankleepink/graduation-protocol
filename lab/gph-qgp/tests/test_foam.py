"""Tests for gph_qgp.foam."""

import numpy as np
import pytest

from gph_qgp import foam


def test_fluctuation_monotonic_in_length():
    lengths = np.logspace(-3, 4, 20)
    for model in foam.MODELS:
        d = foam.fluctuation(lengths, model=model)
        assert np.all(np.diff(d) > 0), f"{model} not monotonic"


def test_fluctuation_scaling_laws():
    # holographic: delta_l ~ L^(1/3); random_walk: delta_l ~ L^(1/2)
    l1, l2 = 1.0, 1000.0
    holo = foam.fluctuation(np.array([l1, l2]), model="holographic")
    assert holo[1] / holo[0] == pytest.approx((l2 / l1) ** (1 / 3), rel=1e-9)
    rw = foam.fluctuation(np.array([l1, l2]), model="random_walk")
    assert rw[1] / rw[0] == pytest.approx(np.sqrt(l2 / l1), rel=1e-9)


def test_random_walk_larger_than_holographic_at_lab_scales():
    # At metre-to-km scales the random-walk scaling is far larger.
    assert foam.fluctuation(4000.0, "random_walk") > \
        foam.fluctuation(4000.0, "holographic")


def test_fluctuation_zero_length():
    assert foam.fluctuation(0.0, "holographic") == 0.0
    assert foam.fluctuation(0.0, "random_walk") == 0.0


def test_fluctuation_negative_length_raises():
    with pytest.raises(ValueError):
        foam.fluctuation(-1.0, "holographic")


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        foam.fluctuation(1.0, model="nope")


def test_phase_noise_scales_with_inverse_wavelength():
    p1 = foam.phase_noise(100.0, 1064e-9, "holographic")
    p2 = foam.phase_noise(100.0, 532e-9, "holographic")
    assert p2 == pytest.approx(2 * p1, rel=1e-9)


def test_simulate_interferometer_matches_theory():
    sim = foam.simulate_interferometer(4000.0, 1064e-9, model="holographic",
                                       n_samples=20000, seed=0)
    assert sim["sigma_phi_theory_rad"] > 0
    # Sample std should be within 5% of theory for 20000 samples.
    assert abs(sim["sigma_phi_sample_rad"] / sim["sigma_phi_theory_rad"] - 1) < 0.05
    assert sim["phase_rad"].shape == (20000,)
