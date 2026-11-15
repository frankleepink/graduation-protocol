"""Tests for gph_qgp.collapse."""

import numpy as np
import pytest

from gph_qgp import collapse as c

X = np.linspace(-5e-7, 5e-7, 512)
MASS = 1e-25  # kg, toy particle


def _packet():
    return c.gaussian_wavepacket(X, x0=0.0, sigma=1e-7)


def test_oracle_raises_and_labels():
    with pytest.raises(c.CollapseNotSimulable) as exc:
        c.collapse_oracle()
    msg = str(exc.value)
    assert "NOT simulated" in msg
    assert "Faizal" in msg
    # It is a NotImplementedError subclass.
    assert issubclass(c.CollapseNotSimulable, NotImplementedError)


def test_unitary_evolution_conserves_norm():
    psi = _packet()
    assert c.norm(psi, X) == pytest.approx(1.0, rel=1e-9)
    psi2 = c.evolve_unitary(psi, X, dt=1e-12, mass_kg=MASS, n_steps=50)
    assert c.norm(psi2, X) == pytest.approx(1.0, rel=1e-6)


def test_unitary_evolution_spreads_wavepacket():
    psi = _packet()
    psi2 = c.evolve_unitary(psi, X, dt=1e-9, mass_kg=MASS, n_steps=200)
    width0 = np.sqrt(np.trapz(X**2 * np.abs(psi) ** 2, X))
    width1 = np.sqrt(np.trapz(X**2 * np.abs(psi2) ** 2, X))
    assert width1 > width0


def test_grw_hit_localises_and_renormalises():
    psi = _packet()
    psi2, centre = c.grw_hit(psi, X, r_c_m=1e-7, rng=0)
    assert c.norm(psi2, X) == pytest.approx(1.0, rel=1e-9)
    width0 = np.sqrt(np.trapz(X**2 * np.abs(psi) ** 2, X))
    width1 = np.sqrt(np.trapz((X - centre) ** 2 * np.abs(psi2) ** 2, X))
    assert width1 < width0
    assert X.min() <= centre <= X.max()


def test_grw_trajectory_norm_conserved():
    traj = c.grw_trajectory(_packet(), X, dt=1e-4, mass_kg=MASS,
                            lam_per_s=50.0, n_steps=200, seed=0)
    assert np.all(np.abs(traj["norm_history"] - 1.0) < 1e-6)
    assert traj["n_hits"] >= 0
    assert len(traj["hit_times_s"]) == traj["n_hits"]


def test_grw_no_hits_when_lambda_zero():
    traj = c.grw_trajectory(_packet(), X, dt=1e-4, mass_kg=MASS,
                            lam_per_s=0.0, n_steps=50, seed=0)
    assert traj["n_hits"] == 0


def test_dp_collapse_time_decreases_with_mass():
    taus = [c.dp_collapse_time(m, 1e-6) for m in (1e-15, 1e-12, 1e-9)]
    assert taus[0] > taus[1] > taus[2]
    assert all(t > 0 for t in taus)


def test_dp_collapse_time_order_of_magnitude():
    # 1e-12 kg, 1 um: dE = G m^2 / R ~ 6.7e-29 J, tau = hbar/dE ~ 1.6e-6 s
    tau = c.dp_collapse_time(1e-12, 1e-6)
    assert 1e-7 < tau < 1e-5


def test_dp_displacement_reduces_delta_e():
    tau_no_d = c.dp_collapse_time(1e-12, 1e-6)
    tau_small_d = c.dp_collapse_time(1e-12, 1e-6, displacement_m=1e-9)
    assert tau_small_d > tau_no_d  # smaller Delta_E -> longer tau
    assert c.dp_collapse_time(1e-12, 1e-6, displacement_m=0.0) == float("inf")


def test_dp_invalid_inputs_raise():
    with pytest.raises(ValueError):
        c.dp_collapse_time(-1.0, 1e-6)
    with pytest.raises(ValueError):
        c.dp_collapse_time(1e-12, 0.0)
