"""Trans-Planckian study: sweep effective quantum-gravity scales across channels.

Sweeps sub-Planckian to trans-Planckian effective scales (0.1x to 1e4x
Planck) and classifies each against the published observational bounds
in this package: linear dispersion, vacuum birefringence, and
superluminal photon decay. Also includes a redshift-integrated
dispersion delay, a dispersion simulator + log-scale fitter, and
JSON/CSV export of the sweep. The sweep uses a 45-point geometric grid
(0.1x to 1e4x Planck), matching the QGP Study Workbench built from this
lab: each row also carries obs_at_emax, the channel's live observable
evaluated at the max photon energy (dispersion delay in s, birefringence
rotation in deg, decay threshold in GeV). lever_arm_analysis() adds the
identifiability analysis: how the fitted log10 E_QG uncertainty shrinks
as Emax grows, for n=1 vs n=2.

Honesty note: toy models on simulated data only, like the rest of this
package. The effective scale Lambda_eff = E_QG / |eta|^(1/n) is the
standard EFT suppression-scale bookkeeping, not a new physical claim.
Collapse mechanics are NOT in this sweep yet -- see collapse.py for the
GRW / CSL / Diosi-Penrose toy models and the non-computable oracle
boundary (collapse itself is never simulated).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares

# Physical & cosmological constants (CODATA / Planck 2018 baseline)
KM_S_MPC_TO_S_INV: float = 1_000.0 / 3.0856775814913673e22
HBAR_GEV_S: float = 6.582119569e-25
HC_GEV_M: float = 1.973269804e-16
L_PLANCK_M: float = 1.616255e-35
E_PLANCK_GEV: float = 1.2208908e19
M_ELECTRON_GEV: float = 0.00051099895
DEFAULT_SENSITIVITY_FLOOR_RATIO: float = 1e-3


@dataclass(frozen=True)
class Cosmology:
    h0_km_s_mpc: float = 67.4
    omega_m: float = 0.315

    @property
    def omega_lambda(self) -> float:
        return 1.0 - self.omega_m

    @property
    def h0_si(self) -> float:
        return self.h0_km_s_mpc * KM_S_MPC_TO_S_INV

    @property
    def hubble_time_s(self) -> float:
        return 1.0 / self.h0_si

    @property
    def h0_gev(self) -> float:
        return self.h0_si * HBAR_GEV_S

    def comoving_distance_m(self, z: float, c_m_s: float = 299_792_458.0) -> float:
        def inv_e(zp: float) -> float:
            ez = np.sqrt(self.omega_m * (1.0 + zp) ** 3 + self.omega_lambda)
            return 1.0 / ez

        dc_over_dh, _ = quad(inv_e, 0.0, float(z), epsabs=1e-13, epsrel=1e-11)
        return float((c_m_s / self.h0_si) * dc_over_dh)


@dataclass(frozen=True)
class ConstraintBound:
    name: str
    channel: Literal["dispersion", "birefringence", "superluminal_decay", "foam"]
    order: int
    min_scale_gev: float
    sensitivity_floor_ratio: float = DEFAULT_SENSITIVITY_FLOOR_RATIO


def effective_scale_gev(scale_gev: float, eta: float = 1.0, order: int = 1) -> float:
    """Compute EFT effective suppression scale Lambda_eff = E_QG / |eta|^(1/n)."""
    if scale_gev <= 0.0 or not np.isfinite(scale_gev):
        raise ValueError("scale_gev must be positive and finite.")
    if eta == 0.0 or not np.isfinite(eta):
        raise ValueError("eta must be non-zero and finite.")
    return float(scale_gev / (abs(eta) ** (1.0 / float(order))))


def classify_scale(
    min_scale_gev: float,
    eff_scale_gev: float,
    sensitivity_floor_ratio: float = DEFAULT_SENSITIVITY_FLOOR_RATIO,
) -> dict[str, Any]:
    """Classify an effective scale into EXCLUDED, SURVIVES, or UNCONSTRAINED."""
    if eff_scale_gev <= 0.0 or not np.isfinite(eff_scale_gev):
        raise ValueError("eff_scale_gev must be positive and finite.")
    if not (0.0 < sensitivity_floor_ratio < 1.0):
        raise ValueError("sensitivity_floor_ratio must lie strictly between 0 and 1.")

    ratio = float(min_scale_gev / eff_scale_gev)
    if ratio > 1.0:
        verdict = "EXCLUDED"
    elif ratio < sensitivity_floor_ratio:
        verdict = "UNCONSTRAINED"
    else:
        verdict = "SURVIVES"

    return {
        "verdict": verdict,
        "ratio": ratio,
        "eff_scale_gev": float(eff_scale_gev),
        "eff_scale_over_epl": float(eff_scale_gev / E_PLANCK_GEV),
        "min_scale_over_epl": float(min_scale_gev / E_PLANCK_GEV),
    }


def dispersion_redshift_integral(
    z: float, order: int, cosmo: Cosmology = Cosmology()
) -> float:
    if order not in (1, 2):
        raise ValueError("Supported dispersion orders are 1 and 2.")

    def integrand(zp: float) -> float:
        ez = np.sqrt(cosmo.omega_m * (1.0 + zp) ** 3 + cosmo.omega_lambda)
        return float(((1.0 + zp) ** order) / ez)

    val, _ = quad(integrand, 0.0, float(z), epsabs=1e-13, epsrel=1e-11)
    return float(val)


def dispersion_delay_s(
    energies_gev: np.ndarray | float,
    z: float,
    e_qg_gev: float,
    order: int = 1,
    eta: float = 1.0,
    cosmo: Cosmology = Cosmology(),
) -> np.ndarray | float:
    """Redshift-integrated dispersion arrival-time lag in seconds."""
    scalar_input = np.ndim(energies_gev) == 0
    e_arr = np.asarray(energies_gev, dtype=float)
    iz = dispersion_redshift_integral(z, order=order, cosmo=cosmo)
    prefactor = ((order + 1.0) / 2.0) * cosmo.hubble_time_s
    out = eta * prefactor * ((e_arr / float(e_qg_gev)) ** order) * iz
    return float(out) if scalar_input else out


def birefringence_rotation_deg(
    energies_gev: np.ndarray | float,
    z: float,
    scale_gev: float,
    eta_pol: float = 1.0,
    cosmo: Cosmology = Cosmology(),
) -> np.ndarray | float:
    """Linear-order vacuum birefringence polarization rotation angle in degrees."""
    scalar_input = np.ndim(energies_gev) == 0
    e_arr = np.asarray(energies_gev, dtype=float)
    iz = dispersion_redshift_integral(z, order=1, cosmo=cosmo)
    rad = eta_pol * ((e_arr ** 2) / (2.0 * cosmo.h0_gev * float(scale_gev))) * iz
    deg = np.degrees(rad)
    return float(deg) if scalar_input else deg


def photon_decay_threshold_gev(
    order: int,
    scale_gev: float,
    eta_super: float = 1.0,
    high_bound_gev: float = 1e26,
) -> float:
    """Superluminal photon decay (gamma -> e+ e-) threshold energy in GeV."""
    if eta_super <= 0.0:
        return float("inf")
    rhs = 4.0 * (M_ELECTRON_GEV ** 2)

    def f(e_val: float) -> float:
        return float(eta_super * ((e_val / float(scale_gev)) ** order) * (e_val ** 2) - rhs)

    low = 1e-3
    if f(low) >= 0.0:
        return float(low)
    if f(high_bound_gev) < 0.0:
        return float("inf")
    return float(brentq(f, low, high_bound_gev, xtol=1e-12, rtol=1e-12, maxiter=200))


def foam_path_fluctuation_m(
    energies_gev: np.ndarray | float,
    distance_m: float,
    l_qg_m: float = L_PLANCK_M,
    alpha_foam: float = 0.5,
) -> np.ndarray | float:
    """Stochastic spacetime-foam path fluctuation sigma_x in meters."""
    scalar_input = np.ndim(energies_gev) == 0
    e_arr = np.maximum(np.asarray(energies_gev, dtype=float), 1e-15)
    wavelength_m = HC_GEV_M / e_arr
    n_steps = np.maximum(float(distance_m) / np.maximum(wavelength_m, 1e-35), 1.0)
    sigma_x = (float(l_qg_m) ** (1.0 - alpha_foam)) * (wavelength_m ** alpha_foam) * np.sqrt(n_steps)
    return float(sigma_x) if scalar_input else sigma_x


def simulate_dispersion_arrival_times(
    energies_gev: np.ndarray,
    z: float = 0.9,
    order: int = 1,
    eta_true: float = 1.0,
    e_qg_true_gev: float = 2e17,
    t0_s: float = 0.25,
    noise_std_s: float = 0.05,
    seed: int = 42,
    cosmo: Cosmology = Cosmology(),
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    e_arr = np.asarray(energies_gev, dtype=float)
    signal_s = np.asarray(
        dispersion_delay_s(e_arr, z=z, e_qg_gev=e_qg_true_gev, order=order, eta=eta_true, cosmo=cosmo),
        dtype=float,
    )
    sigma_arr = np.full_like(e_arr, float(noise_std_s), dtype=float)
    t_obs_s = float(t0_s) + signal_s + rng.normal(0.0, sigma_arr)
    return {
        "energies_gev": e_arr,
        "signal_s": signal_s,
        "t_obs_s": t_obs_s,
        "sigma_s": sigma_arr,
    }


def fit_dispersion_log_scale(
    energies_gev: np.ndarray,
    t_obs_s: np.ndarray,
    sigma_s: np.ndarray,
    z: float = 0.9,
    order: int = 1,
    eta_sign: int = 1,
    init_log10_eqg_gev: float = 18.0,
    init_t0_s: float = 0.0,
    bounds_log10_eqg: tuple[float, float] = (5.0, 26.0),
    cosmo: Cosmology = Cosmology(),
) -> dict[str, float]:
    e_arr = np.asarray(energies_gev, dtype=float)
    t_arr = np.asarray(t_obs_s, dtype=float)
    s_arr = np.asarray(sigma_s, dtype=float)
    weights = 1.0 / np.maximum(s_arr, 1e-15)
    eta_val = 1.0 if eta_sign >= 0 else -1.0

    def residuals(params: np.ndarray) -> np.ndarray:
        t0_hat, log10_eqg_hat = params
        pred = t0_hat + np.asarray(
            dispersion_delay_s(
                e_arr, z=z, e_qg_gev=10.0 ** float(log10_eqg_hat), order=order, eta=eta_val, cosmo=cosmo
            ),
            dtype=float,
        )
        return weights * (pred - t_arr)

    lo_log10, hi_log10 = bounds_log10_eqg
    x0 = np.array(
        [float(init_t0_s), float(np.clip(init_log10_eqg_gev, lo_log10 + 1e-3, hi_log10 - 1e-3))]
    )
    sol = least_squares(residuals, x0=x0, bounds=([-np.inf, lo_log10], [np.inf, hi_log10]))
    t0_fit, log10_fit = sol.x
    dof = max(len(e_arr) - 2, 1)
    chi2 = float(np.sum(sol.fun ** 2))
    chi2_red = chi2 / dof

    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * max(chi2_red, 1.0)
        log10_err = float(np.sqrt(cov[1, 1]))
        t0_err = float(np.sqrt(cov[0, 0]))
    except np.linalg.LinAlgError:
        log10_err = float("nan")
        t0_err = float("nan")

    return {
        "log10_e_qg_fit_gev": float(log10_fit),
        "e_qg_fit_gev": float(10.0 ** float(log10_fit)),
        "log10_e_qg_err": log10_err,
        "t0_fit_s": float(t0_fit),
        "t0_err_s": t0_err,
        "chi2": chi2,
        "reduced_chi2": chi2_red,
    }


def evaluate_fitted_eqg_verdict(
    log10_eqg_fit_gev: float,
    log10_eqg_err: float,
    min_scale_gev: float = 12.3 * E_PLANCK_GEV,
    n_sigma: float = 1.0,
    sensitivity_floor_ratio: float = DEFAULT_SENSITIVITY_FLOOR_RATIO,
    max_log10_err: float = 50.0,
) -> dict[str, Any]:
    err = abs(float(log10_eqg_err)) if np.isfinite(log10_eqg_err) else 0.0
    err = min(err, float(max_log10_err))

    center_log10 = float(log10_eqg_fit_gev)
    low_log10 = center_log10 - n_sigma * err
    high_log10 = center_log10 + n_sigma * err

    center_c = classify_scale(min_scale_gev, float(10.0 ** center_log10), sensitivity_floor_ratio)
    low_c = classify_scale(min_scale_gev, float(10.0 ** low_log10), sensitivity_floor_ratio)
    high_c = classify_scale(min_scale_gev, float(10.0 ** high_log10), sensitivity_floor_ratio)

    ci_verdicts: list[str] = []
    for v in (low_c["verdict"], center_c["verdict"], high_c["verdict"]):
        if v not in ci_verdicts:
            ci_verdicts.append(v)

    return {
        "center_log10_eqg_gev": center_log10,
        "ci_low_log10_eqg_gev": low_log10,
        "ci_high_log10_eqg_gev": high_log10,
        "center_over_epl": center_c["eff_scale_over_epl"],
        "ci_low_over_epl": low_c["eff_scale_over_epl"],
        "ci_high_over_epl": high_c["eff_scale_over_epl"],
        "center_verdict": center_c["verdict"],
        "center_ratio": center_c["ratio"],
        "ci_low_verdict": low_c["verdict"],
        "ci_high_verdict": high_c["verdict"],
        "ci_verdicts": ci_verdicts,
        "spans_boundary": len(ci_verdicts) > 1,
    }


def lever_arm_analysis(
    emax_values_gev: np.ndarray | None = None,
    orders: tuple[int, ...] = (1, 2),
    n_photons: int = 100,
    z: float = 0.9,
    e_qg_true_gev: float = 1e21,
    noise_std_s: float = 0.05,
    seed: int = 42,
    cosmo: Cosmology = Cosmology(),
) -> list[dict[str, Any]]:
    """Identifiability lever arm: fitted log10 E_QG uncertainty vs Emax.

    For each max photon energy and dispersion order, simulate N photons
    log-spaced up to Emax, fit log10 E_QG, and record the fit uncertainty.
    Higher Emax is a longer lever arm: the delay signal grows with energy,
    so the uncertainty should shrink.

    Default true scale is 1e21 GeV (trans-Planckian) on purpose: with the
    package's 0.05 s timing noise, a 2e17 GeV scale puts the delay signal
    thousands of times above the noise at every Emax, so the curve is flat
    at ~1e-6 dex and shows nothing. At 1e21 GeV the GeV-photon delays sit
    near the noise floor at low Emax and rise clear of it at high Emax,
    so the lever arm actually shows: n=1 uncertainty falls from ~0.8 dex
    at 100 GeV to ~0.002 dex at 10 TeV.

    Honest caveat: the n=2 (quadratic) signal stays far below the timing
    noise at every Emax here, so its uncertainty is inf and the fit wanders
    to the bounds. That is itself the finding -- quadratic dispersion is
    unidentifiable from GeV photons alone at these scales. A non-finite
    uncertainty is recorded as inf, never hidden.
    """
    if emax_values_gev is None:
        emax_values_gev = np.geomspace(100.0, 1e4, num=9)

    rows: list[dict[str, Any]] = []
    for order in orders:
        for emax in emax_values_gev:
            emax_f = float(emax)
            energies = np.geomspace(emax_f / 100.0, emax_f, num=n_photons)
            sim = simulate_dispersion_arrival_times(
                energies,
                z=z,
                order=order,
                e_qg_true_gev=e_qg_true_gev,
                noise_std_s=noise_std_s,
                seed=seed,
                cosmo=cosmo,
            )
            fit = fit_dispersion_log_scale(
                sim["energies_gev"], sim["t_obs_s"], sim["sigma_s"],
                z=z, order=order, cosmo=cosmo,
            )
            err = fit["log10_e_qg_err"]
            rows.append({
                "emax_gev": emax_f,
                "order": int(order),
                "log10_e_qg_true_gev": float(np.log10(e_qg_true_gev)),
                "log10_e_qg_fit_gev": fit["log10_e_qg_fit_gev"],
                "log10_e_qg_err": float(err) if np.isfinite(err) else float("inf"),
                "reduced_chi2": fit["reduced_chi2"],
            })
    return rows


def run_trans_planckian_study(
    scale_multipliers_epl: np.ndarray | None = None,
    z: float = 0.9,
    ref_energy_gev: float = 100.0,
    emax_gev: float = 5000.0,
    foam_alpha: float = 0.5,
    cosmo: Cosmology = Cosmology(),
) -> list[dict[str, Any]]:
    """Sweep sub-Planckian to trans-Planckian effective scales across all channels.

    Default grid is 45 geometric points from 0.1x to 1e4x Planck, matching
    the QGP Study Workbench. Each row carries obs_at_emax: the channel's
    live observable at the max photon energy -- dispersion delay (s),
    birefringence rotation (deg), or decay threshold (GeV).
    """
    if scale_multipliers_epl is None:
        scale_multipliers_epl = np.geomspace(0.1, 1e4, num=45)

    bounds = [
        ConstraintBound("Linear dispersion (n=1)", "dispersion", 1, 12.3 * E_PLANCK_GEV),
        ConstraintBound("Birefringence (n=1)", "birefringence", 1, 10.0 * E_PLANCK_GEV),
        ConstraintBound("Superluminal decay (n=1)", "superluminal_decay", 1, 1.0 * E_PLANCK_GEV),
    ]
    dist_m = cosmo.comoving_distance_m(z)
    rows: list[dict[str, Any]] = []

    for idx, mult in enumerate(scale_multipliers_epl):
        eff_scale = float(mult) * E_PLANCK_GEV
        dt_s = float(dispersion_delay_s(ref_energy_gev, z=z, e_qg_gev=eff_scale, order=1, eta=1.0, cosmo=cosmo))
        rot_deg = float(birefringence_rotation_deg(ref_energy_gev, z=z, scale_gev=eff_scale, eta_pol=1.0, cosmo=cosmo))
        e_th_gev = photon_decay_threshold_gev(order=1, scale_gev=eff_scale, eta_super=1.0)
        l_foam_m = L_PLANCK_M / max(float(mult), 1e-12)
        sigma_foam_m = float(foam_path_fluctuation_m(ref_energy_gev, distance_m=dist_m, l_qg_m=l_foam_m, alpha_foam=foam_alpha))

        obs_at_emax = {
            "dispersion": float(dispersion_delay_s(emax_gev, z=z, e_qg_gev=eff_scale, order=1, eta=1.0, cosmo=cosmo)),
            "birefringence": float(birefringence_rotation_deg(emax_gev, z=z, scale_gev=eff_scale, eta_pol=1.0, cosmo=cosmo)),
            "superluminal_decay": e_th_gev,
        }
        for b in bounds:
            cls = classify_scale(b.min_scale_gev, eff_scale, b.sensitivity_floor_ratio)
            rows.append({
                "index": idx,
                "scale_over_epl": float(mult),
                "eff_scale_gev": eff_scale,
                "constraint": b.name,
                "channel": b.channel,
                "verdict": cls["verdict"],
                "ratio": cls["ratio"],
                "dispersion_delay_s": dt_s,
                "birefringence_deg": rot_deg,
                "photon_decay_threshold_gev": e_th_gev,
                "foam_sigma_m": sigma_foam_m,
                "obs_at_emax": obs_at_emax[b.channel],
            })
    return rows


def export_study_results(rows: list[dict[str, Any]], out_prefix: str = "qgp_study_output") -> dict[str, str]:
    json_path = Path(f"{out_prefix}.json")
    csv_path = Path(f"{out_prefix}.csv")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, allow_nan=True)

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return {"json": str(json_path), "csv": str(csv_path)}
