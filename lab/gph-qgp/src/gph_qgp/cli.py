"""Command-line runner for the GPH Quantum Gravity Phenomenology lab.

Usage:
    python -m gph_qgp run --experiment {dispersion,foam,collapse,pilot,superluminal,birefringence,gup,csl,neutrinos} --out <dir>
    python -m gph_qgp compare --out <dir>

Each experiment writes CSV results and a PNG plot into ``--out`` and
prints a one-paragraph summary. All data is simulated; see the module
docstrings and the README honesty section.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import (birefringence, collapse, constraints, dispersion, foam,
               gup, neutrinos, pilot, superluminal)


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_dispersion(out_dir, seed=0):
    """GRB photon time-of-flight: simulate arrivals, fit eta."""
    energies = np.logspace(-1, 2, 40)  # 0.1 .. 100 GeV
    sim = dispersion.simulate_grb(energies, distance_mpc=1000.0, eta=1.0,
                                  emission_jitter_s=0.02, seed=seed)
    fit = dispersion.fit_eta(sim["energies_gev"], sim["arrival_s"],
                             distance_mpc=1000.0)
    _write_csv(os.path.join(out_dir, "dispersion.csv"),
               ["energy_gev", "true_delay_s", "arrival_s"],
               [{"energy_gev": f"{e:.6g}", "true_delay_s": f"{d:.6g}",
                 "arrival_s": f"{a:.6g}"}
                for e, d, a in zip(sim["energies_gev"], sim["true_delay_s"],
                                   sim["arrival_s"])])
    fig, ax = plt.subplots()
    ax.scatter(sim["energies_gev"], sim["arrival_s"], s=18, label="simulated arrivals")
    xs = np.array([energies.min(), energies.max()])
    slope = fit["eta"] * (1000.0 * dispersion.MPC_IN_M) / (
        dispersion.C_M_PER_S * dispersion.E_PLANCK_GEV)
    ax.plot(xs, fit["intercept_s"] + slope * xs, "r-", label="linear fit")
    ax.set_xlabel("photon energy (GeV)")
    ax.set_ylabel("arrival time (s)")
    ax.set_title("GRB photon time-of-flight (simulated, eta_true = 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dispersion.png"), dpi=120)
    plt.close(fig)
    print(
        f"Dispersion experiment (simulated GRB at 1000 Mpc, 40 photons "
        f"from 0.1 to 100 GeV, true eta = 1): the linear fit of arrival "
        f"time against energy recovers eta = {fit['eta']:.3f} +/- "
        f"{fit['eta_err']:.3f} (r = {fit['r_value']:.3f}). A positive eta "
        f"means high-energy photons arrive later; this is a pipeline "
        f"demonstration on synthetic data, not a measurement."
    )
    return fit


def run_foam(out_dir, seed=0):
    """Spacetime-foam phase noise vs arm length, both scalings."""
    lengths = np.logspace(-3, 4, 40)  # 1 mm .. 10 km
    wavelength_m = 1064e-9
    rows = []
    for model in foam.MODELS:
        d = foam.fluctuation(lengths, model=model)
        ph = foam.phase_noise(lengths, wavelength_m, model=model)
        for L, dl, p in zip(lengths, d, ph):
            rows.append({"model": model, "arm_length_m": f"{L:.6g}",
                         "delta_l_m": f"{dl:.6g}", "sigma_phi_rad": f"{p:.6g}"})
    _write_csv(os.path.join(out_dir, "foam.csv"),
               ["model", "arm_length_m", "delta_l_m", "sigma_phi_rad"], rows)
    fig, ax = plt.subplots()
    for model in foam.MODELS:
        d = foam.fluctuation(lengths, model=model)
        ax.loglog(lengths, d, label=model)
    ax.set_xlabel("arm length (m)")
    ax.set_ylabel("distance fluctuation delta_l (m)")
    ax.set_title("Spacetime-foam fluctuation vs length (toy scalings)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "foam.png"), dpi=120)
    plt.close(fig)
    sim = foam.simulate_interferometer(4000.0, wavelength_m,
                                       model="holographic", seed=seed)
    print(
        f"Foam experiment (laser wavelength 1064 nm): at a 4 km arm the "
        f"holographic scaling gives delta_l = "
        f"{foam.fluctuation(4000.0, 'holographic'):.3g} m "
        f"(phase noise {foam.phase_noise(4000.0, wavelength_m, 'holographic'):.3g} rad) "
        f"and the random-walk scaling gives delta_l = "
        f"{foam.fluctuation(4000.0, 'random_walk'):.3g} m "
        f"(phase noise {foam.phase_noise(4000.0, wavelength_m, 'random_walk'):.3g} rad). "
        f"A simulated 10 s phase series at 4 km has sample sigma "
        f"{sim['sigma_phi_sample_rad']:.3g} rad vs theory "
        f"{sim['sigma_phi_theory_rad']:.3g} rad. These are toy scalings "
        f"for an effect never observed; the two models disagree by "
        f"orders of magnitude, which is itself the point."
    )
    return {"holographic_4km_m": float(foam.fluctuation(4000.0, "holographic")),
            "random_walk_4km_m": float(foam.fluctuation(4000.0, "random_walk"))}


def run_collapse(out_dir, seed=0):
    """GRW toy trajectory plus Diosi-Penrose collapse times."""
    x = np.linspace(-5e-7, 5e-7, 512)
    psi0 = collapse.gaussian_wavepacket(x, x0=0.0, sigma=1e-7)
    mass_kg = 1e-25
    dt = 1e-4
    traj = collapse.grw_trajectory(psi0, x, dt, mass_kg, lam_per_s=50.0,
                                   n_steps=200, seed=seed)
    masses = np.logspace(-18, -9, 25)
    taus = [collapse.dp_collapse_time(m, 1e-6) for m in masses]
    _write_csv(os.path.join(out_dir, "collapse_dp.csv"),
               ["mass_kg", "radius_m", "tau_s"],
               [{"mass_kg": f"{m:.6g}", "radius_m": "1e-06",
                 "tau_s": f"{t:.6g}"} for m, t in zip(masses, taus)])
    with open(os.path.join(out_dir, "collapse_grw.json"), "w") as fh:
        json.dump({"n_hits": traj["n_hits"],
                   "hit_times_s": traj["hit_times_s"].tolist(),
                   "hit_centres_m": traj["hit_centres_m"].tolist(),
                   "norm_min": float(traj["norm_history"].min()),
                   "norm_max": float(traj["norm_history"].max())}, fh, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(x * 1e9, np.abs(psi0) ** 2, label="initial")
    axes[0].plot(x * 1e9, np.abs(traj["psi"]) ** 2, label="after trajectory")
    axes[0].set_xlabel("x (nm)")
    axes[0].set_ylabel("|psi|^2 (1/m)")
    axes[0].set_title("GRW toy trajectory (phenomenological hits)")
    axes[0].legend()
    axes[1].loglog(masses, taus)
    axes[1].set_xlabel("mass (kg)")
    axes[1].set_ylabel("Diosi-Penrose tau (s)")
    axes[1].set_title("DP collapse time, R = 1 um sphere")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "collapse.png"), dpi=120)
    plt.close(fig)
    dp_example = collapse.dp_collapse_time(1e-12, 1e-6)
    print(
        f"Collapse experiment (toy 1-D wavepacket, 200 steps): "
        f"{traj['n_hits']} GRW hit(s) fired, norm stayed within "
        f"[{traj['norm_history'].min():.6f}, {traj['norm_history'].max():.6f}] "
        f"(unitary evolution conserves it). The Diosi-Penrose time for a "
        f"1e-12 kg, 1 um sphere is {dp_example:.3g} s. The collapse step "
        f"itself was NOT simulated: collapse_oracle() raises "
        f"CollapseNotSimulable at that boundary, per Faizal et al. — the "
        f"GRW hits above are only a phenomenological rule for the "
        f"aftermath of a collapse."
    )
    return {"n_hits": traj["n_hits"], "dp_tau_1e-12kg_s": dp_example}


def run_pilot(out_dir, seed=7):
    """Sober pilot pipeline demo on simulated data."""
    summary = pilot.run_demo(out_dir, seed=seed)
    rng = np.random.default_rng(seed)
    img = pilot.synth_speckle(256, 4.0, rng)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title("Synthetic speckle (fully developed, contrast ~1)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pilot_speckle.png"), dpi=120)
    plt.close(fig)
    pc = summary["per_condition"]
    comp = summary["comparison"]
    inv = summary["invariance_demo"]
    print(
        f"Pilot experiment (simulated observers, 12 raters x 30 locations, "
        f"'stored' mode with q = 0.5): Krippendorff's alpha is "
        f"{pc['W']['alpha_nominal']:.3f} in W (live wall) vs "
        f"{pc['S']['alpha_nominal']:.3f} in S (synthetic speckle), "
        f"difference {comp['alpha_diff_a_minus_b']:.3f} "
        f"(permutation p = {comp['perm_p_two_sided']:.4f}). The invariance "
        f"demo gives mean best-r across changed settings of "
        f"{inv['stored']['mean_best_r_changed']:.3f} with a wall-locked "
        f"component vs {inv['conventional']['mean_best_r_changed']:.3f} "
        f"without one. These numbers show the pipeline detecting an "
        f"injected pattern in simulated data only — they are not evidence "
        f"of anything real."
    )
    return summary


def run_superluminal(out_dir, seed=0):
    """Crab photon survival under superluminal LIV, both orders."""
    energies = np.logspace(3, 6.5, 40)  # 1 TeV .. ~3 PeV
    rows = []
    for order in (1, 2):
        sim = superluminal.simulate_crab_survival(
            energies, superluminal.E_PLANCK_GEV, order=order)
        for e, s in zip(sim["energies_gev"],
                        sim["survival_probability"]):
            rows.append({"order": order, "energy_gev": f"{e:.6g}",
                         "survival_probability": f"{s:.6g}"})
    _write_csv(os.path.join(out_dir, "superluminal.csv"),
               ["order", "energy_gev", "survival_probability"], rows)
    fig, ax = plt.subplots()
    for order in (1, 2):
        sim = superluminal.simulate_crab_survival(
            energies, superluminal.E_PLANCK_GEV, order=order)
        ax.semilogx(sim["energies_gev"], sim["survival_probability"],
                    label=f"n={order}, E_LV = E_Pl")
    ax.axvline(1.42e6, color="k", ls="--",
               label="LHAASO 1.42 PeV Crab photon")
    ax.set_xlabel("photon energy (GeV)")
    ax.set_ylabel("survival probability over 2 kpc")
    ax.set_title("Crab photon survival, superluminal LIV at E_Pl")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "superluminal.png"), dpi=120)
    plt.close(fig)
    th1 = superluminal.pair_decay_threshold(superluminal.E_PLANCK_GEV,
                                            order=1)
    th2 = superluminal.pair_decay_threshold(superluminal.E_PLANCK_GEV,
                                            order=2)
    print(
        f"Superluminal experiment (Crab, 2 kpc, E_LV = E_Pl): the "
        f"pair-decay threshold is {th1:.3g} GeV at n=1 and {th2:.3g} "
        f"GeV at n=2. LHAASO's 1.42 PeV Crab photon is far above the "
        f"n=1 threshold (it would have decayed) and far below the n=2 "
        f"threshold — so a Planck-scale superluminal n=1 model is dead "
        f"while a Planck-scale n=2 model survives pair decay (photon "
        f"splitting is the n=2 constraint; see compare)."
    )
    return {"threshold_n1_gev": th1, "threshold_n2_gev": th2}


def run_birefringence(out_dir, seed=0):
    """GRB polarisation rotation under vacuum birefringence."""
    energies = np.linspace(0.1, 0.5, 25)  # 100 .. 500 keV, in MeV
    energies_gev = energies / 1000.0
    rows = []
    for eta in (1e-16, 1e-15, 1e-14):
        sim = birefringence.simulate_grb_polarimetry(
            energies_gev, 1000.0, eta, p0=0.6)
        for e, r in zip(sim["energies_gev"], sim["rotation_rad"]):
            rows.append({"eta": f"{eta:.6g}", "energy_gev": f"{e:.6g}",
                         "rotation_rad": f"{r:.6g}",
                         "polarization_fraction":
                             f"{sim['polarization_fraction']:.6g}"})
    _write_csv(os.path.join(out_dir, "birefringence.csv"),
               ["eta", "energy_gev", "rotation_rad",
                "polarization_fraction"], rows)
    fig, ax = plt.subplots()
    for eta in (1e-16, 1e-15, 1e-14):
        sim = birefringence.simulate_grb_polarimetry(
            energies_gev, 1000.0, eta, p0=0.6)
        ax.plot(sim["energies_gev"] * 1000.0, sim["rotation_rad"],
                label=f"eta = {eta:.0e}")
    ax.axhline(np.pi / 2, color="k", ls="--",
               label="pi/2 (band scrambled)")
    ax.set_xlabel("photon energy (MeV)")
    ax.set_ylabel("polarisation rotation (rad)")
    ax.set_title("GRB polarisation rotation, 1 Gpc (simulated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "birefringence.png"), dpi=120)
    plt.close(fig)
    p_bound = birefringence.band_polarization(energies_gev, 1000.0,
                                              1e-15, p0=0.6)
    p_big = birefringence.band_polarization(energies_gev, 1000.0,
                                            1e-14, p0=0.6)
    print(
        f"Birefringence experiment (simulated GRB, 100-500 keV band, "
        f"1 Gpc, intrinsic polarisation 0.6): at eta = 1e-16 the "
        f"surviving polarisation is "
        f"{birefringence.band_polarization(energies_gev, 1000.0, 1e-16, p0=0.6):.3f}, "
        f"at the published bound eta = 1e-15 it is {p_bound:.3f} — "
        f"this wide band is already substantially scrambled there, "
        f"which is why the Wei et al. bound comes from energy-resolved "
        f"fits of the rotation vs energy rather than band-averaged "
        f"survival — and eta = 1e-14 leaves only {p_big:.3f}. Real "
        f"GRBs are observed polarised, which is why eta above ~1e-15 "
        f"is excluded."
    )
    return {"polarization_at_bound": p_bound,
            "polarization_at_10x_bound": p_big}


def run_gup(out_dir, seed=0):
    """GUP oscillator levels, frequency shift, and black-hole temperature."""
    ns = np.arange(0, 6)
    levels = gup.oscillator_levels(ns, 2 * np.pi * 1e4, 0.3, beta0=1.0)
    amps = np.logspace(-9, -7, 25)  # 1 nm .. 100 nm (perturbative regime)
    shifts = [gup.frequency_shift_fraction(a, 0.3, 2 * np.pi * 1e4,
                                           beta0=5.2e6) for a in amps]
    masses = np.logspace(9, 15, 25)  # 1e9 .. 1e15 kg
    temps = [gup.hawking_temperature(m, beta0=1.0) for m in masses]
    _write_csv(os.path.join(out_dir, "gup.csv"),
               ["amplitude_m", "freq_shift_fraction_at_beta0_bound",
                "mass_kg", "hawking_T_K_at_beta0_1"],
               [{"amplitude_m": f"{a:.6g}",
                 "freq_shift_fraction_at_beta0_bound": f"{s:.6g}",
                 "mass_kg": f"{m:.6g}",
                 "hawking_T_K_at_beta0_1": f"{t:.6g}" if t else "remnant"}
                for a, s, m, t in zip(amps, shifts, masses, temps)])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].loglog(amps, shifts)
    axes[0].set_xlabel("oscillation amplitude (m)")
    axes[0].set_ylabel("Delta_w / w_0 at beta_0 = 5.2e6")
    axes[0].set_title("GUP frequency shift, 0.3 kg, 10 kHz")
    axes[1].loglog(masses, [t if t else np.nan for t in temps])
    axes[1].set_xlabel("black-hole mass (kg)")
    axes[1].set_ylabel("GUP Hawking temperature (K)")
    axes[1].set_title("GUP black-hole temperature, beta_0 = 1")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gup.png"), dpi=120)
    plt.close(fig)
    print(
        f"GUP experiment (0.3 kg resonator at 10 kHz, black holes "
        f"1e9-1e15 kg): the minimal length at beta_0 = 1 is "
        f"{gup.minimal_length(1.0):.3g} m (the Planck length). At the "
        f"published bound beta_0 = 5.2e6, a 1 nm oscillation amplitude "
        f"gives a fractional frequency shift of "
        f"{gup.frequency_shift_fraction(1e-9, 0.3, 2 * np.pi * 1e4, 5.2e6):.3g} "
        f"— the size of effect a resonator experiment can see. The "
        f"GUP black-hole temperature at 1e12 kg is "
        f"{gup.hawking_temperature(1e12, 1.0):.3g} K, essentially the "
        f"ordinary Hawking temperature at that mass."
    )
    return {"minimal_length_m": gup.minimal_length(1.0),
            "levels_j": levels.tolist()}


def run_csl(out_dir, seed=0):
    """CSL collapse rates and heating vs the published bounds."""
    masses = np.logspace(-18, -9, 25)
    rates = [collapse.csl_rate(m) for m in masses]
    heat = [collapse.csl_heating_power(m) for m in masses]
    _write_csv(os.path.join(out_dir, "csl.csv"),
               ["mass_kg", "csl_rate_per_s", "csl_heating_W"],
               [{"mass_kg": f"{m:.6g}", "csl_rate_per_s": f"{r:.6g}",
                 "csl_heating_W": f"{h:.6g}"}
                for m, r, h in zip(masses, rates, heat)])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].loglog(masses, rates)
    axes[0].set_xlabel("mass (kg)")
    axes[0].set_ylabel("CSL collapse rate (1/s)")
    axes[0].set_title("CSL rate, lambda = 1e-16 /s, r_C = 1e-7 m")
    axes[1].loglog(masses, heat)
    axes[1].set_xlabel("mass (kg)")
    axes[1].set_ylabel("CSL heating power (W)")
    axes[1].set_title("CSL spontaneous heating")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "csl.png"), dpi=120)
    plt.close(fig)
    grw = constraints.check_csl(1e-16, 1e-7)
    print(
        f"CSL experiment (standard values lambda = 1e-16 /s, "
        f"r_C = 1e-7 m): the collapse rate for a 1e-12 kg object is "
        f"{collapse.csl_rate(1e-12):.3g} /s and the spontaneous heating "
        f"is {collapse.csl_heating_power(1e-12):.3g} W. Against the "
        f"published bounds the standard values are {grw['verdict']} — "
        f"lambda/r_C^2 = {grw['lambda_over_rc2']:.3g} s^-1 m^-2 vs the "
        f"XENONnT bound 3.0e-3. The collapse step itself was NOT "
        f"simulated: collapse_oracle() still raises CollapseNotSimulable."
    )
    return {"check_grw_values": grw}


def run_neutrinos(out_dir, seed=0):
    """SN1987A neutrino delays under LIV, both orders."""
    energies_mev = np.linspace(7.5, 40.0, 40)
    rows = []
    for order in (1, 2):
        sim = neutrinos.simulate_sn1987a(
            energies_mev, neutrinos.E_PLANCK_GEV, order=order)
        for e, d in zip(sim["energies_mev"], sim["delays_s"]):
            rows.append({"order": order, "energy_mev": f"{e:.6g}",
                         "delay_s": f"{d:.6g}"})
    _write_csv(os.path.join(out_dir, "neutrinos.csv"),
               ["order", "energy_mev", "delay_s"], rows)
    fig, ax = plt.subplots()
    for order in (1, 2):
        sim = neutrinos.simulate_sn1987a(
            energies_mev, neutrinos.E_PLANCK_GEV, order=order)
        ax.plot(sim["energies_mev"], sim["delays_s"],
                label=f"n={order}, M_QG = E_Pl")
    ax.set_xlabel("neutrino energy (MeV)")
    ax.set_ylabel("LIV delay (s)")
    ax.set_title("SN1987A neutrino delays (simulated, 51.4 kpc)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "neutrinos.png"), dpi=120)
    plt.close(fig)
    spread1 = neutrinos.simulate_sn1987a(
        energies_mev, neutrinos.E_PLANCK_GEV, order=1)["spread_s"]
    print(
        f"Neutrino experiment (SN1987A, 51.4 kpc, 7.5-40 MeV, "
        f"M_QG = E_Pl): the energy-dependent spread across the burst "
        f"is {spread1:.3g} s at n=1 — far inside the observed ~13 s "
        f"burst, so a Planck-scale neutrino LIV term survives SN1987A "
        f"easily. The binding neutrino limits are the TXS 0506+056 "
        f"flare (n=1) and the IceCube PeV superluminal bound; see "
        f"compare."
    )
    return {"spread_n1_s": spread1}


EXPERIMENTS = {
    "dispersion": run_dispersion,
    "foam": run_foam,
    "collapse": run_collapse,
    "pilot": run_pilot,
    "superluminal": run_superluminal,
    "birefringence": run_birefringence,
    "gup": run_gup,
    "csl": run_csl,
    "neutrinos": run_neutrinos,
}


def run_compare(out_dir):
    """Compare the package's toy models with the real published limits.

    Evaluates the linear and quadratic dispersion models (eta = 1 at
    the Planck scale) against the GRB time-of-flight limits, both foam
    scalings against the LIGO O1 noise floor (plus the 40-m parameter
    bounds and the Holometer result), and — new in v0.3 — the
    superluminal branch against the LHAASO limits, birefringence
    against GRB polarimetry, the GUP parameter against the resonator
    bound, CSL against LISA Pathfinder + X-ray bounds, the DP cutoff
    against Majorana, and neutrino LIV against SN1987A / TXS 0506+056
    / IceCube. Writes comparison.csv and comparison.png, and prints a
    plain-language verdict per model.
    """
    rows = []
    verdicts = {}

    # --- dispersion: linear toy model, eta = 1 at the Planck scale ---
    disp = constraints.check_dispersion(eta=1.0, order=1)
    verdicts["dispersion_linear_eta1"] = disp["verdict"]
    for p in disp["per_constraint"]:
        rows.append({
            "check": "dispersion_linear",
            "model": "linear MDR, eta=1, E_QG=E_Pl",
            "predicted": f"{disp['effective_scale_gev']:.6g}",
            "predicted_units": "GeV",
            "limit": f"{p['limit_gev']:.6g}",
            "limit_units": "GeV (lower limit)",
            "ratio": f"{disp['effective_scale_gev'] / p['limit_gev']:.6g}",
            "verdict": p["verdict"],
            "source": p["source"],
        })
    # --- dispersion: quadratic toy model, eta = 1 at the Planck scale ---
    disp2 = constraints.check_dispersion(eta=1.0, order=2)
    verdicts["dispersion_quadratic"] = disp2["verdict"]
    for p in disp2["per_constraint"]:
        rows.append({
            "check": "dispersion_quadratic",
            "model": "quadratic MDR, eta=1, E_QG=E_Pl",
            "predicted": f"{disp2['effective_scale_gev']:.6g}",
            "predicted_units": "GeV",
            "limit": f"{p['limit_gev']:.6g}",
            "limit_units": "GeV (lower limit)",
            "ratio": f"{disp2['effective_scale_gev'] / p['limit_gev']:.6g}",
            "verdict": p["verdict"],
            "source": p["source"],
        })

    # --- foam: both scalings vs the LIGO O1 floor ---
    for model in foam.MODELS:
        res = constraints.check_foam(model)
        verdicts[f"foam_{model}"] = res["verdict"]
        if model == "random_walk":
            rows.append({
                "check": "foam_ligo_psd",
                "model": "random-walk foam, 4 km arm, 100 Hz",
                "predicted": f"{res['predicted_asd_m_per_rt_hz']:.6g}",
                "predicted_units": "m/sqrt(Hz)",
                "limit": f"{res['ligo_floor_m_per_rt_hz']:.6g}",
                "limit_units": "m/sqrt(Hz) (LIGO O1 floor)",
                "ratio": f"{res['ratio']:.6g}",
                "verdict": res["verdict"],
                "source": "Abbott et al., PRL 116, 131103 (2016)",
            })
        else:
            rows.append({
                "check": "foam_ligo_rms",
                "model": "holographic foam, 4 km arm",
                "predicted": f"{res['predicted_rms_m']:.6g}",
                "predicted_units": "m (RMS)",
                "limit": f"{res['ligo_floor_1hz_band_m']:.6g}",
                "limit_units": "m (LIGO floor, 1 Hz band)",
                "ratio": f"{res['ratio']:.6g}",
                "verdict": res["verdict"],
                "source": "Abbott et al., PRL 116, 131103 (2016)",
            })
        res40 = constraints.check_foam_40m(model)
        verdicts[f"foam_40m_{model}"] = res40["verdict"]
        rows.append({
            "check": "foam_40m_parameter_bound",
            "model": f"{model} foam, l_QG = l_P",
            "predicted": f"{res40['model_length_scale_m']:.6g}",
            "predicted_units": "m",
            "limit": f"{res40['bound_m']:.6g}",
            "limit_units": "m (upper bound on l_QG)",
            "ratio": f"{res40['ratio']:.6g}",
            "verdict": res40["verdict"],
            "source": "Ng & van Dam, arXiv:gr-qc/9906003",
        })

    # --- Holometer: Hogan's shear-symmetry variant ---
    holo = constraints.check_holometer()
    verdicts["holometer_hogan_shear"] = holo["verdict"]
    rows.append({
        "check": "holometer_shear",
        "model": "Hogan shear-symmetry holographic noise",
        "predicted": "1.0",
        "predicted_units": "fraction of Planck-scale prediction",
        "limit": "0.1",
        "limit_units": "fraction probed (2 sigma)",
        "ratio": "10",
        "verdict": holo["verdict"],
        "source": holo["source"],
    })

    # --- superluminal branch: Planck-scale model vs LHAASO ---
    for order in (1, 2):
        sup = constraints.check_superluminal_photon(
            superluminal.E_PLANCK_GEV, order=order)
        verdicts[f"superluminal_photon_n{order}"] = sup["verdict"]
        for p in sup["per_constraint"]:
            rows.append({
                "check": f"superluminal_photon_n{order}",
                "model": f"superluminal photon MDR, n={order}, E_LV=E_Pl",
                "predicted": f"{sup['e_lv_gev']:.6g}",
                "predicted_units": "GeV",
                "limit": f"{p['limit_gev']:.6g}",
                "limit_units": "GeV (lower limit)",
                "ratio": f"{sup['e_lv_gev'] / p['limit_gev']:.6g}",
                "verdict": p["verdict"],
                "source": p["source"],
            })
    sup_e = constraints.check_superluminal_electron(
        superluminal.E_PLANCK_GEV, order=1)
    verdicts["superluminal_electron_n1"] = sup_e["verdict"]
    for p in sup_e["per_constraint"]:
        rows.append({
            "check": "superluminal_electron_n1",
            "model": "superluminal electron MDR, n=1, E_LV=E_Pl",
            "predicted": f"{sup_e['e_lv_gev']:.6g}",
            "predicted_units": "GeV",
            "limit": f"{p['limit_gev']:.6g}",
            "limit_units": "GeV (lower limit)",
            "ratio": f"{sup_e['e_lv_gev'] / p['limit_gev']:.6g}",
            "verdict": p["verdict"],
            "source": p["source"],
        })

    # --- birefringence: eta = 1 benchmark vs GRB polarimetry ---
    bir = constraints.check_birefringence(eta=1.0)
    verdicts["birefringence_eta1"] = bir["verdict"]
    for p in bir["per_constraint"]:
        rows.append({
            "check": "birefringence",
            "model": "vacuum birefringence, eta=1",
            "predicted": "1",
            "predicted_units": "dimensionless (eta)",
            "limit": f"{p['limit']:.6g}",
            "limit_units": "dimensionless (upper limit on |eta|)",
            "ratio": f"{1.0 / p['limit']:.6g}",
            "verdict": p["verdict"],
            "source": p["source"],
        })

    # --- GUP: beta_0 = 1 (Planck-length minimal length) vs resonator ---
    gup_res = constraints.check_gup(beta0=1.0)
    verdicts["gup_beta0_1"] = gup_res["verdict"]
    for p in gup_res["per_constraint"]:
        rows.append({
            "check": "gup",
            "model": "GUP minimal length, beta_0=1",
            "predicted": "1",
            "predicted_units": "dimensionless (beta_0)",
            "limit": f"{p['limit']:.6g}",
            "limit_units": "dimensionless (upper limit on beta_0)",
            "ratio": f"{1.0 / p['limit']:.6g}",
            "verdict": p["verdict"],
            "source": p["source"],
        })

    # --- CSL: standard GRW values vs LISA Pathfinder + X-ray ---
    csl_res = constraints.check_csl(1e-16, 1e-7)
    verdicts["csl_grw_values"] = csl_res["verdict"]
    for p in csl_res["per_constraint"]:
        rows.append({
            "check": "csl",
            "model": "CSL, lambda=1e-16 /s, r_C=1e-7 m (GRW values)",
            "predicted": f"{p['ratio_value']:.6g}",
            "predicted_units": p["limit_units"],
            "limit": f"{p['limit']:.6g}",
            "limit_units": p["limit_units"],
            "ratio": f"{p['ratio_value'] / p['limit']:.6g}",
            "verdict": p["verdict"],
            "source": p["source"],
        })

    # --- Diosi-Penrose: parameter-free model (R_0 = 0) vs Majorana ---
    dp_res = constraints.check_dp_cutoff(0.0)
    verdicts["dp_parameter_free"] = dp_res["verdict"]
    for p in dp_res["per_constraint"]:
        rows.append({
            "check": "dp_cutoff",
            "model": "Diosi-Penrose, parameter-free (R_0 = 0)",
            "predicted": "0",
            "predicted_units": "m (R_0)",
            "limit": f"{p['limit']:.6g}",
            "limit_units": "m (lower limit on R_0)",
            "ratio": "0",
            "verdict": p["verdict"],
            "source": p["source"],
        })

    # --- neutrinos: Planck-scale LIV vs SN1987A / TXS / IceCube ---
    for order in (1, 2):
        nu = constraints.check_neutrino(
            neutrinos.E_PLANCK_GEV, order=order, branch="subluminal")
        verdicts[f"neutrino_subluminal_n{order}"] = nu["verdict"]
        for p in nu["per_constraint"]:
            rows.append({
                "check": f"neutrino_subluminal_n{order}",
                "model": f"neutrino LIV, n={order}, M_QG=E_Pl (subluminal)",
                "predicted": f"{nu['m_qg_gev']:.6g}",
                "predicted_units": "GeV",
                "limit": f"{p['limit_gev']:.6g}",
                "limit_units": "GeV (lower limit)",
                "ratio": f"{nu['m_qg_gev'] / p['limit_gev']:.6g}",
                "verdict": p["verdict"],
                "source": p["source"],
            })
    nu_sup = constraints.check_neutrino(
        neutrinos.E_PLANCK_GEV, order=1, branch="superluminal")
    verdicts["neutrino_superluminal_n1"] = nu_sup["verdict"]
    for p in nu_sup["per_constraint"]:
        rows.append({
            "check": "neutrino_superluminal_n1",
            "model": "neutrino LIV, n=1, M_QG=E_Pl (superluminal)",
            "predicted": f"{nu_sup['m_qg_gev']:.6g}",
            "predicted_units": "GeV",
            "limit": f"{p['limit_gev']:.6g}",
            "limit_units": "GeV (lower limit)",
            "ratio": f"{nu_sup['m_qg_gev'] / p['limit_gev']:.6g}",
            "verdict": p["verdict"],
            "source": p["source"],
        })
    nu_speed = constraints.check_neutrino_speed(
        neutrinos.speed_deviation(0.01, neutrinos.E_PLANCK_GEV, order=1))
    verdicts["neutrino_speed_sn1987a"] = nu_speed["verdict"]
    for p in nu_speed["per_constraint"]:
        rows.append({
            "check": "neutrino_speed",
            "model": "neutrino LIV, n=1, M_QG=E_Pl, at 10 MeV",
            "predicted": f"{nu_speed['delta']:.6g}",
            "predicted_units": "dimensionless (|delta|)",
            "limit": f"{p['limit']:.6g}",
            "limit_units": "dimensionless (upper limit on |delta|)",
            "ratio": f"{nu_speed['delta'] / p['limit']:.6g}",
            "verdict": p["verdict"],
            "source": p["source"],
        })

    _write_csv(os.path.join(out_dir, "comparison.csv"),
               ["check", "model", "predicted", "predicted_units", "limit",
                "limit_units", "ratio", "verdict", "source"], rows)

    # --- plot: eight panels, model predictions vs exclusion limits ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # Panel 1: linear LIV, E_QG in units of E_Pl (log scale).
    lin = [p for p in disp["per_constraint"]]
    labels = ["Abdo\n2009", "Vasileiou\n2013", "LHAASO\n2024", "Yang\n2024"]
    limits_epl = [p["limit_e_pl"] for p in lin]
    axes[0, 0].bar(labels, limits_epl, color="#c0392b", alpha=0.75,
                   label="published lower limit")
    axes[0, 0].axhline(disp["effective_scale_e_pl"], color="k", ls="--",
                       label="toy model: eta=1 at E_Pl")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("E_QG,1 (Planck energies)")
    axes[0, 0].set_title("Linear LIV vs GRB limits\n(below a bar = excluded)")
    axes[0, 0].legend(fontsize=8)
    # Panel 2: quadratic LIV, same units.
    quad = [p for p in disp2["per_constraint"]]
    qlabels = ["Vasileiou\n2013", "LHAASO\n2024", "Yang\n2024"]
    qlimits_epl = [p["limit_e_pl"] for p in quad]
    axes[0, 1].bar(qlabels, qlimits_epl, color="#2e86c1", alpha=0.75,
                   label="published lower limit")
    axes[0, 1].axhline(disp2["effective_scale_e_pl"], color="k", ls="--",
                       label="toy model: eta=1 at E_Pl")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("E_QG,2 (Planck energies)")
    axes[0, 1].set_title("Quadratic LIV vs GRB limits\n(model line far above = survives)")
    axes[0, 1].legend(fontsize=8)
    # Panel 3: superluminal photon limits (GeV, log scale).
    sup1 = constraints.check_superluminal_photon(
        superluminal.E_PLANCK_GEV, order=1)
    sup2 = constraints.check_superluminal_photon(
        superluminal.E_PLANCK_GEV, order=2)
    sup_labels = ["n=1\nLHAASO", "n=1\nHe & Ma", "n=2\nLHAASO"]
    sup_limits = [p["limit_gev"] for p in
                  sup1["per_constraint"] + sup2["per_constraint"]]
    axes[0, 2].bar(sup_labels, sup_limits, color="#c0392b", alpha=0.75,
                   label="published lower limit")
    axes[0, 2].axhline(superluminal.E_PLANCK_GEV, color="k", ls="--",
                       label="toy model: E_LV = E_Pl")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylabel("E_LV (GeV)")
    axes[0, 2].set_title("Superluminal photons vs LHAASO\n(n=1 dead, n=2 survives)")
    axes[0, 2].legend(fontsize=8)
    # Panel 4: foam RMS fluctuation at 4 km vs LIGO floor (1 Hz band).
    rw_rms = float(foam.fluctuation(4000.0, model="random_walk"))
    ho_rms = float(foam.fluctuation(4000.0, model="holographic"))
    floor = constraints.ligo_displacement_floor()
    axes[0, 3].bar(["random-walk\nRMS", "holographic\nRMS"], [rw_rms, ho_rms],
                   color=["#c0392b", "#2e86c1"], alpha=0.75)
    axes[0, 3].axhline(floor, color="k", ls="--",
                       label="LIGO O1 floor, 1 Hz band")
    axes[0, 3].set_yscale("log")
    axes[0, 3].set_ylabel("distance fluctuation (m)")
    axes[0, 3].set_title("Foam at 4 km vs LIGO floor\n(above the line = seen)")
    axes[0, 3].legend(fontsize=8)
    # Panel 4: birefringence rotation spread vs eta (100-500 keV, 1 Gpc).
    etas = np.logspace(-17, -13, 25)
    band = np.linspace(0.1, 0.5, 25) / 1000.0  # GeV
    spreads = [birefringence.rotation_difference(
        band.min(), band.max(), 1000.0, e) for e in etas]
    axes[1, 0].loglog(etas, spreads, color="#8e44ad")
    axes[1, 0].axhline(np.pi / 2, color="k", ls="--",
                       label="pi/2 (band scrambled)")
    axes[1, 0].axvline(1e-15, color="#c0392b", ls=":",
                       label="Wei et al. bound")
    axes[1, 0].set_xlabel("birefringence eta")
    axes[1, 0].set_ylabel("rotation spread across band (rad)")
    axes[1, 0].set_title("Birefringence: GRB band rotation\n(above the line = depolarised)")
    axes[1, 0].legend(fontsize=8, loc="lower right")
    # Panel 5: CSL exclusion in the (r_C, lambda) plane.
    rcs = np.logspace(-9, -4, 100)
    axes[1, 1].loglog(rcs, 3.0e-3 * rcs**2, color="#c0392b",
                      label="XENONnT X-ray bound")
    axes[1, 1].loglog(rcs, 0.494 * rcs**2, color="#e67e22",
                      label="Majorana X-ray bound")
    axes[1, 1].scatter([1e-7], [3e-8], color="k", zorder=5,
                       label="LISA Pathfinder bound (at r_C=1e-7 m)")
    axes[1, 1].scatter([1e-7], [1e-16], color="#2e86c1", s=60, zorder=5,
                       label="GRW values (excluded by XENONnT)")
    axes[1, 1].set_xlabel("r_C (m)")
    axes[1, 1].set_ylabel("lambda (1/s)")
    axes[1, 1].set_title("CSL parameter space\n(above a line = excluded)")
    axes[1, 1].legend(fontsize=7, loc="lower right")
    # Panel 6: neutrino LIV limits (GeV, log scale).
    nu_labels = ["SN1987A\nn=1", "TXS\nn=1", "IceCube\nsuperluminal n=1"]
    nu_limits = [2.7e10, 1.0e14, 1.0e5 * constraints.E_PLANCK_GEV]
    axes[1, 2].bar(nu_labels, nu_limits, color="#2e86c1", alpha=0.75,
                   label="published lower limit")
    axes[1, 2].axhline(neutrinos.E_PLANCK_GEV, color="k", ls="--",
                       label="toy model: M_QG = E_Pl")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_ylabel("M_QG (GeV)")
    axes[1, 2].set_title("Neutrino LIV vs SN1987A / TXS / IceCube\n(subluminal survives, superluminal dead)")
    axes[1, 2].legend(fontsize=8)
    # Panel 8: GUP beta_0 bound vs the Planck-length model.
    axes[1, 3].bar(["Bushev 2019\nbound", "toy model\nbeta_0 = 1"],
                   [5.2e6, 1.0], color=["#c0392b", "#2e86c1"], alpha=0.75)
    axes[1, 3].set_yscale("log")
    axes[1, 3].set_ylabel("beta_0 (dimensionless)")
    axes[1, 3].set_title("GUP: resonator bound vs model\n(model far below bound = survives)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=120)
    plt.close(fig)

    rw_asd = constraints.random_walk_psd_asd(100.0)
    ligo_floor = constraints.ligo_displacement_floor()
    print(
        f"Comparison against real published limits (v0.3):\n"
        f"  Linear dispersion, eta = 1 at the Planck scale: "
        f"{disp['verdict']} — effective E_QG,1 = "
        f"{disp['effective_scale_e_pl']:.2f} E_Pl, below every published "
        f"limit (1.2, 7.6, 10 and 12.3 E_Pl). The Planck-scale linear "
        f"model is dead; a linear effect would need E_QG,1 > 12.3 E_Pl.\n"
        f"  Quadratic dispersion: {disp2['verdict']} — the toy model "
        f"now has a real quadratic term (eta = 1, E_QG = E_Pl), and the "
        f"best quadratic limit is 1.2e12 GeV, four orders below the "
        f"Planck scale. A Planck-scale quadratic term is allowed; the "
        f"data only exclude E_QG,2 below 1.2e12 GeV.\n"
        f"  Random-walk foam at 4 km: "
        f"{verdicts['foam_random_walk']} — predicted noise "
        f"{rw_asd:.3g} m/sqrt(Hz) at "
        f"100 Hz vs LIGO's floor "
        f"{ligo_floor:.3g} m/sqrt(Hz), a "
        f"factor of ~{rw_asd / ligo_floor:.0f}. "
        f"LIGO would have seen it.\n"
        f"  Holographic foam at 4 km: "
        f"{verdicts['foam_holographic']} — RMS fluctuation "
        f"{ho_rms:.3g} m, ~{ligo_floor / ho_rms:.0f}x below LIGO's floor "
        f"in a 1 Hz band. "
        f"Below sensitivity, not tested.\n"
        f"  Hogan shear-symmetry holographic noise: "
        f"{holo['verdict']} by the Fermilab Holometer (final data probe "
        f"~10x below the Planck-scale prediction). That is Hogan's "
        f"specific model, not the generic holographic scaling and not "
        f"the holographic principle.\n"
        f"  40-m parameter bounds: both foam models SURVIVE — a "
        f"Planck-scale length parameter passes the Ng & van Dam bounds.\n"
        f"  Superluminal photons, E_LV = E_Pl: n=1 "
        f"{verdicts['superluminal_photon_n1']} — the pair-decay "
        f"threshold at the Planck scale is "
        f"{superluminal.pair_decay_threshold(superluminal.E_PLANCK_GEV, order=1):.3g} "
        f"GeV, far below LHAASO's 1.42 PeV Crab photon, which would "
        f"have decayed. n=2 "
        f"{verdicts['superluminal_photon_n2']} — the splitting bound "
        f"is only 1e-3 M_Pl, and the pair-decay threshold at E_Pl is "
        f"{superluminal.pair_decay_threshold(superluminal.E_PLANCK_GEV, order=2):.3g} "
        f"GeV, above the observed photon. Superluminal electrons, "
        f"E_LV = E_Pl: {verdicts['superluminal_electron_n1']} — the "
        f"Crab's 2.3 PeV electron needs E_LV > 9.4e25 GeV.\n"
        f"  Birefringence, eta = 1: {verdicts['birefringence_eta1']} — "
        f"GRB polarimetry bounds |eta| < 1e-15 (Wei et al. 2025); a "
        f"Planck-scale birefringence would scramble every GRB band.\n"
        f"  GUP, beta_0 = 1: {verdicts['gup_beta0_1']} — the resonator "
        f"bound is beta_0 < 5.2e6 (Bushev et al. 2019), six orders "
        f"above a Planck-length minimal length. The experiment cannot "
        f"see beta_0 = 1; it only rules out much larger values.\n"
        f"  CSL, standard GRW values (lambda = 1e-16 /s, r_C = 1e-7 m): "
        f"{verdicts['csl_grw_values']} — lambda/r_C^2 = "
        f"{csl_res['lambda_over_rc2']:.3g} s^-1 m^-2 vs the XENONnT "
        f"X-ray bound 3.0e-3 (2025). The newest X-ray bound kills the "
        f"textbook GRW values; LISA Pathfinder alone (lambda < 3e-8 "
        f"/s at r_C = 1e-7 m) would have passed them.\n"
        f"  Diosi-Penrose, parameter-free (R_0 = 0): "
        f"{verdicts['dp_parameter_free']} — Majorana bounds "
        f"R_0 > 2.54e-10 m, so a cutoff-free DP model is excluded.\n"
        f"  Neutrinos, M_QG = E_Pl: subluminal n=1 "
        f"{verdicts['neutrino_subluminal_n1']} and n=2 "
        f"{verdicts['neutrino_subluminal_n2']} — SN1987A and the TXS "
        f"0506+056 flare only reach 1e14 GeV, five orders below E_Pl. "
        f"Superluminal n=1 "
        f"{verdicts['neutrino_superluminal_n1']} — IceCube's PeV "
        f"neutrinos need M_QG > ~1e5 M_Pl. SN1987A speed "
        f"{verdicts['neutrino_speed_sn1987a']} — the 10 MeV deviation "
        f"at E_Pl is {nu_speed['delta']:.3g}, far below 4e-9."
    )
    return verdicts


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gph-qgp",
        description="GPH Quantum Gravity Phenomenology lab — toy experiments "
                    "on simulated data.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one experiment")
    run.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--seed", type=int, default=0)
    compare = sub.add_parser("compare", help="compare toy models with real "
                             "published limits")
    compare.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "compare":
        run_compare(args.out)
    else:
        EXPERIMENTS[args.experiment](args.out, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
