"""Command-line runner for the GPH Quantum Gravity Phenomenology lab.

Usage:
    python -m gph_qgp run --experiment {dispersion,foam,collapse,pilot} --out <dir>

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

from . import collapse, constraints, dispersion, foam, pilot


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


EXPERIMENTS = {
    "dispersion": run_dispersion,
    "foam": run_foam,
    "collapse": run_collapse,
    "pilot": run_pilot,
}


def run_compare(out_dir):
    """Compare the package's toy models with the real published limits.

    Evaluates the linear dispersion model (eta = 1 at the Planck scale)
    against the GRB time-of-flight limits, and both foam scalings
    against the LIGO O1 noise floor (plus the 40-m parameter bounds and
    the Holometer result). Writes comparison.csv and comparison.png,
    and prints a plain-language verdict per model.
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
    disp2 = constraints.check_dispersion(eta=1.0, order=2)
    verdicts["dispersion_quadratic"] = disp2["verdict"]
    for p in disp2["per_constraint"]:
        rows.append({
            "check": "dispersion_quadratic",
            "model": "linear MDR, eta=1 (no quadratic term)",
            "predicted": "n/a",
            "predicted_units": "GeV",
            "limit": f"{p['limit_gev']:.6g}",
            "limit_units": "GeV (lower limit)",
            "ratio": "n/a",
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

    _write_csv(os.path.join(out_dir, "comparison.csv"),
               ["check", "model", "predicted", "predicted_units", "limit",
                "limit_units", "ratio", "verdict", "source"], rows)

    # --- plot: two panels, model predictions vs exclusion limits ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Panel 1: linear LIV, E_QG in units of E_Pl (log scale).
    lin = [p for p in disp["per_constraint"]]
    labels = ["Abdo\n2009", "Vasileiou\n2013", "LHAASO\n2024", "Yang\n2024"]
    limits_epl = [p["limit_e_pl"] for p in lin]
    axes[0].bar(labels, limits_epl, color="#c0392b", alpha=0.75,
                label="published lower limit")
    axes[0].axhline(disp["effective_scale_e_pl"], color="k", ls="--",
                    label="toy model: eta=1 at E_Pl")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("E_QG,1 (Planck energies)")
    axes[0].set_title("Linear LIV: model vs GRB limits\n(below a bar = excluded)")
    axes[0].legend(fontsize=8)
    # Panel 2: foam RMS fluctuation at 4 km vs LIGO floor (1 Hz band).
    rw_rms = float(foam.fluctuation(4000.0, model="random_walk"))
    ho_rms = float(foam.fluctuation(4000.0, model="holographic"))
    floor = constraints.ligo_displacement_floor()
    axes[1].bar(["random-walk\nRMS", "holographic\nRMS"], [rw_rms, ho_rms],
                color=["#c0392b", "#2e86c1"], alpha=0.75)
    axes[1].axhline(floor, color="k", ls="--",
                    label="LIGO O1 floor, 1 Hz band")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("distance fluctuation (m)")
    axes[1].set_title("Foam at 4 km vs LIGO floor\n(above the line = would have been seen)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=120)
    plt.close(fig)

    rw_asd = constraints.random_walk_psd_asd(100.0)
    ligo_floor = constraints.ligo_displacement_floor()
    print(
        f"Comparison against real published limits (v0.2):\n"
        f"  Linear dispersion, eta = 1 at the Planck scale: "
        f"{disp['verdict']} — effective E_QG,1 = "
        f"{disp['effective_scale_e_pl']:.2f} E_Pl, below every published "
        f"limit (1.2, 7.6, 10 and 12.3 E_Pl). The Planck-scale linear "
        f"model is dead; a linear effect would need E_QG,1 > 12.3 E_Pl.\n"
        f"  Quadratic dispersion: {disp2['verdict']} — the toy model has "
        f"no quadratic term, so the quadratic GRB limits (best "
        f"1.2e12 GeV) do not test it.\n"
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
        f"Planck-scale length parameter passes the Ng & van Dam bounds."
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
