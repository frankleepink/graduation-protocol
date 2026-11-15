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

from . import collapse, dispersion, foam, pilot


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
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    EXPERIMENTS[args.experiment](args.out, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
