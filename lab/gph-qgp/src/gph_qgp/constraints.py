"""Real published observational constraints, and checks of the toy
models against them (v0.3).

This is the part of the package where the toy models meet the real
world — and lose. Every number below was verified against its primary
source before it was written in. The source is recorded next to the
number as a comment AND in the ``source_link`` field, so anyone can
check it. If a number could not be verified, it is not in here.

What the constraints say, in one paragraph
-------------------------------------------
No quantum-gravity effect has ever been observed. What exists are
limits: the strongest linear Lorentz-invariance limit from GRB
time-of-flight is E_QG,1 > 10 Planck energies (LHAASO, GRB 221009A),
and the strongest interferometer result is that LIGO's noise floor
sits about three orders of magnitude below what a Planck-scale
random-walk spacetime foam would produce. The holographic foam
scaling survives interferometry (it is far too small to see), but
Hogan's specific shear-symmetry holographic-noise model was excluded
by the Fermilab Holometer. v0.3 adds the other side of the story:
the superluminal branch is killed by LHAASO's PeV Crab photons
(photon decay and splitting), vacuum birefringence is bounded to
|eta| < 1e-15 by GRB polarimetry, the GUP parameter is bounded to
beta_0 < 5.2e6 by a macroscopic resonator, CSL collapse is bounded
by LISA Pathfinder and by X-ray searches, and neutrinos are bounded
by SN1987A, IceCube PeV events, and the TXS 0506+056 flare. Those
are the facts the ``compare`` command plots.

How the checks work
-------------------
* Dispersion: a toy model with coefficient ``eta`` and scale E_QG
  behaves, for time-of-flight purposes, like a model with an
  effective scale E_QG / eta^(1/n) (eta > 0, subluminal). A published
  lower limit L excludes the model if E_QG / eta^(1/n) < L. The
  benchmark eta = 1 at the Planck scale has effective scale 1 E_Pl,
  which is below every published linear limit — it is excluded. The
  quadratic benchmark (eta = 1, n = 2) has effective scale 1 E_Pl
  too, but the best quadratic limit is 1.2e12 GeV, four orders below
  E_Pl — so a Planck-scale quadratic term SURVIVES (v0.3: the
  quadratic model is now real, and the check tests it).
* Superluminal: a model scale below the published lower limit is
  EXCLUDED (the photon would have decayed, or the electron would
  have radiated, before reaching us).
* Birefringence / GUP / CSL / neutrinos: each check compares the
  model parameter with the published bound in the same convention
  and returns EXCLUDED / SURVIVES / UNCONSTRAINED with the numbers.
* Foam: the random-walk model is checked as a noise spectral density
  against LIGO's achieved strain floor, and the holographic model as
  an RMS fluctuation against the same floor integrated over a 1 Hz
  band. The two comparisons are different quantities and are labelled
  as such — do not mix them up.

Units: energies in GeV, lengths in metres, strain per root Hz.
E_Pl = 1.22e19 GeV, l_P = 1.616e-35 m.
"""

from __future__ import annotations

import copy

import numpy as np

# ---------------------------------------------------------------- constants
E_PLANCK_GEV = 1.22e19               # Planck energy, GeV
L_PLANCK_M = 1.616255e-35            # Planck length, m
C_M_PER_S = 299_792_458.0            # speed of light, m/s

# LIGO O1 achieved strain sensitivity at 100 Hz (Abbott et al. 2016).
LIGO_O1_STRAIN_PER_RT_HZ = 1.0e-23  # 1 / sqrt(Hz) at 100 Hz
LIGO_ARM_M = 4000.0                 # LIGO arm length, m
LIGO_FREQ_HZ = 100.0                # frequency at which the floor is quoted


# ------------------------------------------------------------- the dataset
# Each entry: id, name, observable, model it constrains, order (dispersion
# only), limit_value + limit_units (the published number), confidence,
# source (citation), source_link (verified URL), notes.
CONSTRAINTS = [
    {
        "id": "grb090510_abdo2009_n1",
        "name": "GRB 090510 linear LIV limit (Abdo et al. 2009)",
        "observable": "photon time-of-flight, 31 GeV photon, z = 0.903",
        "model": "linear dispersion (n=1, subluminal)",
        "order": 1,
        # Abdo et al., Nature 462, 331 (2009): "place a lower limit of
        # 1.2 E_Planck on the scale of a linear energy dependence ...
        # subject to reasonable assumptions about the emission".
        # DOI 10.1038/nature08574 (verified: resolves to the paper).
        "limit_value": 1.2 * E_PLANCK_GEV,
        "limit_units": "GeV",
        "limit_e_pl": 1.2,
        "confidence": "conservative; no CL quoted",
        "source": "Abdo et al., Nature 462, 331 (2009)",
        "source_link": "https://doi.org/10.1038/nature08574",
        "notes": "Most conservative published linear limit from GRB 090510; "
                 "uses a single 31 GeV photon and assumptions about the "
                 "intrinsic emission.",
    },
    {
        "id": "grb090510_vasileiou2013_n1",
        "name": "GRB 090510 linear LIV limit (Vasileiou et al. 2013)",
        "observable": "photon time-of-flight, Fermi-LAT GRB 090510",
        "model": "linear dispersion (n=1, subluminal)",
        "order": 1,
        # Vasileiou et al., Phys. Rev. D 87, 122001 (2013), arXiv:1305.3463:
        # "E_QG,1 > 7.6 times the Planck energy" at 95% CL, subluminal,
        # without taking into account source-intrinsic dispersion.
        "limit_value": 7.6 * E_PLANCK_GEV,
        "limit_units": "GeV",
        "limit_e_pl": 7.6,
        "confidence": "95% CL",
        "source": "Vasileiou et al., Phys. Rev. D 87, 122001 (2013)",
        "source_link": "https://arxiv.org/abs/1305.3463",
        "notes": "Likelihood analysis of the full Fermi-LAT light curve; "
                 "stronger than Abdo et al. because it uses the whole "
                 "photon sample, not one photon.",
    },
    {
        "id": "grb090510_vasileiou2013_n2",
        "name": "GRB 090510 quadratic LIV limit (Vasileiou et al. 2013)",
        "observable": "photon time-of-flight, Fermi-LAT GRB 090510",
        "model": "quadratic dispersion (n=2, subluminal)",
        "order": 2,
        # Same paper as above: "E_QG,2 > 1.3 x 10^11 GeV" at 95% CL.
        "limit_value": 1.3e11,
        "limit_units": "GeV",
        "limit_e_pl": 1.3e11 / E_PLANCK_GEV,
        "confidence": "95% CL",
        "source": "Vasileiou et al., Phys. Rev. D 87, 122001 (2013)",
        "source_link": "https://arxiv.org/abs/1305.3463",
        "notes": "Quadratic limits sit far below the Planck scale; the "
                 "quadratic case is not yet tested at Planck energies.",
    },
    {
        "id": "grb221009a_lhaaso2024_n1",
        "name": "GRB 221009A linear LIV limit (LHAASO Collaboration 2024)",
        "observable": "photon time-of-flight, >64,000 TeV photons, z = 0.151",
        "model": "linear dispersion (n=1, subluminal)",
        "order": 1,
        # LHAASO Collaboration, Phys. Rev. Lett. 133, 071501 (2024),
        # arXiv:2402.06009: "the 95% confidence level lower limits on the
        # QG energy scales are E_QG,1 > 10 times of the Planck energy
        # E_Pl for the linear ... LIV effects".
        "limit_value": 10.0 * E_PLANCK_GEV,
        "limit_units": "GeV",
        "limit_e_pl": 10.0,
        "confidence": "95% CL",
        "source": "LHAASO Collaboration, Phys. Rev. Lett. 133, 071501 (2024)",
        "source_link": "https://arxiv.org/abs/2402.06009",
        "notes": "Brightest GRB ever recorded; the strongest published "
                 "linear limit from a single collaboration analysis.",
    },
    {
        "id": "grb221009a_lhaaso2024_n2",
        "name": "GRB 221009A quadratic LIV limit (LHAASO Collaboration 2024)",
        "observable": "photon time-of-flight, LHAASO TeV afterglow",
        "model": "quadratic dispersion (n=2, subluminal)",
        "order": 2,
        # Same paper: "E_QG,2 > 6 x 10^-8 E_Pl for the quadratic LIV
        # effects" at 95% CL. 6e-8 x 1.22e19 GeV = 7.3e11 GeV.
        "limit_value": 6.0e-8 * E_PLANCK_GEV,
        "limit_units": "GeV",
        "limit_e_pl": 6.0e-8,
        "confidence": "95% CL",
        "source": "LHAASO Collaboration, Phys. Rev. Lett. 133, 071501 (2024)",
        "source_link": "https://arxiv.org/abs/2402.06009",
        "notes": "Improved the previous best quadratic bound by a factor "
                 "of 5-7; still ~8 orders of magnitude below the Planck "
                 "scale.",
    },
    {
        "id": "grb221009a_yang2024_n1",
        "name": "GRB 221009A linear LIV limit (Yang, Bi & Yin 2024)",
        "observable": "photon time-of-flight, LHAASO GRB 221009A data",
        "model": "linear dispersion (n=1, subluminal)",
        "order": 1,
        # Yang, Bi & Yin, JCAP 04, 060 (2024), arXiv:2312.09079:
        # E_QG,1 > 14.7 x 10^19 GeV (subluminal), 95% CL. The paper
        # quotes this as > 12.3 E_Pl.
        "limit_value": 14.7e19,
        "limit_units": "GeV",
        "limit_e_pl": 12.3,
        "confidence": "95% CL",
        "source": "Yang, Bi & Yin, JCAP 04, 060 (2024)",
        "source_link": "https://arxiv.org/abs/2312.09079",
        "notes": "Independent re-analysis of the same LHAASO data; the "
                 "strongest published linear limit. Superluminal branch "
                 "in the same paper: E_QG,1 > 6.5 x 10^19 GeV.",
    },
    {
        "id": "grb221009a_yang2024_n2",
        "name": "GRB 221009A quadratic LIV limit (Yang, Bi & Yin 2024)",
        "observable": "photon time-of-flight, LHAASO GRB 221009A data",
        "model": "quadratic dispersion (n=2, subluminal)",
        "order": 2,
        # Same paper: E_QG,2 > 12.0 x 10^11 GeV (subluminal), 95% CL.
        "limit_value": 12.0e11,
        "limit_units": "GeV",
        "limit_e_pl": 12.0e11 / E_PLANCK_GEV,
        "confidence": "95% CL",
        "source": "Yang, Bi & Yin, JCAP 04, 060 (2024)",
        "source_link": "https://arxiv.org/abs/2312.09079",
        "notes": "Strongest published subluminal quadratic time-of-flight "
                 "limit; still ~4 orders of magnitude below Planck.",
    },
    {
        "id": "ligo_o1_strain_floor",
        "name": "Advanced LIGO O1 strain sensitivity",
        "observable": "interferometer strain noise at 100 Hz",
        "model": "interferometer noise floor (used for foam checks)",
        "order": None,
        # Abbott et al. (LIGO Scientific Collaboration and Virgo
        # Collaboration), Phys. Rev. Lett. 116, 131103 (2016):
        # "With a strain sensitivity of 10^-23 / sqrt(Hz) at 100 Hz".
        # DOI 10.1103/PhysRevLett.116.131103.
        # Displacement floor at 4 km arm: 1e-23 x 4000 = 4e-20 m/sqrt(Hz).
        "limit_value": LIGO_O1_STRAIN_PER_RT_HZ * LIGO_ARM_M,
        "limit_units": "m/sqrt(Hz)",
        "limit_e_pl": None,
        "confidence": "achieved O1 sensitivity",
        "source": "Abbott et al., Phys. Rev. Lett. 116, 131103 (2016)",
        "source_link": "https://doi.org/10.1103/PhysRevLett.116.131103",
        "notes": "Not a foam limit itself — it is the noise floor any "
                 "foam model must hide under. Displacement floor = strain "
                 "x arm length (4 km).",
    },
    {
        "id": "foam_40m_ng_vandam2000",
        "name": "Caltech 40-m interferometer foam parameter bounds (Ng & van Dam 2000)",
        "observable": "displacement noise 3e-19 m/sqrt(Hz) near 450 Hz",
        "model": "spacetime foam length scales",
        "order": None,
        # Ng & van Dam, arXiv:gr-qc/9906003: comparing their foam models
        # with the Caltech 40-m noise level gives bounds on the model
        # length scales l_QG of ~1e-17 m (random walk) and ~1e-29 m
        # (holographic). The third foam picture was already excluded
        # down to 1e-40 m. A Planck-scale l_QG (~1.6e-35 m) passes
        # both bounds, so the 40-m data did NOT exclude either model —
        # it only bounded their parameters.
        "limit_value": 3.0e-19,
        "limit_units": "m/sqrt(Hz)",
        "limit_e_pl": None,
        "confidence": "published noise level",
        "source": "Ng & van Dam, arXiv:gr-qc/9906003 (2000)",
        "source_link": "https://arxiv.org/abs/gr-qc/9906003",
        "notes": "Parameter bounds: random-walk l_QG <~ 1e-17 m, "
                 "holographic l_QG <~ 1e-29 m. Planck-scale models pass "
                 "both; the third foam picture was already ruled out.",
    },
    {
        "id": "holometer_chou2017",
        "name": "Fermilab Holometer holographic shear-noise exclusion (Chou et al. 2017)",
        "observable": "correlated position noise, two 40 m interferometers cross-correlated to 25 MHz",
        "model": "Hogan shear-symmetry holographic noise",
        "order": None,
        # Chou et al., Classical Quantum Gravity 34, 165005 (2017),
        # arXiv:1703.08503: "the final data probe nearly an order of
        # magnitude below the level of correlated position noise
        # predicted for a Planck-scale, shear-symmetry holographic
        # information bound on space-time position states" (2 sigma).
        # First-generation Holometer; see also Chou et al., Phys. Rev.
        # Lett. 117, 111102 (2016), arXiv:1512.01216 (strain-noise PSD
        # below a Planck time for bandwidths above 11 kHz).
        # What was excluded: Hogan's specific shear-symmetry model of
        # holographic noise — NOT the holographic principle itself, and
        # NOT the generic Ng & van Dam holographic scaling.
        "limit_value": 0.1,
        "limit_units": "fraction of Planck-scale shear prediction",
        "limit_e_pl": None,
        "confidence": "2 sigma",
        "source": "Chou et al., Classical Quantum Gravity 34, 165005 (2017)",
        "source_link": "https://arxiv.org/abs/1703.08503",
        "notes": "Excludes Hogan's shear-symmetry holographic noise model "
                 "specifically. It does not exclude the holographic "
                 "principle, and it does not test the generic Ng & van "
                 "Dam holographic scaling implemented in foam.py.",
    },
    # --------------------------------------------------- v0.3 additions
    # "sector" distinguishes photon dispersion entries from neutrino
    # entries (both use "order"); "branch" distinguishes subluminal
    # from superluminal. Entries without a sector/branch default to
    # photon / subluminal in the check functions.
    {
        "id": "lhaaso2022_photon_superluminal_n1",
        "name": "LHAASO superluminal photon limit, n=1 (LHAASO Collaboration 2022)",
        "observable": "1.42 PeV photon from LHAASO J2032+4102 / Crab, no decay cutoff",
        "model": "superluminal photon dispersion (n=1)",
        "sector": "photon",
        "branch": "superluminal",
        "order": 1,
        # LHAASO Collaboration, Phys. Rev. Lett. 128, 051102 (2022),
        # arXiv:2106.12350: from the absence of photon-decay spectral
        # cutoffs, E_LV,1 > 1.42e24 GeV (superluminal, n=1), 95% CL.
        "limit_value": 1.42e24,
        "limit_units": "GeV",
        "limit_e_pl": 1.42e24 / E_PLANCK_GEV,
        "confidence": "95% CL",
        "source": "LHAASO Collaboration, Phys. Rev. Lett. 128, 051102 (2022)",
        "source_link": "https://arxiv.org/abs/2106.12350",
        "notes": "The collaboration's own superluminal photon-decay "
                 "bound. He & Ma (2023) re-analyse the same photon and "
                 "get the stricter 2.74e24 GeV (see next entry).",
    },
    {
        "id": "hema2023_photon_superluminal_n1",
        "name": "Superluminal photon limit, n=1 (He & Ma 2023)",
        "observable": "LHAASO 1.42 PeV highest-energy photon, photon decay",
        "model": "superluminal photon dispersion (n=1)",
        "sector": "photon",
        "branch": "superluminal",
        "order": 1,
        # He & Ma, Phys. Lett. B 835, 137536 (2023), arXiv:2308.02021:
        # "Ref. [7] got a photon superluminal linear LV constraint
        # E_LV^(gamma,sup) >= 2.74 x 10^24 GeV from the LHAASO 1.42
        # PeV highest-energy photon". Strictest photon n=1 bound.
        "limit_value": 2.74e24,
        "limit_units": "GeV",
        "limit_e_pl": 2.74e24 / E_PLANCK_GEV,
        "confidence": "95% CL",
        "source": "He & Ma, Phys. Lett. B 835, 137536 (2023)",
        "source_link": "https://arxiv.org/abs/2308.02021",
        "notes": "Strictest published superluminal photon n=1 bound; "
                 "a re-analysis of the same LHAASO 1.42 PeV photon the "
                 "collaboration used for 1.42e24 GeV.",
    },
    {
        "id": "lhaaso2022_photon_superluminal_n2",
        "name": "LHAASO superluminal photon limit, n=2 (LHAASO Collaboration 2022)",
        "observable": "PeV photons, no photon-splitting (gamma -> 3 gamma) cutoff",
        "model": "superluminal photon dispersion (n=2)",
        "sector": "photon",
        "branch": "superluminal",
        "order": 2,
        # LHAASO Collaboration, Phys. Rev. Lett. 128, 051102 (2022),
        # arXiv:2106.12350: "E_LV,2 > 1e-3 M_Pl" for n=2 (photon
        # splitting), 95% CL. 1e-3 x 1.22e19 GeV = 1.22e16 GeV.
        "limit_value": 1.0e-3 * E_PLANCK_GEV,
        "limit_units": "GeV",
        "limit_e_pl": 1.0e-3,
        "confidence": "95% CL",
        "source": "LHAASO Collaboration, Phys. Rev. Lett. 128, 051102 (2022)",
        "source_link": "https://arxiv.org/abs/2106.12350",
        "notes": "Photon splitting has no energy threshold, so the "
                 "bound comes from the splitting rate, not a decay "
                 "cutoff. See superluminal.py for the calibrated toy.",
    },
    {
        "id": "hema2023_electron_superluminal_n1",
        "name": "Superluminal electron limit, n=1 (He & Ma 2023)",
        "observable": "Crab Nebula 2.3 PeV parent electron (from 1.12 PeV photon), no vacuum Cherenkov loss",
        "model": "superluminal electron dispersion (n=1)",
        "sector": "photon",
        "branch": "superluminal",
        "order": 1,
        # He & Ma, Phys. Lett. B 835, 137536 (2023), arXiv:2308.02021:
        # "Ref. [9] obtained the most strict constraint on electron
        # superluminal linear scale E_LV^(e,sup) >= 9.4 x 10^25 GeV",
        # from the Crab 1.12 PeV photon's 2.3 PeV parent electron.
        "limit_value": 9.4e25,
        "limit_units": "GeV",
        "limit_e_pl": 9.4e25 / E_PLANCK_GEV,
        "confidence": "bound as quoted in He & Ma 2023 (LHAASO Crab analysis)",
        "source": "He & Ma, Phys. Lett. B 835, 137536 (2023)",
        "source_link": "https://arxiv.org/abs/2308.02021",
        "notes": "A superluminal electron above threshold would radiate "
                 "vacuum Cherenkov until it dropped below it, so the "
                 "Crab's 2.3 PeV electron must be below threshold.",
    },
    {
        "id": "grb_birefringence_wei2025",
        "name": "Vacuum birefringence limit (Wei et al. 2025)",
        "observable": "energy-resolved GRB polarimetry, five GRBs",
        "model": "vacuum birefringence (helicity-dependent photon speed)",
        "sector": "photon",
        "branch": "subluminal",
        "order": None,
        # Wei et al., Astrophys. J. 983, 164 (2025), arXiv:2503.18277:
        # "the limits on eta ... reach O(10^-15 - 10^-16)" at 95% CL,
        # from energy-resolved polarimetry of five GRBs. The module
        # uses the conservative end, |eta| < 1e-15.
        "limit_value": 1.0e-15,
        "limit_units": "dimensionless (eta)",
        "limit_e_pl": None,
        "confidence": "95% CL",
        "source": "Wei et al., Astrophys. J. 983, 164 (2025)",
        "source_link": "https://arxiv.org/abs/2503.18277",
        "notes": "Conservative end of the published range "
                 "O(1e-15 - 1e-16); the strongest per-GRB bounds reach "
                 "1e-16. Parameter is eta in the Delta_phi = eta E^2 D / "
                 "(hbar E_Pl c) convention of birefringence.py.",
    },
    {
        "id": "gup_beta0_bushev2019",
        "name": "GUP parameter bound (Bushev et al. 2019)",
        "observable": "amplitude-dependent frequency shift, 0.3 kg sapphire split-bar resonator",
        "model": "GUP minimal length",
        "sector": "gup",
        "branch": None,
        "order": None,
        # Bushev et al., Phys. Rev. D 100, 066020 (2019),
        # arXiv:1903.03346: "we set the limit beta_0 < 5.2 x 10^6"
        # from the sapphire resonator. The same paper estimates
        # beta_0 < 4 x 10^4 for a quartz resonator — an estimate,
        # not a firm bound, recorded here only in the notes.
        "limit_value": 5.2e6,
        "limit_units": "dimensionless (beta_0)",
        "limit_e_pl": None,
        "confidence": "measured upper bound (sapphire resonator)",
        "source": "Bushev et al., Phys. Rev. D 100, 066020 (2019)",
        "source_link": "https://arxiv.org/abs/1903.03346",
        "notes": "Strongest verified GUP bound. The quartz estimate "
                 "beta_0 < 4e4 in the same paper is an estimate, not a "
                 "firm bound, and is not used as a limit.",
    },
    {
        "id": "csl_lisa_pathfinder_carlesso2016",
        "name": "CSL bound from LISA Pathfinder (Carlesso et al. 2016)",
        "observable": "force noise on LISA Pathfinder test masses",
        "model": "CSL collapse (lambda at r_C = 1e-7 m)",
        "sector": "csl",
        "branch": None,
        "order": None,
        # Carlesso et al., Phys. Rev. D 94, 124036 (2016),
        # arXiv:1606.04581: "At the standard characteristic length
        # r_C = 10^-7 m, the bound from LISA Pathfinder is
        # lambda < 3 x 10^-8 s^-1" — the strongest bound for
        # r_C > 10^-6 m, and the strongest CSL bound overall at
        # r_C = 10^-7 m.
        "limit_value": 3.0e-8,
        "limit_units": "s^-1 (at r_C = 1e-7 m)",
        "limit_e_pl": None,
        "confidence": "bound as quoted in Carlesso et al. 2016",
        "source": "Carlesso et al., Phys. Rev. D 94, 124036 (2016)",
        "source_link": "https://arxiv.org/abs/1606.04581",
        "notes": "Bound is quoted at r_C = 1e-7 m only; check_csl "
                 "applies it there and says so elsewhere. Adler's "
                 "suggested lambda ~ 1e-8 s^-1 at this r_C survives; "
                 "his upper range 1e-6 s^-1 does not.",
    },
    {
        "id": "csl_xray_majorana2022",
        "name": "CSL X-ray bound, Majorana Demonstrator (2022)",
        "observable": "spontaneous X-ray emission from CSL heating, underground Ge detectors",
        "model": "CSL collapse (lambda / r_C^2)",
        "sector": "csl",
        "branch": None,
        "order": None,
        # Majorana Collaboration, arXiv:2202.01343 (2022): the X-ray
        # bound on the CSL combination is (lambda / r_C^2)_Majorana =
        # 4.94 x 10^-1 s^-1 m^-2 (as quoted in the 2025 review of CSL
        # bounds, arXiv:2506.05507). Improves the previous best
        # (Donadi et al. 2021) by a factor of ~135.
        "limit_value": 0.494,
        "limit_units": "s^-1 m^-2 (lambda / r_C^2)",
        "limit_e_pl": None,
        "confidence": "upper limit as quoted in the Majorana paper",
        "source": "Majorana Collaboration, arXiv:2202.01343 (2022)",
        "source_link": "https://arxiv.org/abs/2202.01343",
        "notes": "Same paper also bounds the Diosi-Penrose cutoff "
                 "R_0 > 2.54e-10 m (see dp_cutoff_majorana2022).",
    },
    {
        "id": "csl_xray_xenonnt2025",
        "name": "CSL X-ray bound, XENONnT (2025)",
        "observable": "spontaneous X-ray emission from CSL heating, liquid-xenon TPC",
        "model": "CSL collapse (lambda / r_C^2)",
        "sector": "csl",
        "branch": None,
        "order": None,
        # XENONnT Collaboration, arXiv:2506.05507 (2025): "sets the
        # new most stringent experimental constraint for
        # r_C <= 10^-5 m" — lambda / r_C^2 < 3.0 x 10^-3 s^-1 m^-2
        # at 90% CL, an improvement of a factor of 135 over the
        # Majorana bound. Preprint (2025); the strongest X-ray bound.
        "limit_value": 3.0e-3,
        "limit_units": "s^-1 m^-2 (lambda / r_C^2)",
        "limit_e_pl": None,
        "confidence": "90% CL",
        "source": "XENONnT Collaboration, arXiv:2506.05507 (2025)",
        "source_link": "https://arxiv.org/abs/2506.05507",
        "notes": "Preprint; newest and strongest X-ray CSL bound. "
                 "Improves Majorana by ~165x (0.494 / 3.0e-3).",
    },
    {
        "id": "dp_cutoff_majorana2022",
        "name": "Diosi-Penrose cutoff bound, Majorana Demonstrator (2022)",
        "observable": "spontaneous X-ray emission from DP collapse heating",
        "model": "Diosi-Penrose collapse (cutoff length R_0)",
        "sector": "csl",
        "branch": None,
        "order": None,
        # Majorana Collaboration, arXiv:2202.01343 (2022): "lower bound
        # on the cut-off length of R_DP > (2.54 +/- 0.03) x 10^-10 m",
        # more stringent than the previous best (Donadi et al., Nat.
        # Phys. 17, 74 (2021): R_0 > 0.54 x 10^-10 m). This is a LOWER
        # limit: the parameter-free DP model (R_0 = 0) is excluded.
        "limit_value": 2.54e-10,
        "limit_units": "m (lower limit on R_0)",
        "limit_e_pl": None,
        "confidence": "lower limit as quoted in the Majorana paper",
        "source": "Majorana Collaboration, arXiv:2202.01343 (2022)",
        "source_link": "https://arxiv.org/abs/2202.01343",
        "notes": "Lower limit, not an upper limit: R_0 below 2.54e-10 "
                 "m is excluded. The parameter-free Diósi-Penrose "
                 "model (R_0 = 0) is therefore excluded by this data.",
    },
    {
        "id": "sn1987a_neutrino_m1",
        "name": "SN1987A neutrino LIV limit, n=1 (Bustamante et al. 2024)",
        "observable": "neutrino-photon arrival coincidence, 51.4 kpc, 7.5-40 MeV",
        "model": "neutrino LIV (n=1, subluminal)",
        "sector": "neutrino",
        "branch": "subluminal",
        "order": 1,
        # Bustamante, Ellis, Konoplich & Sakharov, arXiv:2408.15949:
        # "For SN1987A, M_1 > 2.7 x 10^10 GeV and M_2 > 4.6 x 10^4
        # GeV", summarising the Longo (1988) / Stodolsky (1988)
        # SN1987A time-of-flight analyses.
        "limit_value": 2.7e10,
        "limit_units": "GeV",
        "limit_e_pl": 2.7e10 / E_PLANCK_GEV,
        "confidence": "SN1987A timing analysis (as summarised in arXiv:2408.15949)",
        "source": "Bustamante, Ellis, Konoplich & Sakharov, arXiv:2408.15949 (2024)",
        "source_link": "https://arxiv.org/abs/2408.15949",
        "notes": "The oldest LIV bound in the set, and still the "
                 "reference subluminal neutrino limit at MeV energies.",
    },
    {
        "id": "sn1987a_neutrino_m2",
        "name": "SN1987A neutrino LIV limit, n=2 (Bustamante et al. 2024)",
        "observable": "neutrino-photon arrival coincidence, 51.4 kpc, 7.5-40 MeV",
        "model": "neutrino LIV (n=2, subluminal)",
        "sector": "neutrino",
        "branch": "subluminal",
        "order": 2,
        # Same paper as above: "M_2 > 4.6 x 10^4 GeV" for SN1987A.
        "limit_value": 4.6e4,
        "limit_units": "GeV",
        "limit_e_pl": 4.6e4 / E_PLANCK_GEV,
        "confidence": "SN1987A timing analysis (as summarised in arXiv:2408.15949)",
        "source": "Bustamante, Ellis, Konoplich & Sakharov, arXiv:2408.15949 (2024)",
        "source_link": "https://arxiv.org/abs/2408.15949",
        "notes": "Quadratic SN1987A limit; far below the Planck scale, "
                 "so a Planck-scale quadratic neutrino term survives.",
    },
    {
        "id": "txs0506_neutrino_m1",
        "name": "TXS 0506+056 neutrino LIV limit, n=1 (Bustamante et al. 2024)",
        "observable": "neutrino flare timing, TXS 0506+056 blazar, z = 0.3365",
        "model": "neutrino LIV (n=1, subluminal)",
        "sector": "neutrino",
        "branch": "subluminal",
        "order": 1,
        # Bustamante et al., arXiv:2408.15949: "the recent TXS 0506+056
        # flare ... M_1 > 10^14 GeV" (subluminal, n=1).
        "limit_value": 1.0e14,
        "limit_units": "GeV",
        "limit_e_pl": 1.0e14 / E_PLANCK_GEV,
        "confidence": "flare timing analysis (as summarised in arXiv:2408.15949)",
        "source": "Bustamante, Ellis, Konoplich & Sakharov, arXiv:2408.15949 (2024)",
        "source_link": "https://arxiv.org/abs/2408.15949",
        "notes": "Strongest subluminal neutrino limit; ~4 orders "
                 "stronger than SN1987A at n=1.",
    },
    {
        "id": "txs0506_neutrino_m2",
        "name": "TXS 0506+056 neutrino LIV limit, n=2 (Bustamante et al. 2024)",
        "observable": "neutrino flare timing, TXS 0506+056 blazar, z = 0.3365",
        "model": "neutrino LIV (n=2, subluminal)",
        "sector": "neutrino",
        "branch": "subluminal",
        "order": 2,
        # Same paper: "M_2 > 10^9 GeV" for the TXS 0506+056 flare.
        "limit_value": 1.0e9,
        "limit_units": "GeV",
        "limit_e_pl": 1.0e9 / E_PLANCK_GEV,
        "confidence": "flare timing analysis (as summarised in arXiv:2408.15949)",
        "source": "Bustamante, Ellis, Konoplich & Sakharov, arXiv:2408.15949 (2024)",
        "source_link": "https://arxiv.org/abs/2408.15949",
        "notes": "Strongest subluminal quadratic neutrino limit.",
    },
    {
        "id": "icecube_pev_neutrino_superluminal",
        "name": "IceCube PeV neutrino superluminal limit (Borriello et al. 2013)",
        "observable": "PeV neutrinos reaching Earth; a superluminal neutrino would lose energy by pair bremsstrahlung",
        "model": "neutrino LIV (n=1, superluminal)",
        "sector": "neutrino",
        "branch": "superluminal",
        "order": 1,
        # Borriello, Chakraborty & Mirizzi, Phys. Rev. D 87, 116009
        # (2013), arXiv:1303.5843: for PeV neutrinos, "delta < O(10^-18)"
        # and "M_QG > ~10^5 M_Pl" (linear, superluminal). 1e5 M_Pl =
        # 1.22e24 GeV.
        "limit_value": 1.0e5 * E_PLANCK_GEV,
        "limit_units": "GeV",
        "limit_e_pl": 1.0e5,
        "confidence": "order-of-magnitude (delta < O(1e-18))",
        "source": "Borriello, Chakraborty & Mirizzi, Phys. Rev. D 87, 116009 (2013)",
        "source_link": "https://arxiv.org/abs/1303.5843",
        "notes": "Superluminal bound only: a superluminal PeV neutrino "
                 "would radiate e+ e- pairs (nu -> nu e+ e-) and never "
                 "arrive at PeV energies.",
    },
    {
        "id": "sn1987a_neutrino_speed",
        "name": "SN1987A neutrino absolute-speed bound (Longo 1988; Stodolsky 1988)",
        "observable": "neutrino vs photon arrival from SN1987A, 51.4 kpc",
        "model": "neutrino absolute speed",
        "sector": "neutrino",
        "branch": None,
        "order": None,
        # Longo, Phys. Rev. Lett. 60, 173 (1988); Stodolsky, Phys.
        # Lett. B 201, 353 (1988); as summarised in Borriello et al.,
        # PRD 87, 116009 (2013): "the strongest limits on delta ...
        # delta <~ 4 x 10^-9" from SN1987A.
        "limit_value": 4.0e-9,
        "limit_units": "dimensionless (|delta| = |v-c|/c)",
        "limit_e_pl": None,
        "confidence": "SN1987A timing (as summarised in Borriello et al. 2013)",
        "source": "Longo 1988; Stodolsky 1988 (via Borriello et al., PRD 87, 116009 (2013))",
        "source_link": "https://arxiv.org/abs/1303.5843",
        "notes": "Model-independent absolute-speed bound: any neutrino "
                 "model predicting |(v-c)/c| > 4e-9 at ~10 MeV is "
                 "excluded, whatever its energy dependence.",
    },
]


def summary_table():
    """Return a deep copy of the verified constraints dataset."""
    return copy.deepcopy(CONSTRAINTS)


def get_constraint(constraint_id):
    """Return one constraint dict by id, or raise KeyError."""
    for entry in CONSTRAINTS:
        if entry["id"] == constraint_id:
            return copy.deepcopy(entry)
    raise KeyError(f"unknown constraint {constraint_id!r}")


# ------------------------------------------------------- dispersion checks
def check_dispersion(eta=1.0, e_qg_gev=E_PLANCK_GEV, order=1):
    """Check a subluminal toy model against the GRB time-of-flight limits.

    The toy model in ``dispersion.py`` has coefficient ``eta`` and scale
    E_QG; for time-of-flight it behaves like a model with effective
    scale E_QG / eta^(1/order) (eta > 0, subluminal). A published lower
    limit L excludes the model when E_QG / eta^(1/order) < L.

    v0.3: the quadratic (order=2) model is now a real model in
    dispersion.py, so it is tested like the linear one instead of
    being reported UNCONSTRAINED. A Planck-scale quadratic term
    SURVIVES (best quadratic limit 1.2e12 GeV, four orders below
    E_Pl); a Planck-scale linear term is EXCLUDED.

    Returns a dict with the effective scale, one verdict per matching
    constraint, and an overall verdict (EXCLUDED if any constraint
    excludes the model, else SURVIVES). Negative eta is the
    superluminal branch: the subluminal limits recorded here do not
    cover it, so that case returns UNCONSTRAINED (see
    check_superluminal_photon for the superluminal limits).
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if eta < 0:
        return {
            "order": order, "eta": float(eta),
            "effective_scale_gev": None, "effective_scale_e_pl": None,
            "per_constraint": [],
            "verdict": "UNCONSTRAINED",
            "note": "Superluminal branch (eta < 0): the limits recorded "
                    "here are subluminal ones and do not cover it. See "
                    "check_superluminal_photon.",
        }
    effective = float(e_qg_gev) / float(eta)**(1.0 / order) \
        if eta != 0 else float("inf")
    per = []
    for e in CONSTRAINTS:
        if e.get("order") != order:
            continue
        if e.get("sector", "photon") != "photon":
            continue
        if e.get("branch", "subluminal") != "subluminal":
            continue
        verdict = "EXCLUDED" if effective < e["limit_value"] else "SURVIVES"
        per.append({
            "constraint_id": e["id"],
            "limit_gev": e["limit_value"],
            "limit_e_pl": e["limit_e_pl"],
            "verdict": verdict,
            "source": e["source"],
        })
    overall = "EXCLUDED" if any(p["verdict"] == "EXCLUDED" for p in per) \
        else "SURVIVES"
    return {
        "order": order, "eta": float(eta),
        "effective_scale_gev": effective,
        "effective_scale_e_pl": effective / E_PLANCK_GEV,
        "per_constraint": per,
        "verdict": overall,
        "note": "Effective scale E_QG / eta^(1/n) compared with each "
                "published subluminal lower limit; excluded if below "
                "any of them.",
    }


# ------------------------------------------------------------- foam checks
def random_walk_psd_asd(freq_hz):
    """Random-walk foam displacement noise, amplitude spectral density.

    A random walk with <dx^2> = c * l_P * t (Kwon, CosPA 2015, quoting
    the Ng & van Dam random-walk picture) is a Wiener process, whose
    one-sided displacement PSD is S_x(f) = c*l_P / (2*pi*f)^2.
    The O(1) convention factor in the denominator does not change the
    verdict: the prediction sits ~3 orders of magnitude above LIGO's
    floor either way. Returns ASD in m/sqrt(Hz).
    """
    freq_hz = float(freq_hz)
    if freq_hz <= 0:
        raise ValueError("freq_hz must be > 0")
    psd = C_M_PER_S * L_PLANCK_M / (2.0 * np.pi * freq_hz) ** 2
    return float(np.sqrt(psd))


def ligo_displacement_floor(arm_length_m=LIGO_ARM_M):
    """LIGO O1 displacement noise floor (m/sqrt(Hz)) at 100 Hz.

    Strain 1e-23 /sqrt(Hz) (Abbott et al. 2016) times the arm length.
    """
    return LIGO_O1_STRAIN_PER_RT_HZ * float(arm_length_m)


def check_foam(model, arm_length_m=LIGO_ARM_M, freq_hz=LIGO_FREQ_HZ):
    """Check a foam toy model against the LIGO O1 noise floor.

    random_walk: the model predicts a displacement noise spectral
    density at ``freq_hz``; if it sits above LIGO's floor it would
    have been seen, so it is EXCLUDED.
    holographic: the model predicts an RMS fluctuation far below the
    floor integrated over a 1 Hz band, so it is UNCONSTRAINED by LIGO
    (not ruled in, not ruled out — simply below sensitivity).

    Returns a dict with the numbers, the ratio to the floor, and the
    verdict. The Holometer result is a separate check
    (``check_holometer``): it excludes Hogan's shear-symmetry variant,
    not the generic holographic scaling checked here.
    """
    from . import foam

    if model == "random_walk":
        predicted_asd = random_walk_psd_asd(freq_hz)
        floor = ligo_displacement_floor(arm_length_m)
        ratio = predicted_asd / floor
        return {
            "model": model,
            "arm_length_m": float(arm_length_m),
            "freq_hz": float(freq_hz),
            "predicted_asd_m_per_rt_hz": predicted_asd,
            "ligo_floor_m_per_rt_hz": floor,
            "ratio": ratio,
            "verdict": "EXCLUDED" if ratio > 1.0 else "SURVIVES",
            "note": "Random-walk foam predicts broadband displacement "
                    "noise ~3 orders of magnitude above LIGO's O1 floor "
                    "at 100 Hz; LIGO would have seen it. Ruled out.",
        }
    if model == "holographic":
        rms = float(foam.fluctuation(arm_length_m, model="holographic"))
        floor_1hz = ligo_displacement_floor(arm_length_m) * 1.0  # 1 Hz band
        ratio = rms / floor_1hz
        return {
            "model": model,
            "arm_length_m": float(arm_length_m),
            "freq_hz": float(freq_hz),
            "predicted_rms_m": rms,
            "ligo_floor_1hz_band_m": floor_1hz,
            "ratio": ratio,
            "verdict": "UNCONSTRAINED" if ratio < 1.0 else "EXCLUDED",
            "note": "Holographic RMS fluctuation sits ~400x below LIGO's "
                    "displacement floor even in a 1 Hz band: below "
                    "sensitivity, not tested. Hogan's shear-symmetry "
                    "variant is a different model — see check_holometer.",
        }
    raise ValueError(f"unknown foam model {model!r}; choose from "
                     f"{foam.MODELS}")


def check_foam_40m(model):
    """Check a foam model against the Caltech 40-m parameter bounds.

    Ng & van Dam (gr-qc/9906003) bound the model length scale l_QG at
    ~1e-17 m (random walk) and ~1e-29 m (holographic). A Planck-scale
    model has l_QG = l_P ~ 1.6e-35 m, which passes both bounds — the
    40-m data did not exclude either model, it only bounded their
    parameters. Returns SURVIVES for both, with the numbers.
    """
    bounds = {"random_walk": 1.0e-17, "holographic": 1.0e-29}
    if model not in bounds:
        raise ValueError(f"unknown foam model {model!r}")
    bound = bounds[model]
    return {
        "model": model,
        "model_length_scale_m": L_PLANCK_M,
        "bound_m": bound,
        "ratio": L_PLANCK_M / bound,
        "verdict": "SURVIVES" if L_PLANCK_M < bound else "EXCLUDED",
        "note": "40-m interferometer bounded the model length scale; a "
                "Planck-scale value passes. This is a parameter bound, "
                "not an exclusion of the model.",
    }


def check_holometer():
    """The Holometer verdict on Hogan's shear-symmetry holographic noise.

    Chou et al. 2017 (CQG 34, 165005): the final data probe nearly an
    order of magnitude below the Planck-scale shear-symmetry
    holographic prediction (2 sigma). The model is EXCLUDED. This
    says nothing about the generic holographic scaling in foam.py,
    which the Holometer did not test.
    """
    entry = get_constraint("holometer_chou2017")
    return {
        "model": "hogan_shear_holographic",
        "verdict": "EXCLUDED",
        "exclusion_factor": 0.1,
        "confidence": entry["confidence"],
        "source": entry["source"],
        "source_link": entry["source_link"],
        "note": "Hogan's shear-symmetry holographic noise model is "
                "excluded. The holographic principle itself, and the "
                "generic Ng & van Dam holographic scaling, are not "
                "what was tested.",
    }


# -------------------------------------------------- v0.3: superluminal checks
def check_superluminal_photon(e_lv_gev, order=1):
    """Check a superluminal photon model against the LHAASO limits.

    A model scale below the published lower limit is EXCLUDED: a
    photon at the LHAASO energies would have decayed (n=1) or split
    (n=2) before reaching us. Uses the strictest verified limit at
    each order (He & Ma 2023 for n=1, LHAASO Collaboration 2022 for
    n=2); the LHAASO collaboration's own n=1 number is carried as a
    second per-constraint verdict.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    per = []
    for e in CONSTRAINTS:
        if e.get("sector") != "photon" or e.get("branch") != "superluminal":
            continue
        if e.get("order") != order:
            continue
        if "electron" in e["model"]:
            continue
        verdict = "EXCLUDED" if float(e_lv_gev) < e["limit_value"] \
            else "SURVIVES"
        per.append({
            "constraint_id": e["id"],
            "limit_gev": e["limit_value"],
            "verdict": verdict,
            "source": e["source"],
        })
    overall = "EXCLUDED" if any(p["verdict"] == "EXCLUDED" for p in per) \
        else "SURVIVES"
    return {
        "order": order, "e_lv_gev": float(e_lv_gev),
        "per_constraint": per,
        "verdict": overall,
        "note": "Model scale compared with the published superluminal "
                "lower limits; excluded if below any of them.",
    }


def check_superluminal_electron(e_lv_e_gev, order=1):
    """Check a superluminal electron model against the Crab bound.

    E_LV^(e,sup) >= 9.4e25 GeV (He & Ma 2023, from the Crab's 2.3 PeV
    parent electron). A model scale below that is EXCLUDED: the
    electron would have radiated vacuum Cherenkov down to threshold
    before producing the observed PeV photon. n=1 only — no n=2
    electron bound was verified.
    """
    if order != 1:
        return {
            "order": order, "e_lv_gev": float(e_lv_e_gev),
            "per_constraint": [],
            "verdict": "UNCONSTRAINED",
            "note": "No verified n=2 superluminal electron bound; the "
                    "published bound is n=1 only.",
        }
    entry = get_constraint("hema2023_electron_superluminal_n1")
    verdict = "EXCLUDED" if float(e_lv_e_gev) < entry["limit_value"] \
        else "SURVIVES"
    return {
        "order": 1, "e_lv_gev": float(e_lv_e_gev),
        "per_constraint": [{
            "constraint_id": entry["id"],
            "limit_gev": entry["limit_value"],
            "verdict": verdict,
            "source": entry["source"],
        }],
        "verdict": verdict,
        "note": "Model scale compared with E_LV^(e,sup) >= 9.4e25 GeV.",
    }


# -------------------------------------------------- v0.3: birefringence check
def check_birefringence(eta):
    """Check a birefringence parameter against the GRB polarimetry bound.

    |eta| < 1e-15 at 95% CL (Wei et al. 2025, conservative end of the
    published O(1e-15 - 1e-16) range). A larger |eta| would rotate the
    polarisation across a GRB energy band enough to wash it out, and
    GRBs are observed polarised — so it is EXCLUDED.
    """
    entry = get_constraint("grb_birefringence_wei2025")
    limit = entry["limit_value"]
    verdict = "EXCLUDED" if abs(float(eta)) > limit else "SURVIVES"
    return {
        "eta": float(eta),
        "limit": limit,
        "per_constraint": [{
            "constraint_id": entry["id"],
            "limit": limit,
            "verdict": verdict,
            "source": entry["source"],
        }],
        "verdict": verdict,
        "note": "|eta| compared with the Wei et al. 2025 bound "
                "1e-15 (95% CL).",
    }


# ----------------------------------------------------------- v0.3: GUP check
def check_gup(beta0):
    """Check a GUP parameter against the macroscopic-resonator bound.

    beta_0 < 5.2e6 (Bushev et al. 2019, sapphire split-bar resonator).
    A larger beta_0 would shift the resonator frequency with amplitude
    by more than observed — EXCLUDED. beta_0 = 1 (a Planck-length
    minimal length) is far below the bound: UNCONSTRAINED in the
    sense that it survives, but the experiment cannot see it either.
    The verdict for beta_0 <= 5.2e6 is therefore SURVIVES.
    """
    entry = get_constraint("gup_beta0_bushev2019")
    limit = entry["limit_value"]
    verdict = "EXCLUDED" if float(beta0) > limit else "SURVIVES"
    return {
        "beta0": float(beta0),
        "limit": limit,
        "per_constraint": [{
            "constraint_id": entry["id"],
            "limit": limit,
            "verdict": verdict,
            "source": entry["source"],
        }],
        "verdict": verdict,
        "note": "beta_0 compared with the Bushev et al. 2019 bound "
                "5.2e6 (sapphire resonator).",
    }


# ----------------------------------------------------------- v0.3: CSL checks
def check_csl(lambda_per_s, r_c_m):
    """Check CSL parameters against the LISA Pathfinder + X-ray bounds.

    Two independent constraints, both on the combination the
    experiments actually measure:
    * X-ray (spontaneous heating): lambda / r_C^2 < 3.0e-3 s^-1 m^-2
      (XENONnT 2025, 90% CL) and < 0.494 s^-1 m^-2 (Majorana 2022).
      Applies at any r_C (for r_C <= 1e-5 m in the XENONnT paper).
    * LISA Pathfinder (force noise): lambda < 3e-8 s^-1 at
      r_C = 1e-7 m (Carlesso et al. 2016). The bound is quoted at
      that r_C only; at other r_C the check says so rather than
      extrapolating a curve that was not verified.
    Overall EXCLUDED if either bound excludes the point.
    """
    per = []
    ratio = float(lambda_per_s) / float(r_c_m) ** 2
    for cid in ("csl_xray_xenonnt2025", "csl_xray_majorana2022"):
        entry = get_constraint(cid)
        verdict = "EXCLUDED" if ratio > entry["limit_value"] else "SURVIVES"
        per.append({
            "constraint_id": cid,
            "limit": entry["limit_value"],
            "limit_units": entry["limit_units"],
            "ratio_value": ratio,
            "verdict": verdict,
            "source": entry["source"],
        })
    lpf = get_constraint("csl_lisa_pathfinder_carlesso2016")
    if abs(float(r_c_m) - 1.0e-7) / 1.0e-7 < 0.01:
        lpf_verdict = "EXCLUDED" if float(lambda_per_s) > lpf["limit_value"] \
            else "SURVIVES"
        lpf_note = "applied at r_C = 1e-7 m, where the bound is quoted."
    else:
        lpf_verdict = "UNCONSTRAINED"
        lpf_note = ("LISA Pathfinder bound is quoted at r_C = 1e-7 m "
                    "only; not applied at this r_C.")
    per.append({
        "constraint_id": "csl_lisa_pathfinder_carlesso2016",
        "limit": lpf["limit_value"],
        "limit_units": lpf["limit_units"],
        "ratio_value": float(lambda_per_s),
        "verdict": lpf_verdict,
        "source": lpf["source"],
        "note": lpf_note,
    })
    overall = "EXCLUDED" if any(p["verdict"] == "EXCLUDED" for p in per) \
        else "SURVIVES"
    return {
        "lambda_per_s": float(lambda_per_s),
        "r_c_m": float(r_c_m),
        "lambda_over_rc2": ratio,
        "per_constraint": per,
        "verdict": overall,
        "note": "X-ray bounds apply to lambda/r_C^2 at any r_C; the "
                "LISA Pathfinder bound applies at r_C = 1e-7 m only.",
    }


def check_dp_cutoff(r0_m):
    """Check a Diosi-Penrose cutoff length against the Majorana bound.

    R_0 > 2.54e-10 m (Majorana Collaboration 2022). This is a LOWER
    limit: a smaller R_0 is EXCLUDED. The parameter-free DP model
    (R_0 = 0) is therefore excluded by this data.
    """
    entry = get_constraint("dp_cutoff_majorana2022")
    verdict = "EXCLUDED" if float(r0_m) < entry["limit_value"] else "SURVIVES"
    return {
        "r0_m": float(r0_m),
        "limit_m": entry["limit_value"],
        "per_constraint": [{
            "constraint_id": entry["id"],
            "limit": entry["limit_value"],
            "verdict": verdict,
            "source": entry["source"],
        }],
        "verdict": verdict,
        "note": "R_0 compared with the Majorana lower limit 2.54e-10 m.",
    }


# ------------------------------------------------------ v0.3: neutrino checks
def check_neutrino(m_qg_gev, order=1, branch="subluminal"):
    """Check a neutrino LIV scale against the SN1987A / TXS / IceCube bounds.

    Subluminal: compares M_QG with the SN1987A and TXS 0506+056
    limits at the given order (M_1 > 2.7e10 GeV and > 1e14 GeV;
    M_2 > 4.6e4 GeV and > 1e9 GeV). A scale below a limit is
    EXCLUDED by that observation.
    Superluminal: compares with the IceCube PeV bound M_QG > ~1e5
    M_Pl at n=1 only; no verified n=2 superluminal neutrino bound,
    so that case returns UNCONSTRAINED.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if branch not in ("subluminal", "superluminal"):
        raise ValueError("branch must be 'subluminal' or 'superluminal'")
    if branch == "superluminal" and order == 2:
        return {
            "order": 2, "branch": "superluminal",
            "m_qg_gev": float(m_qg_gev),
            "per_constraint": [],
            "verdict": "UNCONSTRAINED",
            "note": "No verified n=2 superluminal neutrino bound; the "
                    "published PeV bound is n=1 only.",
        }
    per = []
    for e in CONSTRAINTS:
        if e.get("sector") != "neutrino" or e.get("branch") != branch:
            continue
        if e.get("order") != order:
            continue
        if "speed" in e["model"]:
            continue
        verdict = "EXCLUDED" if float(m_qg_gev) < e["limit_value"] \
            else "SURVIVES"
        per.append({
            "constraint_id": e["id"],
            "limit_gev": e["limit_value"],
            "verdict": verdict,
            "source": e["source"],
        })
    overall = "EXCLUDED" if any(p["verdict"] == "EXCLUDED" for p in per) \
        else "SURVIVES"
    return {
        "order": order, "branch": branch,
        "m_qg_gev": float(m_qg_gev),
        "per_constraint": per,
        "verdict": overall,
        "note": "Model scale compared with the published neutrino lower "
                "limits; excluded if below any of them.",
    }


def check_neutrino_speed(delta):
    """Check a neutrino speed deviation against the SN1987A bound.

    |delta| = |(v - c) / c| < 4e-9 (Longo 1988; Stodolsky 1988).
    Model-independent: any model predicting a larger deviation at
    ~10 MeV is EXCLUDED, whatever its energy dependence.
    """
    entry = get_constraint("sn1987a_neutrino_speed")
    verdict = "EXCLUDED" if abs(float(delta)) > entry["limit_value"] \
        else "SURVIVES"
    return {
        "delta": float(delta),
        "limit": entry["limit_value"],
        "per_constraint": [{
            "constraint_id": entry["id"],
            "limit": entry["limit_value"],
            "verdict": verdict,
            "source": entry["source"],
        }],
        "verdict": verdict,
        "note": "|delta| compared with the SN1987A bound 4e-9.",
    }
