"""Real published observational constraints, and checks of the toy
models against them (v0.2).

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
by the Fermilab Holometer. Those are the facts the ``compare`` command
plots.

How the checks work
-------------------
* Dispersion: a linear toy model with coefficient ``eta`` and scale
  E_QG behaves, for time-of-flight purposes, like a model with an
  effective scale E_QG / eta (eta > 0, subluminal). A published lower
  limit L excludes the model if E_QG / eta < L. The benchmark
  eta = 1 at the Planck scale has effective scale 1 E_Pl, which is
  below every published linear limit — it is excluded.
* Quadratic (n = 2) limits are recorded but the toy model has no
  quadratic term, so they do not constrain it; the check says so
  (UNCONSTRAINED) rather than pretending otherwise.
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
    """Check a linear (or quadratic) toy model against the GRB limits.

    The toy model in ``dispersion.py`` has coefficient ``eta`` and scale
    E_QG; for time-of-flight it behaves like a model with effective
    scale E_QG / eta (eta > 0, subluminal). A published lower limit L
    excludes the model when E_QG / eta < L.

    For order=2 the toy model has no quadratic term, so the check
    returns UNCONSTRAINED with that stated plainly.

    Returns a dict with the effective scale, one verdict per matching
    constraint, and an overall verdict (EXCLUDED if any constraint
    excludes the model, else SURVIVES; UNCONSTRAINED for order 2).
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
                    "here are subluminal ones and do not cover it.",
        }
    if order == 2:
        per = [{
            "constraint_id": e["id"],
            "limit_gev": e["limit_value"],
            "verdict": "UNCONSTRAINED",
            "source": e["source"],
            "note": "toy model has no quadratic term; this limit does "
                    "not test it.",
        } for e in CONSTRAINTS if e.get("order") == 2]
        return {
            "order": 2, "eta": float(eta),
            "effective_scale_gev": None, "effective_scale_e_pl": None,
            "per_constraint": per,
            "verdict": "UNCONSTRAINED",
            "note": "The linear toy model has no quadratic term, so the "
                    "quadratic GRB limits do not constrain it.",
        }
    effective = float(e_qg_gev) / float(eta) if eta != 0 else float("inf")
    per = []
    for e in CONSTRAINTS:
        if e.get("order") != 1:
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
        "order": 1, "eta": float(eta),
        "effective_scale_gev": effective,
        "effective_scale_e_pl": effective / E_PLANCK_GEV,
        "per_constraint": per,
        "verdict": overall,
        "note": "Effective scale E_QG / eta compared with each published "
                "lower limit; excluded if below any of them.",
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
