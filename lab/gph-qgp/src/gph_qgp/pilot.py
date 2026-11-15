"""The GPH sober blinded-pilot analysis pipeline.

This reimplements, as a clean package, the pilot scripts from the GPH
v3.2 sober pilot protocol (Part A): synthetic speckle generation
(``speckle_synth``), blinded-observer agreement analysis
(``agreement_analysis``), and camera invariance checks
(``invariance_analysis``). The original scripts were written for the
paper; this module keeps their methods and their warnings.

Protocol in brief (Part A, v3.2 section 06)
-------------------------------------------
Sober observers look at laser speckle on a wall at 30 marked locations
and report what, if anything, they see, coded by raters blind to
condition into five categories::

    none / texture only / geometric structure / character-like forms / other

Three conditions are compared: W (live wall), R (recording), S
(synthetic speckle). The decision rules ask whether agreement between
independent observers is higher in W than in S, and whether a
wall-locked pattern survives changes of viewpoint, aperture and
wavelength (the invariance test, P1).

What this module does
---------------------
* ``synth_speckle`` — fully developed speckle with controllable grain
  size and contrast, plus an optional fixed "stored" component of
  controllable strength (the wall-locked hypothesis, as a toy).
* Agreement statistics — Cohen's kappa (two raters), Fleiss' kappa
  (many raters), Krippendorff's alpha (nominal, the pilot's main
  statistic), mean pairwise agreement, bootstrap CIs, and a
  permutation test for W vs S.
* Invariance — normalised cross-correlation between images of the
  same location under different settings, against a null built from
  deliberately misaligned regions.
* ``simulate_observers`` / ``run_demo`` — end-to-end simulated runs
  of the pipeline on synthetic data, for testing the code and for
  sensitivity planning.

Assumptions and limits (plain language)
----------------------------------------
* All data here is SIMULATED. These functions test the analysis
  pipeline and show what its outputs look like under a null
  (conventional speckle optics) and under a toy "stored structure"
  hypothesis. A high kappa or a high invariance z-score in a
  simulation is a demonstration that the pipeline can detect the
  injected pattern — it is NOT evidence that the pattern exists in
  the real world, and it is NOT evidence for GPH.
* Responses must be coded into a small set of categories by raters
  blind to condition before analysis. Raw free-text descriptions
  will almost never match exactly and would make agreement look like
  zero everywhere.
* The invariance test cannot by itself distinguish wall-stored
  structure from ordinary reflectance or illumination structure on
  the wall; that is why the real protocol requires an incoherent-light
  reference and a control wall.
* Speckle images are in arbitrary intensity units, normalised to
  mean 1. Grain size (FWHM) is in pixels.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
from collections import defaultdict

import numpy as np

# Pilot coding scheme (Part A): the five coded response categories.
CATEGORIES = ["none", "texture only", "geometric structure",
              "character-like forms", "other"]
NONE_LABELS = {"", "none", "nothing", "no", "null", "n/a", "na"}
CONDITIONS = ("W", "R", "S")


# ------------------------------------------------------------ speckle synth
def aperture_radius_for_fwhm(n, fwhm_px):
    """Aperture radius (frequency pixels) giving the requested speckle FWHM.

    For a circular aperture, the intensity autocorrelation falls to half
    at a lag of about 0.2563 n / R, so FWHM ~ 0.5125 n / R.
    """
    return 0.5125 * n / float(fwhm_px)


def synth_speckle(n, fwhm_px, rng, n_independent=1, structure_strength=0.0,
                  structure=None):
    """Return an n x n synthetic speckle intensity image, mean 1.

    A random-phase field is band-limited by a circular aperture in the
    Fourier domain, inverse transformed, and squared — fully developed
    speckle (exponential intensity statistics, contrast ~ 1). Contrast
    is lowered by averaging ``n_independent`` uncorrelated patterns
    (contrast ~ 1/sqrt(n_independent)).

    ``structure_strength`` (0..1) blends in a fixed, wall-locked
    pattern: 0 is pure speckle (the conventional account); 1 is the
    fixed pattern alone. Pass ``structure`` to reuse a specific fixed
    pattern across images (e.g. the same wall); if None and
    ``structure_strength`` > 0, a fixed pattern is generated from the
    RNG and returned alongside the image is not possible — instead use
    ``fixed_speckle`` to make one first. Raises ValueError for an
    unreachable grain size on the grid.
    """
    radius = aperture_radius_for_fwhm(n, fwhm_px)
    if radius < 1.0 or radius > n / 2.0:
        raise ValueError(
            f"fwhm_px={fwhm_px:g} is not reachable on an n={n} grid "
            f"(aperture radius {radius:.2f} must be between 1 and n/2).")
    f = np.fft.fftfreq(n) * n
    fy, fx = np.meshgrid(f, f, indexing="ij")
    aperture = (fx ** 2 + fy ** 2) <= radius ** 2
    total = np.zeros((n, n))
    for _ in range(int(n_independent)):
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, n))
        field = np.fft.ifft2(aperture * np.exp(1j * phase))
        total += np.abs(field) ** 2
    total /= n_independent
    img = total / total.mean()
    if structure_strength > 0:
        if structure is None:
            raise ValueError(
                "structure_strength > 0 needs a fixed `structure` image; "
                "make one with fixed_speckle() so the same wall pattern "
                "is shared across images.")
        if not 0.0 <= structure_strength <= 1.0:
            raise ValueError("structure_strength must be between 0 and 1")
        w = float(structure_strength)
        img = (1.0 - w) * img + w * np.asarray(structure, dtype=float)
        img = img / img.mean()
    return img


def fixed_speckle(n, fwhm_px, rng):
    """A single fixed speckle realisation, to be shared as the
    wall-locked component across images (see ``synth_speckle``)."""
    return synth_speckle(n, fwhm_px, rng)


def autocorr2d(img):
    """Normalised 2-D autocorrelation of an image (peak 1 at zero lag)."""
    x = np.asarray(img, dtype=float)
    x = x - x.mean()
    power = np.abs(np.fft.fft2(x)) ** 2
    ac = np.fft.ifft2(power).real
    ac = ac / ac[0, 0]
    return np.fft.fftshift(ac)


def estimate_fwhm(img):
    """Full width at half maximum (pixels) of the radially averaged
    autocorrelation. Returns nan if the profile never drops below 0.5."""
    ac = autocorr2d(img)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    y, x = np.indices(ac.shape)
    r = np.hypot(y - cy, x - cx)
    bins = r.astype(int)
    counts = np.maximum(np.bincount(bins.ravel()), 1)
    prof = np.bincount(bins.ravel(), weights=ac.ravel()) / counts
    mean_r = np.bincount(bins.ravel(), weights=r.ravel()) / counts
    below = np.where(prof < 0.5)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    frac = (prof[i - 1] - 0.5) / (prof[i - 1] - prof[i])
    r_half = mean_r[i - 1] + frac * (mean_r[i] - mean_r[i - 1])
    return 2.0 * r_half


def contrast(img):
    """Speckle contrast: std / mean of the intensity."""
    img = np.asarray(img, dtype=float)
    return float(img.std() / img.mean())


# ------------------------------------------------------ agreement statistics
def normalise_response(resp):
    """Map a raw response string to a coded category; empty / 'none' /
    'nothing' etc. become 'none'."""
    r = (resp or "").strip().lower()
    return "none" if r in NONE_LABELS else r


def cohen_kappa(rater_a, rater_b, categories=None):
    """Cohen's kappa for two raters' categorical ratings.

    Returns a float in [-1, 1]: 1 is perfect agreement, 0 is chance
    agreement, negative is worse than chance. Returns nan if either
    rater used only one category (kappa undefined).
    """
    a = [normalise_response(r) for r in rater_a]
    b = [normalise_response(r) for r in rater_b]
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("raters must give the same non-zero number of ratings")
    cats = sorted(set(a) | set(b)) if categories is None else list(categories)
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    conf = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    p_o = np.trace(conf) / n
    p_e = float((conf.sum(axis=1) / n) @ (conf.sum(axis=0) / n))
    if p_e >= 1.0:
        return float("nan")
    return float((p_o - p_e) / (1.0 - p_e))


def fleiss_kappa(ratings, categories=None):
    """Fleiss' kappa for many raters.

    ``ratings`` is an (n_units, n_raters) array of category labels
    (strings). Each unit must have the same number of ratings.
    Returns a float in [-1, 1]; nan if only one category is present.
    """
    ratings = np.asarray(ratings, dtype=object)
    if ratings.ndim != 2 or ratings.shape[1] < 2:
        raise ValueError("ratings must be (n_units, n_raters) with >= 2 raters")
    n_units, n_raters = ratings.shape
    cats = sorted({r for row in ratings for r in row}) \
        if categories is None else list(categories)
    idx = {c: i for i, c in enumerate(cats)}
    counts = np.zeros((n_units, len(cats)))
    for i, row in enumerate(ratings):
        for r in row:
            counts[i, idx[r]] += 1
    p_j = counts.sum(axis=0) / (n_units * n_raters)
    p_e = float((p_j ** 2).sum())
    p_i = ((counts * (counts - 1)).sum(axis=1) / (n_raters * (n_raters - 1)))
    p_o = float(p_i.mean())
    if p_e >= 1.0:
        return float("nan")
    return float((p_o - p_e) / (1.0 - p_e))


def _coincidence_matrices(units, categories):
    """Per-unit coincidence matrices (Krippendorff). ``units`` maps
    (condition, location) -> list of responses. Returns dict key ->
    (C x C) array, skipping units with fewer than 2 responses."""
    idx = {c: i for i, c in enumerate(categories)}
    out = {}
    for key, vals in units.items():
        m = len(vals)
        if m < 2:
            continue
        v = np.zeros(len(categories))
        for val in vals:
            v[idx[val]] += 1
        o = np.outer(v, v) - np.diag(v)
        out[key] = o / (m - 1)
    return out


def krippendorff_alpha_nominal(units, categories=None):
    """Krippendorff's alpha (nominal) over all units in ``units``.

    Chance-corrected agreement; 1 is perfect, 0 is chance. Returns nan
    if undefined (e.g. only one category present anywhere, or too few
    responses). ``units`` maps (condition, location) -> responses.
    """
    if categories is None:
        categories = sorted({r for vals in units.values() for r in vals})
    mats = _coincidence_matrices(units, categories)
    if not mats:
        return float("nan")
    o = np.sum(list(mats.values()), axis=0)
    n = o.sum()
    if n <= 1:
        return float("nan")
    n_c = o.sum(axis=1)
    denom = n ** 2 - (n_c ** 2).sum()
    if denom <= 0:
        return float("nan")
    observed_disagree = n - np.trace(o)
    return float(1.0 - (n - 1.0) * observed_disagree / denom)


def pairwise_agreement(responses_per_unit):
    """Mean share of observer pairs giving the same category, averaged
    over units with at least 2 observers. ``responses_per_unit`` is a
    list of lists of responses."""
    vals = []
    for responses in responses_per_unit:
        m = len(responses)
        if m < 2:
            continue
        counts = defaultdict(int)
        for r in responses:
            counts[r] += 1
        same = sum(c * (c - 1) for c in counts.values())
        vals.append(same / (m * (m - 1)))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_alpha_ci(units, categories=None, n_boot=2000, seed=1):
    """Bootstrap 95% CI for Krippendorff's alpha, resampling units."""
    rng = np.random.default_rng(seed)
    if categories is None:
        categories = sorted({r for vals in units.values() for r in vals})
    mats = list(_coincidence_matrices(units, categories).values())
    if len(mats) < 2:
        return (float("nan"), float("nan"))
    stack = np.stack(mats)
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(stack), size=len(stack))
        o = stack[pick].sum(axis=0)
        n = o.sum()
        if n <= 1:
            continue
        n_c = o.sum(axis=1)
        denom = n ** 2 - (n_c ** 2).sum()
        if denom <= 0:
            continue
        vals.append(1.0 - (n - 1.0) * (n - np.trace(o)) / denom)
    vals = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def permutation_test_alpha(units_a, units_b, categories=None, n_perm=2000,
                           seed=1):
    """Permutation test on the difference in alpha between two sets of
    units (e.g. conditions W and S). Shuffles unit labels under the
    null of exchangeability. Returns (observed difference, two-sided p).
    """
    rng = np.random.default_rng(seed)
    if categories is None:
        cats = sorted({r for u in (units_a, units_b)
                       for vals in u.values() for r in vals})
    else:
        cats = list(categories)
    ma = list(_coincidence_matrices(units_a, cats).values())
    mb = list(_coincidence_matrices(units_b, cats).values())
    if len(ma) < 2 or len(mb) < 2:
        return float("nan"), float("nan")

    def _alpha(mats):
        o = np.sum(mats, axis=0)
        n = o.sum()
        if n <= 1:
            return float("nan")
        n_c = o.sum(axis=1)
        denom = n ** 2 - (n_c ** 2).sum()
        if denom <= 0:
            return float("nan")
        return 1.0 - (n - 1.0) * (n - np.trace(o)) / denom

    all_mats = np.stack(ma + mb)
    obs = _alpha(all_mats[:len(ma)]) - _alpha(all_mats[len(ma):])
    if not np.isfinite(obs):
        return float(obs), float("nan")
    count, valid = 0, 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(len(all_mats))
        d = _alpha(all_mats[perm[:len(ma)]]) - _alpha(all_mats[perm[len(ma):]])
        if not np.isfinite(d):
            continue
        valid += 1
        if abs(d) >= abs(obs) - 1e-12:
            count += 1
    p = (count + 1.0) / (valid + 1.0) if valid else float("nan")
    return float(obs), float(p)


# ------------------------------------------------------------- invariance
def pearson(a, b):
    """Pearson correlation between two equal-shaped arrays."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def best_ncc(ia, ib, roi, offset=(0, 0), max_shift=3):
    """Max Pearson r between an ROI of image A and the ROI of image B
    shifted by ``offset`` + (dy, dx), |d| <= ``max_shift`` (pixels).
    ``roi`` is (x, y, w, h). Returns (best_r, (dy, dx))."""
    x, y, w, h = roi
    oy, ox = offset
    ny, nx = ia.shape
    a = ia[y:y + h, x:x + w]
    best, best_s = -2.0, (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            by, bx = y + oy + dy, x + ox + dx
            if by < 0 or bx < 0 or by + h > ny or bx + w > nx:
                continue
            r = pearson(a, ib[by:by + h, bx:bx + w])
            if r > best:
                best, best_s = r, (dy, dx)
    return best, best_s


def invariance_check(images, labels, roi=None, max_shift=3, null_draws=40,
                     min_sep=16, seed=1):
    """Invariance test (P1) over images of one wall location.

    ``images`` is a list of 2-D arrays of the same location taken under
    different settings; ``labels`` names the setting of each image
    (e.g. "v0_650nm"). For every pair, computes the best normalised
    correlation within a small registration tolerance, plus a null
    distribution from deliberately misaligned regions, and a z-score.

    Returns a list of dicts, one per pair, with keys: a, b, pair_class,
    best_r, null_mean, null_sd, z. Conventional speckle optics predicts
    high r only for repeat pairs (same settings); a wall-locked
    component stays correlated across settings. A high z can also come
    from reflectance/illumination structure — use an incoherent-light
    reference and a control wall before drawing any conclusion.
    """
    rng = np.random.default_rng(seed)
    ny, nx = images[0].shape
    roi = tuple(roi) if roi else (nx // 4, ny // 4, nx // 2, ny // 2)
    x, y, w, h = roi
    results = []
    for (i, a), (j, b) in itertools.combinations(enumerate(images), 2):
        r_obs, _ = best_ncc(a, b, roi, (0, 0), max_shift)
        # null: best-r for deliberately misaligned regions
        oy_lo, oy_hi = -y + max_shift, ny - h - y - max_shift
        ox_lo, ox_hi = -x + max_shift, nx - w - x - max_shift
        vals, tries = [], 0
        while len(vals) < null_draws and tries < null_draws * 200:
            tries += 1
            oy = int(rng.integers(oy_lo, oy_hi + 1))
            ox = int(rng.integers(ox_lo, ox_hi + 1))
            if max(abs(oy), abs(ox)) < min_sep:
                continue
            r_null, _ = best_ncc(a, b, roi, (oy, ox), max_shift)
            vals.append(r_null)
        mu = float(np.mean(vals)) if len(vals) >= 5 else float("nan")
        sd = float(np.std(vals, ddof=1)) if len(vals) >= 5 else float("nan")
        z = (r_obs - mu) / sd if np.isfinite(sd) and sd > 0 else float("nan")
        results.append({
            "a": labels[i], "b": labels[j],
            "pair_class": "repeat" if labels[i] == labels[j] else "changed",
            "best_r": round(float(r_obs), 4),
            "null_mean": round(mu, 4), "null_sd": round(sd, 4),
            "z": round(float(z), 2) if np.isfinite(z) else float("nan"),
        })
    return results


# ------------------------------------------------------- simulated observers
def simulate_observers(n_observers=12, n_locations=30, mode="null",
                       p_none=0.3, q=0.5, seed=7):
    """Simulate blinded-observer responses for the pilot.

    mode "null"    : nobody sees shared structure (conventional account).
    mode "stored"  : in condition W each location has a shared 'true'
                     category that observers pick with probability q.
    Conditions W (live wall), R (recording), S (synthetic speckle).
    Returns a list of dict rows with observer_id, location_id,
    condition, response — the same columns the real protocol records.

    This is SIMULATED data for testing the pipeline. It says nothing
    about the real phenomenon.
    """
    rng = np.random.default_rng(seed)
    cats = [c for c in CATEGORIES if c != "none"]
    rows = []
    for cond in CONDITIONS:
        for loc in range(n_locations):
            true_cat = cats[rng.integers(len(cats))]
            for obs in range(n_observers):
                if rng.random() < p_none:
                    resp = "none"
                elif mode == "stored" and cond == "W" and rng.random() < q:
                    resp = true_cat
                else:
                    resp = cats[rng.integers(len(cats))]
                rows.append({"observer_id": f"o{obs:02d}",
                             "location_id": f"L{loc:02d}",
                             "condition": cond, "response": resp})
    return rows


def analyse_observers(rows, cond_a="W", cond_b="S", n_perm=2000, seed=1):
    """Run the full agreement analysis over simulated (or real) rows.

    Returns a dict with per-condition report rate, mean pairwise
    agreement, Krippendorff's alpha and bootstrap 95% CI, plus a
    W-vs-S permutation comparison. ``rows`` are dicts with keys
    observer_id, location_id, condition, response.
    """
    units = defaultdict(list)
    for row in rows:
        units[(row["condition"], row["location_id"])].append(
            normalise_response(row["response"]))
    categories = sorted({r for vals in units.values() for r in vals})
    result = {"rows": len(rows), "categories": categories, "per_condition": {}}
    for cond in sorted({k[0] for k in units}):
        keys = [k for k in units if k[0] == cond]
        cond_units = {k: units[k] for k in keys}
        resp_all = [r for k in keys for r in units[k]]
        result["per_condition"][cond] = {
            "units": len(keys),
            "responses": len(resp_all),
            "report_rate": float(np.mean([r != "none" for r in resp_all])),
            "mean_pairwise_agreement": pairwise_agreement(
                [units[k] for k in keys]),
            "alpha_nominal": krippendorff_alpha_nominal(cond_units, categories),
            "alpha_ci95": list(bootstrap_alpha_ci(cond_units, categories,
                                                   seed=seed)),
        }
    ua = {k: v for k, v in units.items() if k[0] == cond_a}
    ub = {k: v for k, v in units.items() if k[0] == cond_b}
    diff, p = permutation_test_alpha(ua, ub, categories,
                                     n_perm=n_perm, seed=seed)
    result["comparison"] = {"a": cond_a, "b": cond_b,
                            "alpha_diff_a_minus_b": diff,
                            "perm_p_two_sided": p, "n_perm": n_perm}
    return result


# ------------------------------------------------------------------ demo
def run_demo(out_dir, seed=7, n_observers=12, n_locations=30, q=0.5,
             size_px=128, fwhm_px=4.0, structure_strength=0.5):
    """End-to-end demo of the pilot pipeline on simulated data.

    Simulates observers in "stored" mode (a shared pattern in W only),
    analyses agreement, and runs a small invariance check on synthetic
    images (one location, four settings, with and without a fixed
    wall-locked component). Writes ``observers.csv``,
    ``summary.csv`` and ``summary.json`` into ``out_dir`` and returns
    the summary dict.

    The numbers show what the pipeline outputs when a pattern is
    injected — a check of the code, not evidence of anything real.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = simulate_observers(n_observers=n_observers,
                              n_locations=n_locations, mode="stored",
                              q=q, seed=seed)
    with open(os.path.join(out_dir, "observers.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["observer_id", "location_id",
                                           "condition", "response"])
        w.writeheader()
        w.writerows(rows)
    summary = analyse_observers(rows, n_perm=1000, seed=seed)

    # invariance demo: 4 settings of one location, stored vs conventional
    rng = np.random.default_rng(seed)
    fixed = fixed_speckle(size_px, fwhm_px, rng)
    inv = {}
    for mode, wgt in (("stored", structure_strength), ("conventional", 0.0)):
        images, labels = [], []
        for v, wl in (("v0", 650), ("v1", 650), ("v0", 532), ("v1", 532)):
            img = synth_speckle(size_px, fwhm_px,
                                np.random.default_rng([seed, hash((v, wl)) % 2**31]),
                                structure_strength=wgt, structure=fixed)
            images.append(img)
            labels.append(f"{v}_{wl}nm")
        pairs = invariance_check(images, labels, null_draws=20, seed=seed)
        changed = [p for p in pairs if p["pair_class"] == "changed"]
        inv[mode] = {
            "pairs": len(pairs),
            "mean_best_r_changed": float(np.mean([p["best_r"] for p in changed])),
            "mean_z_changed": float(np.nanmean([p["z"] for p in changed])),
        }
    summary["invariance_demo"] = inv

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "units", "responses",
                                           "report_rate",
                                           "mean_pairwise_agreement",
                                           "alpha_nominal",
                                           "alpha_ci95_lo", "alpha_ci95_hi"])
        w.writeheader()
        for cond, d in summary["per_condition"].items():
            w.writerow({"condition": cond, "units": d["units"],
                        "responses": d["responses"],
                        "report_rate": round(d["report_rate"], 4),
                        "mean_pairwise_agreement": round(d["mean_pairwise_agreement"], 4),
                        "alpha_nominal": round(d["alpha_nominal"], 4),
                        "alpha_ci95_lo": round(d["alpha_ci95"][0], 4),
                        "alpha_ci95_hi": round(d["alpha_ci95"][1], 4)})
    return summary
