"""Collapse toy models: GRW stochastic hits and Diosi-Penrose collapse time.

What is simulated here — and what is NOT
-----------------------------------------
Simulated (ordinary quantum mechanics, fully algorithmic):
* Unitary evolution of a 1-D wavefunction under the free-particle
  Schrodinger equation, by split-step Fourier. Norm is conserved.
* GRW-style stochastic "hits": a Poisson process that, when it fires,
  multiplies the wavefunction by a Gaussian localisation operator of
  width r_c centred at a position sampled from |psi|^2, then
  renormalises. This is a *phenomenological rule* for how a collapse
  would look if it happened — it is not a mechanism for collapse.

NOT simulated — the oracle boundary (read this carefully):
* ``collapse_oracle()`` marks the point where an algorithmic simulation
  *stops*, following Faizal et al. (arXiv:2507.22950): if wavefunction
  collapse requires actualising a Godelian truth, no algorithm can
  produce it, so no simulation can contain the collapse step itself.
  The function therefore RAISES ``CollapseNotSimulable`` instead of
  returning a collapsed state. The GRW hit above is a rule for applying
  a collapse *after the fact*; it does not explain or generate one.
  (Redden, arXiv:2512.11807, counters that undecidability constrains
  provability, not execution — the oracle is a labelled boundary for
  exploration, not a settled claim.)

Diosi-Penrose collapse time
---------------------------
tau ~ hbar / Delta_E_grav, where Delta_E_grav is the gravitational
self-energy difference between the two branches of a spatial
superposition. For a uniform sphere of mass m and radius R displaced
by d, Delta_E = G m^2 (1/R - 1/sqrt(R^2 + d^2)); with no displacement
given it reduces to the order-of-magnitude estimate G m^2 / R.
This is an order-of-magnitude estimate, not a measurement.

Continuous Spontaneous Localisation (CSL, v0.3)
-----------------------------------------------
CSL (Ghirardi, Pearle & Rimini 1990; Bassi et al.) is the continuous
version of GRW: a stochastic field couples to the mass density and
localises macroscopic superpositions at a rate that grows with the
number of nucleons. For a rigid body the collapse rate is

    Lambda = lambda * N^2              (body smaller than r_C)
    Lambda = lambda * N^2 / N_cells    (body larger than r_C)

where N is the number of nucleons and N_cells = (size / r_C)^3 is the
number of correlation volumes in the body — nucleons within one r_C
localise coherently. CSL also predicts spontaneous heating,

    dE/dt = (3 lambda hbar^2 / (4 m_0 r_C^2)) * N ,

which is what the X-ray bounds (Majorana, XENONnT) constrain, and
a force noise on test masses, which is what LISA Pathfinder
constrains. CSL is phenomenology of the aftermath of collapse, like
GRW — it does not simulate the collapse step itself (see the oracle
above).

Assumptions and limits (plain language)
----------------------------------------
* Toy 1-D system, SI units (metres, kilograms, seconds, joules).
  The GRW rate lambda and localisation length r_c are the standard
  textbook values (lambda ~ 1e-16 /s per particle, r_c ~ 1e-7 m) but
  here they act on a single toy particle, so hit times are illustrative.
* The split-step solver is first-order in time and assumes a free
  particle (no potential). It conserves norm to numerical precision.
* Nothing here decides whether collapse is real, or whether it is
  non-computable. The oracle exists so the boundary is explicit
  instead of being silently papered over.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
HBAR_J_S = 1.054571817e-34   # reduced Planck constant, J s
G_M3_KG_S2 = 6.67430e-11     # gravitational constant, m^3 kg^-1 s^-2
GRW_LAMBDA_PER_S = 1.0e-16   # GRW hit rate per particle, 1/s (textbook value)
GRW_R_C_M = 1.0e-7           # GRW localisation length, m
CSL_LAMBDA_PER_S = 1.0e-16   # CSL rate, 1/s (standard GRW value)
CSL_R_C_M = 1.0e-7           # CSL correlation length, m
NUCLEON_MASS_KG = 1.67492749804e-27  # neutron mass, kg (CSL reference mass)


class CollapseNotSimulable(NotImplementedError):
    """Raised by ``collapse_oracle``.

    Marks the non-computable boundary: the collapse step itself is not
    simulated. See the module docstring and Faizal et al.,
    arXiv:2507.22950.
    """


# ------------------------------------------------------- the oracle boundary
def collapse_oracle(psi=None, x=None, rng=None):
    """The collapse step itself — NOT simulated.

    Per Faizal et al. (arXiv:2507.22950), if collapse requires
    actualising a Godelian truth, no algorithm can produce it, so an
    algorithmic simulation must stop here. This function therefore
    always raises :class:`CollapseNotSimulable`.

    It exists so that any code path that would need a genuine collapse
    fails loudly at a labelled boundary, instead of quietly substituting
    a random-number draw and calling it collapse. If you want the
    *phenomenological aftermath* of a collapse (a localised
    wavefunction), use :func:`grw_hit` — and remember it is a rule
    applied after the fact, not a mechanism.
    """
    raise CollapseNotSimulable(
        "collapse_oracle: the collapse step itself is NOT simulated. "
        "Per Faizal et al. (arXiv:2507.22950), actualising a Godelian "
        "truth cannot be done algorithmically, so this marks the boundary "
        "where an algorithmic simulation stops. Use grw_hit() only for "
        "the phenomenological aftermath of a collapse, never as a "
        "mechanism for one."
    )


# ---------------------------------------------------------- unitary evolution
def gaussian_wavepacket(x, x0, sigma, k0=0.0):
    """Normalised Gaussian wavepacket centred at x0, width sigma,
    mean wavenumber k0 (1/m). Returns complex psi on grid x (m)."""
    psi = np.exp(-0.5 * ((x - x0) / sigma) ** 2) * np.exp(1j * k0 * (x - x0))
    dx = float(x[1] - x[0])
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)


def evolve_unitary(psi, x, dt, mass_kg, n_steps=1):
    """Free-particle Schrodinger evolution by split-step Fourier.

    psi -> exp(-i hbar k^2 dt / 2m) psi in Fourier space, repeated
    ``n_steps`` times with step ``dt`` (s). The operation is unitary:
    the norm of psi is conserved to numerical precision. ``mass_kg``
    is the particle mass in kg; ``x`` the grid in metres.
    """
    psi = np.asarray(psi, dtype=complex).copy()
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    phase = np.exp(-1j * HBAR_J_S * k**2 * dt / (2.0 * mass_kg))
    for _ in range(int(n_steps)):
        psi = np.fft.ifft(np.fft.fft(psi) * phase)
    return psi


def norm(psi, x):
    """L2 norm of psi on grid x (should be 1 for a normalised state).

    Uses the discrete sum sum(|psi|^2) * dx — the norm the Fourier
    grid actually conserves under split-step evolution.
    """
    return float(np.sqrt(np.sum(np.abs(psi) ** 2) * (x[1] - x[0])))


# ---------------------------------------------------------------- GRW hits
def grw_hit(psi, x, r_c_m=GRW_R_C_M, rng=None):
    """Apply one GRW localisation hit (phenomenological aftermath only).

    A hit centre is sampled from the Born distribution |psi|^2, the
    wavefunction is multiplied by a Gaussian of width ``r_c_m`` centred
    there, and the result is renormalised. Returns (psi_new, hit_centre_m).

    This is a rule for what a collapsed wavefunction would look like
    after the fact. It does not generate or explain a collapse — see
    :func:`collapse_oracle`.
    """
    rng = np.random.default_rng(rng)
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    centre = float(rng.choice(x, p=probs))
    localiser = np.exp(-0.5 * ((x - centre) / r_c_m) ** 2)
    psi_new = psi * localiser
    psi_new = psi_new / np.sqrt(np.sum(np.abs(psi_new) ** 2) * (x[1] - x[0]))
    return psi_new, centre


def grw_trajectory(psi0, x, dt, mass_kg, lam_per_s=GRW_LAMBDA_PER_S,
                   r_c_m=GRW_R_C_M, n_steps=100, seed=0):
    """Alternate unitary evolution with Poisson GRW hits.

    At each step of length ``dt`` a hit fires with probability
    lam * dt. Returns a dict with the final psi, the list of hit times
    (s) and hit centres (m), and the norm history. The hits are the
    phenomenological rule from :func:`grw_hit`; the actual collapse
    step is still not simulated (see :func:`collapse_oracle`).
    """
    rng = np.random.default_rng(seed)
    psi = np.asarray(psi0, dtype=complex).copy()
    hit_times, hit_centres, norms = [], [], []
    p_hit = float(lam_per_s) * float(dt)
    if p_hit > 0.1:
        raise ValueError(
            f"hit probability per step {p_hit:.3g} is too large; "
            "reduce dt or lam so lam*dt << 1")
    for step in range(int(n_steps)):
        psi = evolve_unitary(psi, x, dt, mass_kg, n_steps=1)
        if rng.random() < p_hit:
            psi, centre = grw_hit(psi, x, r_c_m=r_c_m, rng=rng)
            hit_times.append(step * dt)
            hit_centres.append(centre)
        norms.append(norm(psi, x))
    return {
        "psi": psi,
        "hit_times_s": np.array(hit_times),
        "hit_centres_m": np.array(hit_centres),
        "norm_history": np.array(norms),
        "n_hits": len(hit_times),
    }


# ------------------------------------------------------- Diosi-Penrose time
def dp_collapse_time(mass_kg, radius_m, displacement_m=None):
    """Diosi-Penrose collapse time tau ~ hbar / Delta_E_grav (seconds).

    For a uniform sphere of mass ``mass_kg`` (kg) and radius
    ``radius_m`` (m) in a superposition of two positions displaced by
    ``displacement_m`` (m), Delta_E = G m^2 (1/R - 1/sqrt(R^2 + d^2)).
    With no displacement given, uses the order-of-magnitude estimate
    Delta_E = G m^2 / R. Returns tau in seconds; smaller tau means
    faster collapse. This is an order-of-magnitude estimate.
    """
    m = float(mass_kg)
    r = float(radius_m)
    if m <= 0 or r <= 0:
        raise ValueError("mass_kg and radius_m must be positive")
    if displacement_m is None:
        delta_e = G_M3_KG_S2 * m**2 / r
    else:
        d = float(displacement_m)
        if d < 0:
            raise ValueError("displacement_m must be >= 0")
        delta_e = G_M3_KG_S2 * m**2 * (1.0 / r - 1.0 / np.sqrt(r**2 + d**2))
        if delta_e <= 0:
            return float("inf")
    return float(HBAR_J_S / delta_e)


# ---------------------------------------------------------------------- CSL
def csl_rate(mass_kg, lambda_per_s=CSL_LAMBDA_PER_S, r_c_m=CSL_R_C_M,
             size_m=None):
    """CSL collapse rate (1/s) for a rigid body of ``mass_kg``.

    N = mass / m_nucleon nucleons. If the body is smaller than r_C
    (or ``size_m`` is not given), all nucleons localise coherently
    and Lambda = lambda * N^2. For a larger body only the nucleons
    within one correlation volume act coherently, so Lambda =
    lambda * N^2 / N_cells with N_cells = (size / r_C)^3. This is the
    textbook rigid-body scaling (Bassi et al.); it is the rate at
    which a spatial superposition of the body would collapse *after*
    the collapse step — it does not simulate the step itself.
    """
    m = float(mass_kg)
    if m <= 0:
        raise ValueError("mass_kg must be > 0")
    if lambda_per_s < 0 or r_c_m <= 0:
        raise ValueError("lambda_per_s must be >= 0 and r_c_m > 0")
    n_nucleons = m / NUCLEON_MASS_KG
    if size_m is None or float(size_m) <= float(r_c_m):
        return float(lambda_per_s) * n_nucleons**2
    n_cells = (float(size_m) / float(r_c_m)) ** 3
    return float(lambda_per_s) * n_nucleons**2 / n_cells


def csl_heating_power(mass_kg, lambda_per_s=CSL_LAMBDA_PER_S,
                      r_c_m=CSL_R_C_M):
    """CSL spontaneous heating power (W) for a mass ``mass_kg``.

    dE/dt = (3 lambda hbar^2 / (4 m_0 r_C^2)) * N, with N the number
    of nucleons. This is the heating the X-ray bounds constrain:
    spontaneous radiation from the stochastic kicks would show up as
    excess X-rays, and its absence bounds lambda / r_C^2. See
    constraints.check_csl_xray.
    """
    m = float(mass_kg)
    if m <= 0:
        raise ValueError("mass_kg must be > 0")
    n_nucleons = m / NUCLEON_MASS_KG
    return float(3.0 * lambda_per_s * HBAR_J_S**2
                 / (4.0 * NUCLEON_MASS_KG * float(r_c_m)**2)) * n_nucleons
