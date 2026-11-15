"""gph_qgp — the GPH Quantum Gravity Phenomenology lab (v0.3.0).

Toy phenomenological models for teaching and exploration, built around
the Graduation Protocol Hypothesis (GPH) paper, v3.2
(Zenodo DOI 10.5281/zenodo.24237333).

Modules
-------
dispersion : modified dispersion relations (n=1 and n=2) and GRB photon
             time-of-flight
foam       : spacetime-foam distance fluctuations and interferometer phase noise
collapse   : GRW / Diosi-Penrose collapse toy models, CSL phenomenology,
             with an explicit non-computable oracle boundary (collapse
             itself is NOT simulated)
superluminal : superluminal LIV branch — photon decay, photon splitting,
             vacuum Cherenkov
birefringence : helicity-dependent photon speed, GRB polarisation rotation
gup        : generalised uncertainty principle — minimal length, oscillator
             levels, black-hole temperature
neutrinos  : energy-dependent neutrino speed, SN1987A / TXS 0506+056 /
             IceCube time-of-flight
constraints: real published observational limits and checks of the toy
             models against them
pilot      : the GPH sober blinded-pilot analysis pipeline (speckle synthesis,
             rater agreement, invariance checks)
cli        : command-line runner (``python -m gph_qgp run --experiment ...``,
             ``python -m gph_qgp compare --out ...``)

Honesty note (read this first): everything here is a toy model running on
simulated data. Nothing in this package is a real experiment, and nothing it
produces is evidence for or against GPH, simulation, or a curriculum.
A "positive" result inside these simulations only demonstrates that the
analysis pipeline works as designed.
"""

from . import (birefringence, collapse, constraints, dispersion, foam,
               gup, neutrinos, pilot, superluminal)

__version__ = "0.3.0"
__all__ = ["birefringence", "collapse", "constraints", "dispersion",
           "foam", "gup", "neutrinos", "pilot", "superluminal",
           "__version__"]
