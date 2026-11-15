"""gph_qgp — the GPH Quantum Gravity Phenomenology lab (v0.1.0).

Toy phenomenological models for teaching and exploration, built around
the Graduation Protocol Hypothesis (GPH) paper, v3.2
(Zenodo DOI 10.5281/zenodo.24237333).

Modules
-------
dispersion : modified dispersion relations and GRB photon time-of-flight
foam       : spacetime-foam distance fluctuations and interferometer phase noise
collapse   : GRW / Diosi-Penrose collapse toy models, with an explicit
             non-computable oracle boundary (collapse itself is NOT simulated)
pilot      : the GPH sober blinded-pilot analysis pipeline (speckle synthesis,
             rater agreement, invariance checks)
cli        : command-line runner (``python -m gph_qgp run --experiment ...``)

Honesty note (read this first): everything here is a toy model running on
simulated data. Nothing in this package is a real experiment, and nothing it
produces is evidence for or against GPH, simulation, or a curriculum.
A "positive" result inside these simulations only demonstrates that the
analysis pipeline works as designed.
"""

from . import collapse, dispersion, foam, pilot

__version__ = "0.1.0"
__all__ = ["collapse", "dispersion", "foam", "pilot", "__version__"]
