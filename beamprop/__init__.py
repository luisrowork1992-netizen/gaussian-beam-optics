"""beamprop: paraxial ray optics and Gaussian-beam propagation."""

from .abcd import (
    curved_interface,
    curved_mirror,
    effective_focal_length,
    flat_interface,
    free_space,
    imaging_distance,
    lensmaker_focal_length,
    system,
    thick_lens,
    thin_lens,
    trace,
)
from .gaussian import (
    GaussianBeam,
    beam_radius,
    curvature_radius,
    focus_by_thin_lens,
    gouy_phase,
    is_stable,
    rayleigh_range,
    resonator_g,
    resonator_mode,
)

__version__ = "0.1.0"
