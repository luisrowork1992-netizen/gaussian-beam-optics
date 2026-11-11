import math

import numpy as np
import pytest

import beamprop as bp


def test_thin_lens_imaging_equation():
    f, s = 0.1, 0.3
    s_img = bp.imaging_distance(bp.thin_lens(f), s)
    assert 1 / s + 1 / s_img == pytest.approx(1 / f, rel=1e-12)


def test_parallel_ray_crosses_axis_at_focus():
    f = 0.05
    M = bp.system(bp.thin_lens(f), bp.free_space(f))
    y, _ = bp.trace(M, 1e-3, 0.0)
    assert y == pytest.approx(0.0, abs=1e-15)


def test_determinant_equals_index_ratio():
    M = bp.system(bp.curved_interface(1.0, 1.5, 0.02), bp.free_space(0.004), bp.flat_interface(1.5, 1.33))
    assert np.linalg.det(M) == pytest.approx(1.0 / 1.33, rel=1e-12)


def test_thick_lens_matches_lensmaker():
    n, R1, R2, d = 1.5168, 0.05, -0.08, 0.006
    M = bp.thick_lens(n, R1, R2, d)
    assert bp.effective_focal_length(M) == pytest.approx(bp.lensmaker_focal_length(n, R1, R2, d), rel=1e-12)


def test_thin_lens_limit_of_lensmaker():
    # symmetric biconvex, n=1.5, R=0.1  ->  f = R / (2 (n-1)) = 0.1
    assert bp.lensmaker_focal_length(1.5, 0.1, -0.1) == pytest.approx(0.1)


def test_concave_mirror_focal_length_is_half_radius():
    assert bp.effective_focal_length(bp.curved_mirror(0.4)) == pytest.approx(0.2)


def test_beam_radius_at_rayleigh_range_is_sqrt2_waist():
    w0, lam = 1e-3, 1064e-9
    zr = bp.rayleigh_range(w0, lam)
    assert bp.beam_radius(zr, w0, lam) == pytest.approx(math.sqrt(2) * w0)
    assert bp.gouy_phase(zr, w0, lam) == pytest.approx(math.pi / 4)


def test_curvature_minimum_at_rayleigh_range():
    w0, lam = 0.5e-3, 633e-9
    zr = bp.rayleigh_range(w0, lam)
    z = np.linspace(0.2 * zr, 5 * zr, 2001)
    R = bp.curvature_radius(z, w0, lam)
    assert z[np.argmin(R)] == pytest.approx(zr, rel=1e-2)
    assert R.min() == pytest.approx(2 * zr, rel=1e-4)


def test_q_propagation_matches_closed_form():
    w0, lam, z = 0.3e-3, 800e-9, 0.7
    beam = bp.GaussianBeam.from_waist(w0, lam).propagate(bp.free_space(z))
    assert beam.w == pytest.approx(bp.beam_radius(z, w0, lam), rel=1e-12)
    assert beam.R == pytest.approx(float(bp.curvature_radius(z, w0, lam)), rel=1e-12)
    assert beam.waist == pytest.approx(w0, rel=1e-12)


def test_lens_focus_q_vs_self_formula():
    w0, lam, f, s = 2e-3, 1064e-9, 0.2, 0.5
    beam = bp.GaussianBeam.from_waist(w0, lam, z=s).propagate(bp.thin_lens(f))
    w_new, s_new = bp.focus_by_thin_lens(w0, lam, f, s)
    assert beam.waist == pytest.approx(w_new, rel=1e-10)
    assert beam.distance_to_waist == pytest.approx(s_new, rel=1e-10)


def test_collimated_focus_approximation():
    # z_R >> f  ->  w' ~ lam f / (pi w0)
    w0, lam, f = 5e-3, 1064e-9, 0.1
    w_new, s_new = bp.focus_by_thin_lens(w0, lam, f, 0.0)
    assert w_new == pytest.approx(lam * f / (math.pi * w0), rel=1e-3)
    assert s_new == pytest.approx(f, rel=1e-3)


def test_resonator_stability():
    assert bp.is_stable(1.0, math.inf, 2.0)       # hemispherical-ish, g1 g2 = 0.5
    assert bp.is_stable(1.0, 1.0, 1.0)            # confocal, g1 g2 = 0
    assert not bp.is_stable(1.0, 0.3, 0.3)        # g = -2.33 each -> g1 g2 > 1


def test_symmetric_confocal_mode_waist():
    # symmetric confocal-like limit: R = L(1 + eps) avoids g = 0 exactly
    L, lam = 0.5, 1064e-9
    w0 = bp.resonator_mode(L, L * 1.000001, L * 1.000001, lam)
    assert w0 == pytest.approx(math.sqrt(lam * L / (2 * math.pi)), rel=1e-4)


def test_plano_concave_resonator_mode():
    # waist on the flat mirror: z_R^2 = L (R - L)
    L, R, lam = 0.3, 1.0, 633e-9
    w0 = bp.resonator_mode(L, math.inf, R, lam)
    zr = math.sqrt(L * (R - L))
    assert w0 == pytest.approx(math.sqrt(lam * zr / math.pi), rel=1e-10)
