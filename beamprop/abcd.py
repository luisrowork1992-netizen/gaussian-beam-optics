"""Paraxial ray-transfer (ABCD) matrices.

Rays are column vectors ``[y, theta]`` (height, angle in radians). Matrices
are 2x2 NumPy arrays, and a system is built by multiplying elements
right-to-left in the order light meets them, which :func:`system` does for you.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

__all__ = [
    "free_space",
    "thin_lens",
    "flat_interface",
    "curved_interface",
    "curved_mirror",
    "thick_lens",
    "system",
    "trace",
    "effective_focal_length",
    "imaging_distance",
    "lensmaker_focal_length",
]


def free_space(d: float, n: float = 1.0) -> np.ndarray:
    """Propagation over distance ``d`` in a medium of index ``n`` (reduced-angle form uses n=1)."""
    return np.array([[1.0, d / n], [0.0, 1.0]])


def thin_lens(f: float) -> np.ndarray:
    """Thin lens of focal length ``f`` (positive = converging)."""
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def flat_interface(n1: float, n2: float) -> np.ndarray:
    """Refraction at a plane interface from ``n1`` into ``n2``."""
    return np.array([[1.0, 0.0], [0.0, n1 / n2]])


def curved_interface(n1: float, n2: float, R: float) -> np.ndarray:
    """Refraction at a spherical interface of radius ``R`` (R > 0: centre after the surface)."""
    return np.array([[1.0, 0.0], [(n1 - n2) / (R * n2), n1 / n2]])


def curved_mirror(R: float) -> np.ndarray:
    """Reflection from a spherical mirror, ``R > 0`` for concave (focusing)."""
    return np.array([[1.0, 0.0], [-2.0 / R, 1.0]])


def system(*elements: np.ndarray) -> np.ndarray:
    """Combine elements given in the order light traverses them."""
    return reduce(lambda acc, m: m @ acc, elements, np.eye(2))


def thick_lens(n_lens: float, R1: float, R2: float, d: float, n_out: float = 1.0) -> np.ndarray:
    """Thick lens in a medium of index ``n_out``."""
    return system(
        curved_interface(n_out, n_lens, R1),
        free_space(d),
        curved_interface(n_lens, n_out, R2),
    )


def trace(M: np.ndarray, y: float, theta: float) -> tuple[float, float]:
    """Apply a system matrix to a ray."""
    out = M @ np.array([y, theta])
    return float(out[0]), float(out[1])


def effective_focal_length(M: np.ndarray) -> float:
    """Effective focal length ``-1/C`` of a system in air."""
    C = M[1, 0]
    if C == 0:
        return float("inf")
    return -1.0 / C


def imaging_distance(M: np.ndarray, s_obj: float) -> float:
    """Image distance after the last element for an object ``s_obj`` before the first.

    Solves ``B = 0`` for ``free_space(s_img) @ M @ free_space(s_obj)``.
    """
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    num = A * s_obj + B
    den = C * s_obj + D
    if den == 0:
        return float("inf")
    return -num / den


def lensmaker_focal_length(n_lens: float, R1: float, R2: float, d: float = 0.0) -> float:
    """Lensmaker's equation for a lens in air (thick-lens form when ``d > 0``)."""
    power = (n_lens - 1.0) * (1.0 / R1 - 1.0 / R2 + (n_lens - 1.0) * d / (n_lens * R1 * R2))
    return 1.0 / power
