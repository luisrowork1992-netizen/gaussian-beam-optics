"""Gaussian-beam propagation with the complex beam parameter ``q``.

``1/q = 1/R - i * lam / (pi * w**2)``, so at a waist ``q = i * z_R``.
All lengths share one unit (e.g. metres or millimetres).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "GaussianBeam",
    "rayleigh_range",
    "beam_radius",
    "curvature_radius",
    "gouy_phase",
    "focus_by_thin_lens",
    "resonator_g",
    "is_stable",
    "resonator_mode",
]


def rayleigh_range(w0: float, lam: float, n: float = 1.0, m2: float = 1.0) -> float:
    """Rayleigh range ``pi w0^2 n / (M^2 lam)``."""
    return math.pi * w0**2 * n / (m2 * lam)


def beam_radius(z, w0: float, lam: float, n: float = 1.0, m2: float = 1.0):
    """1/e^2 intensity radius at distance ``z`` from the waist."""
    zr = rayleigh_range(w0, lam, n, m2)
    return w0 * np.sqrt(1.0 + (np.asarray(z) / zr) ** 2)


def curvature_radius(z, w0: float, lam: float, n: float = 1.0):
    """Wavefront radius of curvature (inf at the waist)."""
    zr = rayleigh_range(w0, lam, n)
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(z == 0, np.inf, z * (1.0 + (zr / z) ** 2))


def gouy_phase(z, w0: float, lam: float, n: float = 1.0):
    """Gouy phase ``arctan(z / z_R)``."""
    return np.arctan(np.asarray(z) / rayleigh_range(w0, lam, n))


@dataclass
class GaussianBeam:
    """A fundamental Gaussian beam described by its ``q`` parameter at some plane."""

    q: complex
    lam: float
    n: float = 1.0

    @classmethod
    def from_waist(cls, w0: float, lam: float, z: float = 0.0, n: float = 1.0) -> "GaussianBeam":
        """Beam observed a distance ``z`` after a waist of radius ``w0``."""
        zr = rayleigh_range(w0, lam, n)
        return cls(q=complex(z, zr), lam=lam, n=n)

    @property
    def w(self) -> float:
        """Beam radius at this plane."""
        inv = 1.0 / self.q
        return math.sqrt(-self.lam / (math.pi * self.n * inv.imag))

    @property
    def R(self) -> float:
        """Wavefront radius of curvature at this plane."""
        inv = 1.0 / self.q
        return math.inf if inv.real == 0 else 1.0 / inv.real

    @property
    def z_r(self) -> float:
        return self.q.imag

    @property
    def waist(self) -> float:
        """Waist radius of this beam."""
        return math.sqrt(self.q.imag * self.lam / (math.pi * self.n))

    @property
    def distance_to_waist(self) -> float:
        """Signed distance from this plane to the waist (positive: waist is ahead)."""
        return -self.q.real

    def propagate(self, M: np.ndarray) -> "GaussianBeam":
        """Transform through an ABCD matrix ``q' = (A q + B) / (C q + D)``."""
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        q2 = (A * self.q + B) / (C * self.q + D)
        return GaussianBeam(q=complex(q2), lam=self.lam, n=self.n)


def focus_by_thin_lens(w0: float, lam: float, f: float, s: float = 0.0) -> tuple[float, float]:
    """Focus a beam whose waist sits a distance ``s`` before a thin lens.

    Returns ``(new_waist, distance_after_lens)`` using Self's (1983) beam form of
    the lens equation.
    """
    zr = rayleigh_range(w0, lam)
    denom = (s - f) ** 2 + zr**2
    w_new = w0 * f / math.sqrt(denom)
    s_new = f + f**2 * (s - f) / denom
    return w_new, s_new


def resonator_g(L: float, R1: float, R2: float) -> tuple[float, float]:
    """g-parameters ``g_i = 1 - L / R_i`` for a two-mirror resonator."""
    g1 = 1.0 - (0.0 if math.isinf(R1) else L / R1)
    g2 = 1.0 - (0.0 if math.isinf(R2) else L / R2)
    return g1, g2


def is_stable(L: float, R1: float, R2: float) -> bool:
    """A two-mirror resonator is stable when ``0 <= g1 g2 <= 1``."""
    g1, g2 = resonator_g(L, R1, R2)
    return 0.0 <= g1 * g2 <= 1.0


def resonator_mode(L: float, R1: float, R2: float, lam: float) -> float:
    """Waist radius of the TEM00 mode of a stable two-mirror resonator."""
    g1, g2 = resonator_g(L, R1, R2)
    prod = g1 * g2
    if not 0.0 < prod < 1.0:
        raise ValueError("resonator is not strictly stable")
    w0_sq = (lam * L / math.pi) * math.sqrt(prod * (1 - prod) / (g1 + g2 - 2 * prod) ** 2)
    return math.sqrt(w0_sq)
