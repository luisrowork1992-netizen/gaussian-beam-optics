# gaussian-beam-optics

![tests](https://github.com/luisrowork1992-netizen/gaussian-beam-optics/actions/workflows/tests.yml/badge.svg)

`beamprop` is a compact Python toolkit for **paraxial ray optics** (ABCD matrices) and **Gaussian-beam propagation** with the complex beam parameter *q*.

## Features

- ABCD matrices for free space, thin and thick lenses, plane and spherical interfaces, and spherical mirrors
- Effective focal length, imaging distance, and the thick-lens lensmaker's equation
- Gaussian beams: Rayleigh range, beam radius, wavefront curvature, Gouy phase, and an M² factor
- `GaussianBeam` objects that propagate through any ABCD system
- Closed-form thin-lens focusing (Self's beam lens equation)
- Two-mirror resonator g-parameters, stability test and TEM00 mode waist

## Install

```bash
git clone https://github.com/luisrowork1992-netizen/gaussian-beam-optics.git
cd gaussian-beam-optics
pip install -e .
```

## Quick start

```python
import beamprop as bp

lam = 1064e-9                         # metres
beam = bp.GaussianBeam.from_waist(2e-3, lam, z=0.5)
focused = beam.propagate(bp.thin_lens(0.2))
print(focused.waist, focused.distance_to_waist)

# The same result in closed form
print(bp.focus_by_thin_lens(2e-3, lam, f=0.2, s=0.5))

# Thick-lens focal length agrees with the lensmaker's equation
M = bp.thick_lens(1.5168, 0.05, -0.08, 0.006)
print(bp.effective_focal_length(M), bp.lensmaker_focal_length(1.5168, 0.05, -0.08, 0.006))
```

`system(*elements)` takes elements **in the order light reaches them** and multiplies them right-to-left for you.

## Notebook

[`notebooks/beam_propagation.ipynb`](notebooks/beam_propagation.ipynb) covers four examples:

- free-space beam expansion
- lens focusing, comparing the q-parameter method with the closed form
- a 5x Keplerian beam expander
- a resonator stability diagram

## Tests

```bash
pip install pytest
pytest
```

The tests compare results against textbook identities:

- the thin-lens equation
- det(M) = n₁/n₂
- the lensmaker's equation, in its thin and thick forms
- w(z_R) = √2·w₀
- the minimum wavefront radius of 2·z_R
- the focused waist λf/(πw₀) for a collimated beam
- the confocal and plano-concave resonator mode sizes

## References

- A. E. Siegman, *Lasers*, University Science Books (1986).
- S. A. Self, "Focusing of spherical Gaussian beams," *Appl. Opt.* 22, 658 (1983).
- B. E. A. Saleh and M. C. Teich, *Fundamentals of Photonics*, Wiley.

## License

MIT
