"""
Static 3-D surface data for the live player.

The surface height is computed by calling the scenario's own integrand
function, so the plotted surface and the integrated function can never
diverge. Only the *placement* of grid lines uses the scenario's overlay
(cx, cy, r): a fine window around the peak, coarse elsewhere.

Grid choice (measured for the σ = 0.03 peak, bilinear surface vs. true f on
600k test points): uniform 151×151 -> max error 1.2 % of the peak height;
uniform 301×301 -> 0.31 %; uniform 101 + ±4σ window at σ/8 (157×157) ->
0.39 % with a quarter of the 301 grid's points. The last one is used.
"""

import numpy as np

COARSE_POINTS = 101       # uniform lines over each axis
FOCUS_HALF_WIDTHS = 2.0   # window half-width in overlay radii (r = 2σ -> ±4σ)
FOCUS_STEPS_PER_R = 16    # fine spacing r/16 = σ/8
FALLBACK_POINTS = 201     # uniform grid when no overlay is known

_CACHE = {}


def _axis(lo, hi, centre=None, r=None):
    if centre is None or not r:
        return np.linspace(lo, hi, FALLBACK_POINTS)
    coarse = np.linspace(lo, hi, COARSE_POINTS)
    half = FOCUS_HALF_WIDTHS * r
    fine = np.arange(max(lo, centre - half), min(hi, centre + half) + 1e-12,
                     r / FOCUS_STEPS_PER_R)
    pts = np.concatenate([coarse, fine, [centre]])
    pts = pts[(pts >= lo) & (pts <= hi)]
    return np.unique(np.round(pts, 12))


def surface_grid(integrand, digits=5):
    """{x, y, z, zmin, zmax} for a 2-D integrand on its fixed bounds (cached).

    ``z[j][i] = f(x[i], y[j])`` (Plotly's surface convention).
    """
    key = getattr(integrand, "key", None)
    if key in _CACHE:
        return _CACHE[key]
    (x0, x1), (y0, y1) = integrand.fixed_bounds
    overlay = getattr(integrand, "circle", None)
    cx, cy, r = overlay if overlay is not None else (None, None, None)
    ax = _axis(x0, x1, cx, r)
    ay = _axis(y0, y1, cy, r)
    xx, yy = np.meshgrid(ax, ay)  # rows = y, columns = x
    z = np.asarray(integrand.func(np.column_stack([xx.ravel(), yy.ravel()])),
                   dtype=float).reshape(xx.shape)
    out = {
        "x": np.round(ax, 6).tolist(),
        "y": np.round(ay, 6).tolist(),
        "z": np.round(z, digits).tolist(),
        "zmin": float(z.min()),
        "zmax": float(z.max()),
    }
    if key is not None:
        _CACHE[key] = out
    return out
