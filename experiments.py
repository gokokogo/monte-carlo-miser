"""
Integrand registry, reference values, and experiment runners.

이 모듈은 알고리즘(miser.py)을 *호출만* 한다. 적분 추정은 전부
run_plain_monte_carlo / run_miser 가 수행하며, 여기서는 기준값 계산과
반복 실행/통계 집계만 담당한다.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from miser import (
    SIGMA,
    VILLAGE_RADIUS,
    UNIT_SQUARE,
    VILLAGE_X,
    VILLAGE_Y,
    run_miser,
    run_plain_monte_carlo,
    village_density_points,
)
from scenarios import SCENARIOS

# Sample sizes used by the convergence / error / performance sweeps.
SWEEP_SIZES = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)

# Relative error is not reported when |reference| is below this threshold.
REL_ERROR_EPS = 1e-12


@dataclass(frozen=True)
class Integrand:
    key: str
    label: str
    dim: int
    func: Callable[[np.ndarray], np.ndarray]
    scalar: Optional[Callable[[np.ndarray], np.ndarray]]  # f(x) for plotting (1D)
    reference: Callable[[list], float]
    reference_kind: str
    fixed_bounds: Optional[tuple] = None
    circle: Optional[tuple] = None   # (cx, cy, r) drawn on 2D plots
    area_mode: bool = False          # f is a 0/1 indicator: integral = area
    value_kind: str = "continuous"   # "circle" | "binary" | "continuous" (labels)
    view3d: bool = False             # offer the 3-D surface view


# Display labels per kind of integrand. The web app and the live player both
# read these, so scenario-specific wording lives only here.
LABELS = {
    "circle": {
        "one": "Inside circle", "zero": "Outside circle",
        "estimate": "Final estimate (circle area)",
        "reference": "Reference value (πr²)",
        "how_mc": "inside / total × 1", "how_ms": "Σ region area × fraction",
    },
    "binary": {
        "one": "Inside region", "zero": "Outside region",
        "estimate": "Final estimate (area of the f = 1 region)",
        "reference": "Reference value (exact integral)",
        "how_mc": "inside / total × square area",
        "how_ms": "Σ region area × fraction inside",
    },
    "continuous": {
        "one": None, "zero": None, "value": "Function value",
        "estimate": "Final estimate (integral of f)",
        "reference": "Reference value (exact integral)",
        "how_mc": "square area × mean f",
        "how_ms": "Σ region area × mean f in region",
    },
}
SAMPLE_LABELS = {"pilot": "Pilot sample", "production": "Sampling point"}


def labels_for(integrand):
    """Display labels for an integrand (see LABELS)."""
    kind = getattr(integrand, "value_kind", None)
    if kind not in LABELS:
        kind = "binary" if getattr(integrand, "area_mode", False) else "continuous"
    return {"kind": kind, **LABELS[kind], **SAMPLE_LABELS}


def _ref_village(bounds):
    # Analytic: 0.001 * area + 2*pi*sigma^2 * Px * Py, where Px, Py are the
    # Gaussian masses (std = SIGMA) inside the box along each axis.
    (x0, x1), (y0, y1) = bounds
    s = SIGMA * math.sqrt(2.0)
    px = 0.5 * (math.erf((x1 - VILLAGE_X) / s) - math.erf((x0 - VILLAGE_X) / s))
    py = 0.5 * (math.erf((y1 - VILLAGE_Y) / s) - math.erf((y0 - VILLAGE_Y) / s))
    area = (x1 - x0) * (y1 - y0)
    return 0.001 * area + 2.0 * math.pi * SIGMA**2 * px * py


INTEGRANDS = {
    "x2": Integrand(
        "x2", "f(x) = x²", 1,
        func=lambda p: p[:, 0] ** 2,
        scalar=lambda x: x**2,
        reference=lambda b: (b[0][1] ** 3 - b[0][0] ** 3) / 3.0,
        reference_kind="Analytic: (b³ − a³) / 3",
    ),
    "sin": Integrand(
        "sin", "f(x) = sin(x)", 1,
        func=lambda p: np.sin(p[:, 0]),
        scalar=np.sin,
        reference=lambda b: math.cos(b[0][0]) - math.cos(b[0][1]),
        reference_kind="Analytic: cos(a) − cos(b)",
    ),
    "cos": Integrand(
        "cos", "f(x) = cos(x)", 1,
        func=lambda p: np.cos(p[:, 0]),
        scalar=np.cos,
        reference=lambda b: math.sin(b[0][1]) - math.sin(b[0][0]),
        reference_kind="Analytic: sin(b) − sin(a)",
    ),
    "gauss": Integrand(
        "gauss", "f(x) = exp(−x²)", 1,
        func=lambda p: np.exp(-p[:, 0] ** 2),
        scalar=lambda x: np.exp(-x**2),
        reference=lambda b: math.sqrt(math.pi) / 2.0
        * (math.erf(b[0][1]) - math.erf(b[0][0])),
        reference_kind="Analytic: (√π / 2)·(erf(b) − erf(a))",
    ),
    "village": Integrand(
        "village", "Village density f(x, y) — 2D, original demo", 2,
        func=village_density_points,
        scalar=None,
        reference=_ref_village,
        reference_kind="Analytic (Gaussian bump via erf) over [0,1]²",
        fixed_bounds=UNIT_SQUARE,
        circle=(VILLAGE_X, VILLAGE_Y, VILLAGE_RADIUS),
    ),
}


def _scenario_integrand(key, sc):
    """Wrap a scenarios.SCENARIOS entry as an Integrand on the unit square."""
    return Integrand(
        key, sc["label"], 2, func=sc["func"], scalar=None,
        reference=(lambda b, _f=sc["exact"]: _f()),
        reference_kind="exact value (scenarios.py)",
        fixed_bounds=UNIT_SQUARE, circle=sc["overlay"], area_mode=sc["binary"],
        value_kind=sc["kind"], view3d=sc["view3d"])


# Single registry used by the web app and the experiment runners: every
# scenario (incl. the Gaussian peaks) goes through run_plain_monte_carlo /
# run_miser via run_pair.
SCENARIO_INTEGRANDS = {k: _scenario_integrand(k, sc) for k, sc in SCENARIOS.items()}
# The circle scenario is also listed with the other integrands (one definition:
# scenarios.circle, centre (0.70, 0.30), r = 0.15, reference πr²).
INTEGRANDS["circle"] = SCENARIO_INTEGRANDS["circle"]


def standard_error(result):
    """Standard error of an algorithm result.

    MISER reports its own (Σ V²s²/N over leaves). For plain MC it is
    V · s / √N from the returned sample values (computed here so that the
    plain MC algorithm itself stays unchanged).
    """
    if "std_error" in result:
        return float(result["std_error"])
    n = len(result["values"])
    if n < 2:
        return float("nan")
    volume = float(np.prod([hi - lo for lo, hi in result["bounds"]]))
    return volume * float(np.std(result["values"], ddof=1)) / math.sqrt(n)


def make_bounds(integrand, a, b):
    if integrand.fixed_bounds is not None:
        return [list(x) for x in integrand.fixed_bounds]
    return [[float(a), float(b)]]


def validate_bounds(a, b):
    """Return an error message, or None if the interval is valid."""
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return "Bounds must be finite numbers."
    if a >= b:
        return "Lower bound (a) must be strictly less than upper bound (b)."
    return None


def seeds_for(seed):
    """Seeds used for one comparison run.

    Kept from the original app: MC uses ``seed`` and MISER uses ``seed + 1``
    so the two random streams are not identical.
    """
    return int(seed), int(seed) + 1


def run_pair(integrand, bounds, n, seed, leaf_size):
    mc_seed, miser_seed = seeds_for(seed)
    mc = run_plain_monte_carlo(n, mc_seed, func=integrand.func, bounds=bounds)
    ms = run_miser(n, miser_seed, leaf_size, func=integrand.func, bounds=bounds)
    return mc, ms


def errors(estimate, reference):
    abs_err = abs(estimate - reference)
    rel_err = abs_err / abs(reference) if abs(reference) > REL_ERROR_EPS else None
    return abs_err, rel_err


def sweep(integrand, bounds, seed, leaf_size, sizes=SWEEP_SIZES):
    """One run of each algorithm per sample size (same seeds as the main run)."""
    ref = integrand.reference(bounds)
    rows = []
    for n in sizes:
        mc, ms = run_pair(integrand, bounds, n, seed, leaf_size)
        for name, res in (("Standard Monte Carlo", mc), ("MISER", ms)):
            rows.append({
                "algorithm": name,
                "n": n,
                "estimate": res["estimate"],
                "abs_error": abs(res["estimate"] - ref),
                "std_error": standard_error(res),
                "time": res["elapsed"],
                "evaluations": res["evaluations"],
            })
    return rows


def repeated(integrand, bounds, seed, leaf_size, repeats, sizes=SWEEP_SIZES):
    """Independent repetitions per sample size; returns summary rows.

    Repetition r uses base seed ``seed + 1000 * (r + 1)`` so that repetitions
    are independent of each other and of the single-run experiment.
    """
    ref = integrand.reference(bounds)
    rows = []
    for n in sizes:
        est = {"Standard Monte Carlo": [], "MISER": []}
        times = {"Standard Monte Carlo": [], "MISER": []}
        ses = {"Standard Monte Carlo": [], "MISER": []}
        for r in range(repeats):
            mc, ms = run_pair(integrand, bounds, n, seed + 1000 * (r + 1), leaf_size)
            for name, res in (("Standard Monte Carlo", mc), ("MISER", ms)):
                est[name].append(res["estimate"])
                times[name].append(res["elapsed"])
                ses[name].append(standard_error(res))
        for name in est:
            e = np.asarray(est[name])
            err = e - ref
            rows.append({
                "algorithm": name,
                "n": n,
                "repeats": repeats,
                "mean_estimate": float(e.mean()),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "std": float(e.std(ddof=1)) if repeats > 1 else 0.0,
                "mean_std_error": float(np.mean(ses[name])),
                "mean_time": float(np.mean(times[name])),
            })
    return rows


def regions_at_depth(nodes, depth):
    """Regions present after ``depth`` levels of MISER splitting.

    A region is either an internal/leaf node at exactly this depth, or a leaf
    that stopped splitting at a shallower depth.
    """
    return [
        nd for nd in nodes
        if nd["depth"] == depth or (nd["leaf"] and nd["depth"] < depth)
    ]
