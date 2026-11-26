"""
Scenario-independent live sampling animation for Monte Carlo and MISER.

Supports:
- Binary integrands (e.g. circle area, joint default): points are coloured 0/1.
- Continuous integrands (e.g. interference, Gaussian peak): points are coloured
  by their actual function values.
- Optional geometric overlay, such as the circle boundary or Gaussian 2-sigma guide.
- MISER pilot samples (kind 0) vs. production samples (kind 1) by marker shape.
- Play / pause / restart / speed and a 2-D / 3-D view switch, all client-side
  (no Streamlit rerun, the algorithms are never re-executed).

    payload = build_payload(run_id, integrand, mc, ms, reference)

``integrand`` is an experiments.Integrand: its ``circle`` (overlay),
``area_mode`` (0/1 colouring), ``value_kind`` (labels) and ``view3d`` (surface)
fields describe how the scenario is drawn.

Timing: each algorithm's playback length at 1x is
PLAYBACK_SECONDS * elapsed_i / max(elapsed_mc, elapsed_ms), i.e. the measured
algorithm times scaled by one common factor. Samples are shown in their real
evaluation order, spread evenly over that length (per-sample timestamps are
not recorded).
"""

import pathlib
import shutil

import numpy as np
import plotly
import streamlit.components.v1 as components

from visualization.surface import surface_grid

_COMPONENT_DIR = pathlib.Path(__file__).parent / "live_component"
PLAYBACK_SECONDS = 15


def _ensure_plotly_js():
    """Serve the Plotly bundle locally so the component works without a CDN."""
    src = pathlib.Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    dst = _COMPONENT_DIR / "plotly.min.js"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)


_ensure_plotly_js()
_live_component = components.declare_component(
    "live_sampling", path=str(_COMPONENT_DIR)
)


def _r(a, digits=6):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def _empty_algo():
    return {"x": [], "y": [], "v": [], "elapsed": 0.0, "estimate": 0.0}


def _overlay(integrand):
    return list(integrand.circle) if integrand.circle is not None else None


def blank_payload(integrand, reference):
    """Idle state before the first run (empty plots, the scenario's labels)."""
    from experiments import labels_for  # local import: experiments -> miser only
    ms = _empty_algo()
    ms.update(kind=[], leaves=[], splits=[], nodes=[])
    return {
        "run_id": 0, "n": 0, "overlay": _overlay(integrand),
        "binary": bool(integrand.area_mode), "reference": float(reference),
        "labels": labels_for(integrand),
        "volume": 1.0, "bounds": [[0.0, 1.0], [0.0, 1.0]],
        "vrange": [0.0, 1.0], "surface": None,
        "seconds": PLAYBACK_SECONDS,
        "mc": _empty_algo(), "ms": ms,
    }


def build_payload(run_id, integrand, mc, ms, reference):
    """Serialize the actual evaluation order returned by the algorithms."""
    from experiments import labels_for  # local import: experiments -> miser only
    n = int(mc["evaluations"])
    assert n == int(ms["evaluations"]), "MC and MISER evaluation budgets differ"

    mc_values = np.asarray(mc["values"], dtype=float)
    ms_values = np.asarray(ms["values"], dtype=float)

    def algo_payload(result, values):
        values = np.asarray(values, dtype=float)
        return {
            "x": _r(result["points"][:, 0]),
            "y": _r(result["points"][:, 1]),
            # Preserve continuous values; do not cast to int.
            "v": _r(values, digits=8),
            "elapsed": float(result["elapsed"]),
            "estimate": float(result["estimate"]),
        }

    leaves = []
    for nd in sorted((nd for nd in ms["nodes"] if nd["leaf"]),
                     key=lambda nd: nd["eval_start"]):
        size = float(np.prod([hi - lo for lo, hi in nd["box"]]))
        (x0, x1), (y0, y1) = nd["box"]
        leaves.append([
            int(nd["eval_start"]), int(nd["count"]), size,
            x0, x1, y0, y1, int(nd["depth"]), float(nd["estimate"])
        ])

    splits = []
    for ev in ms["splits"]:
        (bx0, bx1), (by0, by1) = ev["box"]
        if ev["axis"] == 0:
            seg = [ev["midpoint"], by0, ev["midpoint"], by1]
        else:
            seg = [bx0, ev["midpoint"], bx1, ev["midpoint"]]
        splits.append([int(ev["after_eval"])] + seg + [int(ev["depth"])])

    nodes = sorted(
        [int(nd["eval_at_create"]), int(nd["depth"])] for nd in ms["nodes"]
    )

    bounds = [[float(lo), float(hi)] for lo, hi in ms["bounds"]]
    volume = float(np.prod([hi - lo for lo, hi in bounds]))
    surface = None
    if integrand.view3d and len(bounds) == 2:
        surface = surface_grid(integrand)
    # One colour / height range shared by both panels (and the 3-D surface).
    lo_vals = [float(np.min(v)) for v in (mc_values, ms_values) if v.size]
    hi_vals = [float(np.max(v)) for v in (mc_values, ms_values) if v.size]
    if surface is not None:
        lo_vals.append(surface["zmin"])
        hi_vals.append(surface["zmax"])
    vrange = [min(lo_vals, default=0.0), max(hi_vals, default=1.0)]

    return {
        "run_id": run_id,
        "n": n,
        "overlay": _overlay(integrand),
        "binary": bool(integrand.area_mode),
        "reference": float(reference),
        "labels": labels_for(integrand),
        "volume": volume,
        "bounds": bounds,
        "vrange": vrange,
        "surface": surface,
        "seconds": PLAYBACK_SECONDS,
        "mc": algo_payload(mc, mc_values),
        "ms": {
            **algo_payload(ms, ms_values),
            "kind": np.asarray(ms["kinds"], dtype=int).tolist(),
            "leaves": leaves,
            "splits": splits,
            "nodes": nodes,
        },
    }


def live_sampling(payload, key="live-sampling"):
    """Render the player. Returns run_id after the animation finishes."""
    return _live_component(payload=payload, key=key, default=None)
