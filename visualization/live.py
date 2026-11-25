"""
Scenario-independent live sampling animation for Monte Carlo and MISER.

Supports:
- Binary integrands (e.g. circle area, joint default): points are coloured 0/1.
- Continuous integrands (e.g. interference, Gaussian peak): points are coloured
  by their actual function values.
- Optional geometric overlay, such as the circle boundary or Gaussian 2-sigma guide.

The existing call shape remains compatible:
    payload = build_payload(run_id, integrand, mc, ms, reference)
Optional metadata can be supplied as keyword arguments:
    build_payload(..., overlay=(cx, cy, r), binary=False, label="...")
"""

import pathlib
import shutil

import numpy as np
import plotly
import streamlit.components.v1 as components

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
    return {
        "x": [], "y": [], "v": [],
        "elapsed": 0.0, "estimate": 0.0,
        "value_min": 0.0, "value_max": 1.0,
    }


def _metadata(integrand, mc_values, overlay=None, binary=None, label=None):
    """Read optional scenario metadata without requiring a particular class."""
    meta = integrand if isinstance(integrand, dict) else {}
    if overlay is None:
        overlay = meta.get("overlay", getattr(integrand, "overlay", None))
    if overlay is None:
        overlay = getattr(integrand, "circle", None)

    if binary is None:
        binary = meta.get("binary", getattr(integrand, "binary", None))
    if binary is None:
        vals = np.asarray(mc_values, dtype=float)
        binary = bool(vals.size == 0 or np.all(np.isclose(vals, 0) | np.isclose(vals, 1)))

    if label is None:
        label = meta.get("label", getattr(integrand, "label", "Integrand"))

    return overlay, bool(binary), str(label)


def blank_payload(circle=None, reference=0.0, binary=True, label="Integrand"):
    """Idle state. The overlay is optional; no circle is assumed."""
    ms = _empty_algo()
    ms.update(kind=[], leaves=[], splits=[], nodes=[])
    overlay = list(circle) if circle is not None else None
    return {
        "run_id": 0, "n": 0, "overlay": overlay,
        "binary": bool(binary), "label": label, "reference": reference,
        "seconds": PLAYBACK_SECONDS,
        "mc": _empty_algo(), "ms": ms,
    }


def build_payload(run_id, integrand, mc, ms, reference,
                  overlay=None, binary=None, label=None):
    """Serialize the actual evaluation order returned by the algorithms."""
    n = int(mc["evaluations"])
    assert n == int(ms["evaluations"]), "MC and MISER evaluation budgets differ"

    mc_values = np.asarray(mc["values"], dtype=float)
    ms_values = np.asarray(ms["values"], dtype=float)
    overlay, binary, label = _metadata(
        integrand, mc_values, overlay=overlay, binary=binary, label=label
    )

    def algo_payload(result, values):
        values = np.asarray(values, dtype=float)
        return {
            "x": _r(result["points"][:, 0]),
            "y": _r(result["points"][:, 1]),
            # Preserve continuous values; do not cast to int.
            "v": _r(values, digits=8),
            "elapsed": float(result["elapsed"]),
            "estimate": float(result["estimate"]),
            "value_min": float(np.min(values)) if values.size else 0.0,
            "value_max": float(np.max(values)) if values.size else 1.0,
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

    return {
        "run_id": run_id,
        "n": n,
        "overlay": list(overlay) if overlay is not None else None,
        "binary": binary,
        "label": label,
        "reference": float(reference),
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
