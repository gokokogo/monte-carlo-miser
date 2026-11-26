"""
Integration tests: every scenario (incl. the Gaussian peaks) runs through the
single shared run_plain_monte_carlo / run_miser, and the web app handles the
continuous Gaussian scenarios with value-based (not point-count) statistics.
"""

import math
import pathlib
import re
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments  # noqa: E402
import miser  # noqa: E402
from experiments import (  # noqa: E402
    SCENARIO_INTEGRANDS, errors, make_bounds, repeated, run_pair,
    standard_error, sweep,
)
from scenarios import DEMO, PEAK_CENTER, PEAK_SIGMA, SCENARIOS  # noqa: E402

GAUSSIANS = ("peak",)  # Gaussian-peak scenarios the gaussian-peak app actually ran


def _grid_integral(f, m=4001):
    xs = np.linspace(0.0, 1.0, m)
    xx, yy = np.meshgrid(xs, xs)
    v = f(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    # trapezoid rule written out (np.trapezoid needs numpy >= 2; numpy >= 1.26 is allowed)
    w = np.full(m, xs[1] - xs[0])
    w[0] = w[-1] = w[0] / 2
    return float(w @ v @ w)


# ------------------------------------------------- scenario registry
def test_registry_covers_every_scenario_and_preset():
    assert list(SCENARIO_INTEGRANDS) == list(SCENARIOS)
    assert set(DEMO) == set(SCENARIOS)
    # Existing scenarios keep their definitions; the unused "gaussian" is gone.
    assert list(SCENARIOS) == ["circle", "joint_default", "interference", "peak"]
    for key, it in SCENARIO_INTEGRANDS.items():
        assert it.dim == 2 and make_bounds(it, 0, 1) == [[0.0, 1.0], [0.0, 1.0]]
        assert it.area_mode == SCENARIOS[key]["binary"]
        assert it.value_kind == SCENARIOS[key]["kind"]
        assert it.view3d == SCENARIOS[key]["view3d"]


# Exact value and f at fixed points, recorded from gaussian-peak/scenarios.py
# (the scenarios the gaussian-peak app ran) before that folder was removed
# (it is still in git history, commit b0cd5fd).
GAUSSIAN_PEAK_APP = {
    "circle": (0.07068583470577035, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "joint_default": (0.0016643604715854906, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "interference": (1.000023745006957, [0.8896127561609525, 1.249540117973338, 1.249540117973338, 1.862318872287684, 1.862318872287684, 0.21776911011288413, 1.7648421872844884, 0.20957602580218437, 0.5838531634528576, 1.862318872287684, 0.0009791866853525732, 1.862318872287684]),
    "peak": (0.005654866776461626, [1.0, 0.606530659712634, 0.08208499862389872, 1.494533852478153e-05, 7.456392881248971e-25, 1.9869020590398477e-12, 1.3684644745720435e-76, 1.72605960481832e-36, 3.6740583978360214e-74, 9.25972017590642e-83, 6.126970855586505e-35, 1.045241266044843e-125]),
}
GAUSSIAN_PEAK_POINTS = [[0.4, 0.4], [0.43, 0.4], [0.37, 0.46], [0.5, 0.5], [0.7, 0.3],
                        [0.62, 0.41], [0.005, 0.002], [0.015, 0.4], [0.019, 0.001],
                        [0.1, 0.9], [0.33, 0.77], [0.999, 0.001]]


def test_scenarios_match_the_gaussian_peak_app():
    """Every scenario the gaussian-peak app ran is still here, unchanged."""
    pts = np.array(GAUSSIAN_PEAK_POINTS)
    for key, (exact, values) in GAUSSIAN_PEAK_APP.items():
        assert key in SCENARIOS
        assert SCENARIOS[key]["exact"]() == exact
        # rtol 1e-12 only absorbs last-bit differences of vectorised exp/cos
        # between CPUs; any change of formula or parameter fails.
        assert np.allclose(SCENARIOS[key]["func"](pts), values, rtol=1e-12, atol=0)


# ------------------------------------------------- labels (one place)
def test_labels_follow_the_scenario_kind():
    from experiments import labels_for
    L = {k: labels_for(it) for k, it in SCENARIO_INTEGRANDS.items()}
    assert (L["circle"]["one"], L["circle"]["zero"]) == ("Inside circle", "Outside circle")
    assert (L["joint_default"]["one"], L["joint_default"]["zero"]) == (
        "Inside region", "Outside region")
    for k in ("interference", "peak"):
        assert L[k]["one"] is None and L[k]["value"] == "Function value"
    for lab in L.values():
        assert lab["pilot"] == "Pilot sample" and lab["production"] == "Sampling point"


# ------------------------------------------------- Gaussian definitions
def test_peak_is_a_continuous_gaussian():
    f = SCENARIOS["peak"]["func"]
    cx, cy = PEAK_CENTER
    s = PEAK_SIGMA
    assert (cx, cy, s) == (0.40, 0.40, 0.03)  # gaussian-peak/scenarios.py values
    assert f(np.array([[cx, cy]]))[0] == 1.0  # height A = 1 at the centre
    # Continuous, radially decreasing, symmetric.
    r = np.linspace(0, 0.15, 50)
    vals = f(np.column_stack([cx + r, np.full_like(r, cy)]))
    assert np.all(np.diff(vals) < 0)
    assert np.allclose(vals, np.exp(-r**2 / (2 * s**2)))
    pts = np.array([[cx + 0.05, cy], [cx - 0.05, cy], [cx, cy + 0.05], [cx, cy - 0.05]])
    assert np.allclose(f(pts), f(pts)[0])
    # Not an indicator: many distinct values.
    assert len(np.unique(f(np.random.default_rng(0).random((1000, 2))))) > 900


@pytest.mark.parametrize("key", GAUSSIANS)
def test_gaussian_reference_is_the_finite_square_integral(key):
    sc = SCENARIOS[key]
    exact = sc["exact"]()
    assert exact == pytest.approx(_grid_integral(sc["func"]), abs=1e-10)


def test_gaussian_reference_is_not_the_whole_plane_value():
    # The finite-square value is not larger than the whole-plane 2πσ² (the
    # centre is >13σ from every edge, so they agree to ~1e-16 here).
    exact = SCENARIOS["peak"]["exact"]()
    plane = 2 * math.pi * PEAK_SIGMA**2
    assert exact <= plane and plane - exact < 1e-12


# ------------------------------------------------- 3-D surface
def test_surface_uses_the_scenario_function_and_resolves_the_peak():
    from visualization.surface import surface_grid
    it = SCENARIO_INTEGRANDS["peak"]
    sf = surface_grid(it)
    x, y, z = np.array(sf["x"]), np.array(sf["y"]), np.array(sf["z"])
    assert x[0] == 0.0 and x[-1] == 1.0 and y[0] == 0.0 and y[-1] == 1.0
    xx, yy = np.meshgrid(x, y)
    truth = SCENARIOS["peak"]["func"](np.column_stack([xx.ravel(), yy.ravel()]))
    assert np.allclose(z.ravel(), truth, atol=1e-5)  # same function (5-digit rounding)
    assert sf["zmax"] == pytest.approx(1.0)  # the peak top is a grid node
    # Fine spacing σ/8 inside ±4σ of the centre, coarse (1/100) elsewhere.
    inside = x[(x > 0.40 - 4 * PEAK_SIGMA) & (x < 0.40 + 4 * PEAK_SIGMA)]
    assert np.max(np.diff(inside)) <= PEAK_SIGMA / 8 + 1e-9
    assert len(x) * len(y) < 30_000
    assert surface_grid(it) is sf  # cached


def test_surface_only_for_view3d_scenarios():
    from visualization.live import build_payload
    for key, it in SCENARIO_INTEGRANDS.items():
        mc, ms = run_pair(it, make_bounds(it, 0, 1), 2_000, 1, 300)
        p = build_payload(1, it, mc, ms, it.reference(None))
        assert (p["surface"] is not None) == SCENARIOS[key]["view3d"]
        assert p["labels"]["kind"] == SCENARIOS[key]["kind"]
        assert p["bounds"] == [[0.0, 1.0], [0.0, 1.0]] and p["volume"] == 1.0
        lo, hi = p["vrange"]  # one range covering both algorithms' values
        for algo in ("mc", "ms"):
            # (values are sent rounded to 8 digits)
            assert lo - 1e-8 <= min(p[algo]["v"]) and max(p[algo]["v"]) <= hi + 1e-8


# ------------------------------------------------- shared algorithms
@pytest.mark.parametrize("key", list(SCENARIOS))
@pytest.mark.parametrize("n", [1_000, 10_000, 37_777])
def test_every_scenario_runs_with_exact_budget(key, n):
    it = SCENARIO_INTEGRANDS[key]
    mc, ms = run_pair(it, make_bounds(it, 0, 1), n, 3, 300)
    assert mc["evaluations"] == len(mc["points"]) == n
    assert ms["evaluations"] == len(ms["points"]) == len(ms["kinds"]) == n
    assert set(np.unique(ms["kinds"])) <= {0, 1}
    assert math.isfinite(standard_error(mc)) and math.isfinite(standard_error(ms))


@pytest.mark.parametrize("key", GAUSSIANS)
def test_gaussian_results_have_pilots_splits_and_are_reproducible(key):
    it = SCENARIO_INTEGRANDS[key]
    b = make_bounds(it, 0, 1)
    mc1, ms1 = run_pair(it, b, 20_000, 9, 300)
    mc2, ms2 = run_pair(it, b, 20_000, 9, 300)
    assert mc1["estimate"] == mc2["estimate"] and ms1["estimate"] == ms2["estimate"]
    assert np.array_equal(ms1["points"], ms2["points"])
    assert (ms1["kinds"] == 0).sum() > 0 and (ms1["kinds"] == 1).sum() > 0
    assert len(ms1["splits"]) > 0
    # Values are the continuous Gaussian at the sampled points.
    assert np.allclose(ms1["values"], SCENARIOS[key]["func"](ms1["points"]))
    # Estimate = Σ leaf area × mean f of the production samples (pilots excluded).
    total = sum(np.prod([hi - lo for lo, hi in nd["box"]])
                * ms1["values"][nd["eval_start"]:nd["eval_start"] + nd["count"]].mean()
                for nd in ms1["nodes"] if nd["leaf"])
    assert total == pytest.approx(ms1["estimate"], rel=1e-12)
    # MC estimate = area (1) × mean f, not a count of points.
    assert mc1["estimate"] == pytest.approx(float(np.mean(mc1["values"])), rel=1e-12)


def test_standard_error_formulas():
    it = SCENARIO_INTEGRANDS["peak"]
    mc, ms = run_pair(it, make_bounds(it, 0, 1), 10_000, 1, 300)
    assert standard_error(mc) == pytest.approx(
        np.std(mc["values"], ddof=1) / math.sqrt(10_000))
    assert standard_error(ms) == ms["std_error"]


@pytest.mark.parametrize("key", GAUSSIANS)
def test_gaussian_estimates_are_statistically_consistent(key):
    it = SCENARIO_INTEGRANDS[key]
    ref = it.reference(None)
    for seed in range(5):
        mc, ms = run_pair(it, make_bounds(it, 0, 1), 20_000, 100 + seed, 300)
        assert abs(mc["estimate"] - ref) < 5 * standard_error(mc)
        assert abs(ms["estimate"] - ref) < 5 * standard_error(ms)
    abs_err, rel_err = errors(ms["estimate"], ref)
    assert rel_err == pytest.approx(abs_err / ref)


def test_experiments_accept_gaussian_scenarios():
    it = SCENARIO_INTEGRANDS["peak"]
    b = make_bounds(it, 0, 1)
    rows = sweep(it, b, 1, 300, sizes=(1_000, 5_000))
    assert len(rows) == 4 and all(r["std_error"] > 0 for r in rows)
    rep = repeated(it, b, 1, 300, repeats=3, sizes=(2_000,))
    assert {r["algorithm"] for r in rep} == {"Standard Monte Carlo", "MISER"}
    assert all(r["mean_std_error"] > 0 for r in rep)


# ------------------------------------------------- single execution path
def test_all_scenarios_call_the_shared_functions(monkeypatch):
    calls = []
    real_mc, real_ms = miser.run_plain_monte_carlo, miser.run_miser

    def spy_mc(*a, **k):
        calls.append(("mc", k["func"]))
        return real_mc(*a, **k)

    def spy_ms(*a, **k):
        calls.append(("ms", k["func"]))
        return real_ms(*a, **k)

    monkeypatch.setattr(experiments, "run_plain_monte_carlo", spy_mc)
    monkeypatch.setattr(experiments, "run_miser", spy_ms)
    for key, it in SCENARIO_INTEGRANDS.items():
        calls.clear()
        run_pair(it, make_bounds(it, 0, 1), 2_000, 0, 300)
        # Both algorithms receive the very same integrand function.
        assert calls == [("mc", it.func), ("ms", it.func)]


def test_no_duplicate_algorithm_in_the_integrated_app():
    sources = [p for p in ROOT.rglob("*.py")
               if ".venv" not in p.parts
               and "tests" not in p.parts]
    text = {p.name: p.read_text(encoding="utf-8") for p in sources}
    assert not any("miser_monte_carlo" in t or "standard_monte_carlo" in t
                   for t in text.values())
    defs = [n for n, t in text.items() if re.search(r"^def run_miser\(", t, re.M)]
    assert defs == ["miser.py"]
    defs = [n for n, t in text.items()
            if re.search(r"^def run_plain_monte_carlo\(", t, re.M)]
    assert defs == ["miser.py"]


def test_gaussian_payload_for_the_live_player():
    from visualization.live import build_payload
    it = SCENARIO_INTEGRANDS["peak"]
    mc, ms = run_pair(it, make_bounds(it, 0, 1), 10_000, 4, 300)
    p = build_payload(1, it, mc, ms, it.reference(None))
    assert p["binary"] is False  # colour by value, not inside/outside
    assert p["overlay"] == [0.40, 0.40, 2 * PEAK_SIGMA]
    assert p["reference"] == pytest.approx(SCENARIOS["peak"]["exact"]())
    assert p["surface"] is not None and p["labels"]["value"] == "Function value"
    assert np.allclose(p["ms"]["v"], ms["values"], atol=1e-8)
    assert 0 in p["ms"]["kind"] and 1 in p["ms"]["kind"]
    assert len(p["ms"]["splits"]) == len(ms["splits"]) > 0


# ------------------------------------------------- web app
def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def _table(at):
    return dict(zip(at.dataframe[0].value["Metric"],
                    at.dataframe[0].value["MISER"]))


def _finish(at):
    at.session_state["anim_done"] = at.session_state["run"]["run_id"]
    return at.run()


def test_app_runs_peak_and_switches_scenarios_cleanly():
    at = _app().run()
    assert [o for o in at.sidebar.selectbox[0].options] == [
        SCENARIOS[k]["label"] for k in SCENARIOS]
    at.sidebar.selectbox[0].set_value("peak").run()
    assert not at.exception, at.exception
    # The scenario's preset and description are applied.
    assert at.session_state["n_samples"] == DEMO["peak"]["risk"]["n"]
    assert at.session_state["seed"] == DEMO["peak"]["risk"]["seed"]
    assert any("point spread function" in c.value for c in at.sidebar.caption)

    at.sidebar.button[0].click().run()
    run = at.session_state["run"]
    assert run["settings"][0] == "peak"
    assert run["payload"]["binary"] is False and run["payload"]["surface"] is not None
    _finish(at)
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert labels.count("Standard error") == 2
    assert "Inside circle" not in labels and "Inside ratio" not in labels
    table = _table(at)
    assert "Final estimate (integral of f)" in table
    assert "Samples inside circle" not in table
    ref = SCENARIOS["peak"]["exact"]()
    assert float(at.dataframe[0].value["Plain Monte Carlo"][1]) == pytest.approx(ref, abs=1e-6)
    assert float(table["Standard error"]) == pytest.approx(run["ms"]["std_error"], rel=1e-3)

    # Changing the scenario without running keeps the previous run, with a warning.
    at.sidebar.selectbox[0].set_value("circle").run()
    assert at.warning and at.session_state["run"]["run_id"] == run["run_id"]
    assert "Final estimate (integral of f)" in _table(at)

    # Running again replaces the data (no leftovers from the peak run).
    at.sidebar.button[0].click().run()
    new = at.session_state["run"]
    assert new["run_id"] == run["run_id"] + 1 and new["settings"][0] == "circle"
    assert new["payload"]["binary"] is True
    _finish(at)
    assert "Samples inside circle" in _table(at)
    assert "Inside circle" in [m.label for m in at.metric]
    assert len(new["sweep"]) == len(run["sweep"])


def test_app_labels_for_a_general_binary_region():
    at = _app().run()
    at.sidebar.selectbox[0].set_value("joint_default").run()
    at.sidebar.button[0].click().run()
    _finish(at)
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Inside region" in labels and "Inside circle" not in labels
    table = _table(at)
    assert "Samples inside region" in table and "Reference value (πr²)" not in table
    assert "Pilot samples / sampling points" in table
    assert at.session_state["run"]["payload"]["labels"]["one"] == "Inside region"
