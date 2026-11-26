"""
MISER pilot-sampling, split and allocation tests.

Covers: shared / inherited pilot pool, pilots bunched on one side, zero and
extreme standard deviations, (V·σ)^(2/3) allocation, tiny budgets, budget
conservation, reproducibility, unbiasedness and visualization compatibility.
All runs use fixed seeds, so every statistical check is deterministic.
"""

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import miser  # noqa: E402
from experiments import regions_at_depth  # noqa: E402
from miser import (  # noqa: E402
    ALLOC_EXPONENT, MIN_PILOT_PER_SIDE, PILOT_RATIO, UNIT_SQUARE, allocate,
    evaluate_split, run_miser,
)
from scenarios import SCENARIOS  # noqa: E402

BOX = [[0.0, 1.0], [0.0, 1.0]]
PEAK = SCENARIOS["peak"]["func"]
CIRCLE = SCENARIOS["circle"]["func"]


def _inside(points, box):
    ok = np.ones(len(points), dtype=bool)
    for axis, (lo, hi) in enumerate(box):
        ok &= (points[:, axis] >= lo) & (points[:, axis] <= hi)
    return ok


def _check_budget(res, n):
    assert res["evaluations"] == n == len(res["points"]) == len(res["kinds"])


# ------------------------------------------------- 1. normal pilot pool
def test_shared_pool_scores_every_axis_on_the_same_points():
    rng = np.random.default_rng(0)
    pts = rng.random((400, 2))
    vals = PEAK(pts)
    for axis in (0, 1):
        t = evaluate_split(BOX, axis, pts, vals)
        assert t["valid"]
        assert t["pilot_left"] + t["pilot_right"] == 400
        left = pts[:, axis] < 0.5
        assert t["sigma_left"] == pytest.approx(np.std(vals[left], ddof=1))
        assert t["sigma_right"] == pytest.approx(np.std(vals[~left], ddof=1))
        w_l = (0.5 * t["sigma_left"]) ** ALLOC_EXPONENT
        w_r = (0.5 * t["sigma_right"]) ** ALLOC_EXPONENT
        assert t["cost"] == pytest.approx((w_l + w_r) ** 3)


def test_root_pilots_are_ten_percent_and_uniform():
    n = 20_000
    res = run_miser(n, 3, 300, PEAK, UNIT_SQUARE)
    root = res["splits"][0]
    assert root["pilot_new"] == round(PILOT_RATIO * n)
    assert root["pilot_inherited"] == 0 and root["pilot_topup"] == 0
    pilots = res["points"][:root["after_eval"]]
    assert (res["kinds"][:root["after_eval"]] == 0).all()
    # Uniform over the whole square: each quadrant ≈ 1/4 (500 ± 4.5 sd).
    q, _, _ = np.histogram2d(pilots[:, 0], pilots[:, 1], bins=2,
                             range=[[0, 1], [0, 1]])
    assert np.all(np.abs(q - 500) < 90)


def test_pilots_are_inherited_without_loss_or_duplication():
    res = run_miser(50_000, 5, 300, PEAK, UNIT_SQUARE)
    pts, kinds = res["points"], res["kinds"]
    by_id = {nd["id"]: nd for nd in res["nodes"]}
    assert any(nd["pilot_inherited"] > 0 for nd in res["nodes"])
    for nd in res["nodes"]:
        before = np.arange(len(pts)) < nd["eval_at_create"]
        inherited = np.sum(before & (kinds == 0) & _inside(pts, nd["box"]))
        # Every earlier pilot inside this box came from an ancestor's pool.
        assert nd["pilot_inherited"] == inherited
        # Newly drawn pilots of this node lie inside the node's box.
        if not nd["leaf"]:
            start = nd["eval_at_create"]
            new = pts[start:start + nd["pilot_new"] + nd["pilot_topup"]]
            assert _inside(new, nd["box"]).all()
            assert (kinds[start:start + len(new)] == 0).all()
    for ev in res["splits"]:
        nd = by_id[ev["node_id"]]
        assert ev["pilot_pool"] == (nd["pilot_inherited"] + ev["pilot_new"]
                                    + ev["pilot_topup"])
        assert ev["pilot_pool"] >= round(PILOT_RATIO * nd["count"])


def test_fresh_pilots_mode_inherits_nothing():
    n = 30_000
    res = run_miser(n, 5, 300, PEAK, UNIT_SQUARE, reuse_parent_pilots=False)
    _check_budget(res, n)
    assert all(nd["pilot_inherited"] == 0 for nd in res["nodes"])
    for ev in res["splits"]:
        assert ev["pilot_pool"] == ev["pilot_new"] + ev["pilot_topup"]


def test_pilots_are_not_used_in_the_estimate():
    res = run_miser(30_000, 8, 300, PEAK, UNIT_SQUARE)
    prod = res["kinds"] == 1
    total = 0.0
    for nd in res["nodes"]:
        if nd["leaf"]:
            block = slice(nd["eval_start"], nd["eval_start"] + nd["count"])
            assert prod[block].all()
            vol = np.prod([hi - lo for lo, hi in nd["box"]])
            total += vol * res["values"][block].mean()
    assert total == pytest.approx(res["estimate"], rel=1e-12)


# ------------------------------------------------- 2-4. bunched / empty sides
class _BunchedRNG:
    """Real generator whose first ``k`` uniform() calls land in one part."""

    def __init__(self, seed, k, side):
        self._g = np.random.Generator(np.random.PCG64(seed))
        self._k = k
        self._side = side

    def uniform(self, lo, hi, count):
        u = self._g.random(count)
        if self._k > 0:
            self._k -= 1
            u = 0.4 * u if self._side == "left" else 0.6 + 0.4 * u
        return lo + (hi - lo) * u


@pytest.mark.parametrize("side", ["left", "right"])
def test_bunched_pilots_trigger_a_bounded_top_up(monkeypatch, side):
    # The root's 2 uniform() calls (x and y) both fall in one half, so every
    # axis has 0 pilots on the other side.
    monkeypatch.setattr(miser.np.random, "default_rng",
                        lambda seed: _BunchedRNG(seed, 2, side))
    n = 5_000
    res = run_miser(n, 0, 300, PEAK, UNIT_SQUARE)
    _check_budget(res, n)
    root = res["splits"][0]
    assert root["node_id"] == 0
    assert root["pilot_topup"] == MIN_PILOT_PER_SIDE  # only the missing pilots
    start = root["pilot_new"]
    top = res["points"][start:start + root["pilot_topup"]]
    half = root["midpoint"]
    if side == "left":
        assert (top[:, root["axis"]] >= half).all()
    else:
        assert (top[:, root["axis"]] < half).all()
    assert math.isfinite(res["estimate"]) and math.isfinite(res["std_error"])


def test_empty_side_makes_the_axis_invalid():
    pts = np.column_stack([np.linspace(0.05, 0.45, 50), np.linspace(0, 1, 50)])
    vals = PEAK(pts)
    tx = evaluate_split(BOX, 0, pts, vals)
    ty = evaluate_split(BOX, 1, pts, vals)
    assert not tx["valid"] and tx["pilot_right"] == 0
    assert ty["valid"]  # the other axis is still usable


def test_single_point_side_is_invalid():
    pts = np.array([[0.1, 0.1], [0.2, 0.9], [0.3, 0.5], [0.9, 0.5]])
    t = evaluate_split(BOX, 0, pts, np.array([1.0, 2.0, 3.0, 4.0]))
    assert not t["valid"] and t["pilot_right"] == 1


def test_no_budget_for_top_up_falls_back_to_plain_mc(monkeypatch):
    monkeypatch.setattr(miser.np.random, "default_rng",
                        lambda seed: _BunchedRNG(seed, 2, "left"))
    # count = 610 > 2*300 + 8 (split allowed) but 610 - 8 - 2 <= 600.
    n = 610
    res = run_miser(n, 0, 300, PEAK, UNIT_SQUARE, pilot_ratio=0.0)
    _check_budget(res, n)
    root = res["nodes"][0]
    assert root["leaf"] and root["leaf_reason"] == "no_valid_split"
    assert root["pilot_cost"] + root["count"] == n


def test_non_finite_values_do_not_crash_the_split_logic():
    def f(p):
        out = np.ones(len(p))
        out[p[:, 0] < 0.5] = np.nan
        return out
    t = evaluate_split(BOX, 0, np.random.default_rng(0).random((100, 2)),
                       f(np.random.default_rng(0).random((100, 2))))
    assert not t["valid"]
    res = run_miser(2_000, 0, 300, f, UNIT_SQUARE)
    _check_budget(res, 2_000)


# ------------------------------------------------- 5-7. zero / extreme sigma
def test_one_sided_zero_std():
    pts = np.random.default_rng(1).random((200, 2))
    vals = np.where(pts[:, 0] < 0.5, 3.0, pts[:, 1])
    t = evaluate_split(BOX, 0, pts, vals)
    assert t["valid"] and t["sigma_left"] == 0.0 and t["w_left"] == 0.0
    n_left, n_right = allocate(1_000, t["w_left"], t["w_right"],
                               t["vol_left"], t["vol_right"], 300)
    assert (n_left, n_right) == (300, 700)  # minimum guaranteed, rest right


def test_both_std_zero_constant_function():
    res = run_miser(20_000, 4, 300, lambda p: np.full(len(p), 2.5), UNIT_SQUARE)
    _check_budget(res, 20_000)
    assert res["estimate"] == pytest.approx(2.5, rel=1e-12)
    assert res["variance"] == 0.0 and res["std_error"] == 0.0
    for ev in res["splits"]:
        # Volume split when both weights are 0.
        assert abs(ev["n_left"] - ev["n_right"]) <= 1
        # Tie broken towards the longest side.
        widths = [hi - lo for lo, hi in ev["box"]]
        assert widths[ev["axis"]] == max(widths)


@pytest.mark.parametrize("w_left, w_right, expect_left", [
    (1e-12, 1.0, 300), (1.0, 1e-12, 700), (1e300, 1e300, 500),
    (float("nan"), 1.0, 500), (0.0, 0.0, 500),
])
def test_allocation_extremes(w_left, w_right, expect_left):
    n_left, n_right = allocate(1_000, w_left, w_right, 0.5, 0.5, 300)
    assert n_left == expect_left and n_left + n_right == 1_000


def test_two_thirds_allocation_rule():
    # σ_L = 8 σ_R with equal volumes -> ratio 8^(2/3) : 1 = 4 : 1.
    n_left, n_right = allocate(10_000, 4.0, 1.0, 0.5, 0.5, 300)
    assert (n_left, n_right) == (8_000, 2_000)
    w = lambda v, s: (v * s) ** ALLOC_EXPONENT  # noqa: E731
    # Unequal volumes: weights include V_i.
    n_left, _ = allocate(9_000, w(0.25, 2.0), w(0.75, 2.0), 0.25, 0.75, 1)
    assert n_left == round(9_000 * w(0.25, 2) / (w(0.25, 2) + w(0.75, 2)))


def test_split_events_use_the_two_thirds_rule():
    res = run_miser(50_000, 2, 300, PEAK, UNIT_SQUARE)
    for ev in res["splits"]:
        remaining = ev["n_left"] + ev["n_right"]
        assert ev["w_left"] == pytest.approx((0.5 * math.prod(
            hi - lo for lo, hi in ev["box"]) * ev["sigma_left"]) ** (2 / 3))
        expect = round(remaining * ev["w_left"] / (ev["w_left"] + ev["w_right"]))
        assert ev["n_left"] == max(300, min(expect, remaining - 300))
        # The chosen axis has the lowest cost among valid axes.
        costs = [c for c in ev["axis_costs"] if c is not None]
        assert ev["axis_costs"][ev["axis"]] == min(costs)


# ------------------------------------------------- 8. tiny budgets
@pytest.mark.parametrize("n", [1, 2, 5, 50, 600, 608, 609, 1_000])
def test_tiny_budgets(n):
    res = run_miser(n, 0, 300, PEAK, UNIT_SQUARE)
    _check_budget(res, n)
    assert math.isfinite(res["estimate"])


def test_tiny_leaf_size_deep_recursion():
    res = run_miser(3_000, 0, 1, PEAK, UNIT_SQUARE)
    _check_budget(res, 3_000)
    assert max(nd["depth"] for nd in res["nodes"]) == miser.MAX_DEPTH


def test_invalid_budget():
    with pytest.raises(ValueError):
        run_miser(0, 0, 300, PEAK, UNIT_SQUARE)


# ------------------------------------------------- 9. multiple splits
def test_multi_level_tree_tiles_the_square():
    res = run_miser(100_000, 6, 300, PEAK, UNIT_SQUARE)
    assert len(res["splits"]) > 20
    leaves = [nd for nd in res["nodes"] if nd["leaf"]]
    assert sum(math.prod(hi - lo for lo, hi in nd["box"]) for nd in leaves) \
        == pytest.approx(1.0)
    for ev in res["splits"]:
        children = [nd for nd in res["nodes"] if nd["parent"] == ev["node_id"]]
        assert len(children) == 2  # always a bisection
    var = sum(nd["estimate_var"] for nd in leaves)
    assert res["variance"] == pytest.approx(var)


# ------------------------------------------------- 10. reproducibility
def test_same_seed_same_result():
    a = run_miser(20_000, 11, 300, PEAK, UNIT_SQUARE)
    b = run_miser(20_000, 11, 300, PEAK, UNIT_SQUARE)
    c = run_miser(20_000, 12, 300, PEAK, UNIT_SQUARE)
    assert a["estimate"] == b["estimate"] and a["std_error"] == b["std_error"]
    assert np.array_equal(a["points"], b["points"])
    assert a["splits"] == b["splits"]
    assert a["estimate"] != c["estimate"]


# ------------------------------------------------- 11. budget never exceeded
@pytest.mark.parametrize("key", list(SCENARIOS))
@pytest.mark.parametrize("n", [1_000, 7_777, 50_000])
def test_budget_is_exact(key, n):
    for seed in range(3):
        _check_budget(run_miser(n, seed, 300, SCENARIOS[key]["func"],
                                UNIT_SQUARE), n)


# ------------------------------------------------- statistical validity
def test_unbiased_and_standard_error_is_calibrated():
    exact = SCENARIOS["peak"]["exact"]()
    est, se = [], []
    for seed in range(200):
        r = run_miser(5_000, 10_000 + seed, 300, PEAK, UNIT_SQUARE)
        est.append(r["estimate"])
        se.append(r["std_error"])
    est, se = np.array(est), np.array(se)
    # Mean within 4 standard errors of the mean.
    assert abs(est.mean() - exact) < 4 * est.std(ddof=1) / math.sqrt(len(est))
    # Reported SE is close to the observed spread and covers ~95 % at 2 SE.
    assert 0.7 < np.mean(se) / est.std(ddof=1) < 1.3
    assert 0.85 < np.mean(np.abs(est - exact) < 2 * se) < 1.0


# ------------------------------------------------- 12. visualization data
def test_visualization_payload_still_builds():
    from experiments import Integrand
    from visualization import plots
    from visualization.live import build_payload
    f = Integrand("peak", "peak", 2, func=PEAK, scalar=None,
                  reference=lambda b: SCENARIOS["peak"]["exact"](),
                  reference_kind="exact", fixed_bounds=UNIT_SQUARE)
    mc = miser.run_plain_monte_carlo(20_000, 1, PEAK, UNIT_SQUARE)
    ms = run_miser(20_000, 2, 300, PEAK, UNIT_SQUARE)
    p = build_payload(1, f, mc, ms, f.reference(None))
    kind = np.asarray(p["ms"]["kind"])
    covered = np.zeros(len(kind), dtype=bool)
    for start, count, *_ in p["ms"]["leaves"]:
        assert (kind[start:start + count] == 1).all()
        covered[start:start + count] = True
    assert (covered == (kind == 1)).all()
    assert len(p["ms"]["splits"]) == len(ms["splits"])
    for ev in ms["splits"]:
        assert (kind[ev["after_eval"] - ev["pilot_cost"]:ev["after_eval"]] == 0).all()
    depth = max(nd["depth"] for nd in ms["nodes"])
    regions = regions_at_depth(ms["nodes"], depth)
    plots.partition_figure_2d(regions, 1.0, None)
    plots.depth_summary_figure(ms["nodes"])
