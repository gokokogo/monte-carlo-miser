"""
Monte Carlo / MISER integration algorithms.

이 파일의 알고리즘은 원래 app.py 안에 있던 run_plain_monte_carlo / run_miser를
UI 코드와 분리해서 옮긴 것이다. run_plain_monte_carlo 는 원본 그대로다.

run_miser 는 Press & Farrar (1990) / Numerical Recipes 의 MISER 에 맞춰
다음을 고쳤다 (자세한 근거는 run_miser docstring 참고):
  * 정찰 표본: 노드마다 하나의 정찰 집합을 뽑아 모든 분할 축 평가에 공유하고,
    분할 후에는 각 하위 영역에 속한 정찰 표본을 자식에게 물려준다(누적 재활용).
    정찰 표본은 분할 판단에만 쓰고 적분 추정에는 쓰지 않는다.
  * 정찰 표본이 한쪽 절반에 몰려 어떤 축도 평가할 수 없으면, 모자란 절반에만
    최소 개수를 한 번 추가로 뽑는다(예산 안에서, 재시도 없음).
  * 표본 배분: w_i = (V_i * sigma_i)^(2/3) 에 비례 (분할 축 선택 기준과 같은 식).
  * 분산과 표준오차를 반환한다 ("variance", "std_error").

공통 확장 사항 (모두 하위 호환):
  * ``func`` / ``bounds`` 선택 인자 -> 1D 함수와 [a, b] 구간 지원.
    생략하면 원래의 2D village density를 [0,1]^2 에서 적분한다.
  * 시각화를 위한 기록은 반환 dict 에 *키를 추가* 하는 방식이다:
      - "nodes": 재귀 호출(영역)별 경계, 깊이, 배정 샘플 수, 파일럿 분산 추정치,
                 생성 시점(eval_at_create), 리프 샘플 시작 인덱스(eval_start),
                 정찰 표본 출처(pilot_inherited / pilot_new / pilot_topup)
      - "splits" 항목에 var_left / var_right / n_left / n_right / 축별 비용 등
      - "dim", "bounds", "evaluations", "variance", "std_error"
    기존 키("points", "values", "estimate", "elapsed", "kinds", "splits",
    "evaluations", "nodes")의 의미는 바뀌지 않았다.
"""

import time

import numpy as np

VILLAGE_X = 0.70
VILLAGE_Y = 0.30
VILLAGE_RADIUS = 0.15
SIGMA = 0.055

UNIT_SQUARE = ((0.0, 1.0), (0.0, 1.0))

# MISER settings (single source of truth).
PILOT_RATIO = 0.10        # pilot pool size = 10 % of the node's sample budget
MIN_PILOT_PER_SIDE = 2    # a half needs >= 2 pilot values for a sample std
MAX_DEPTH = 12            # recursion depth limit
ALLOC_EXPONENT = 2.0 / 3.0  # Press & Farrar: N_i ∝ (V_i * sigma_i)^(2/3)
# Pass the parent's pilots that lie in a child down to it (only the shortfall
# is drawn fresh). False = fresh pilots at every node, as in Numerical Recipes.
# Both keep the estimate unbiased; see README for the measured trade-off.
REUSE_PARENT_PILOTS = True


def density(x, y):
    """Smooth house-density model concentrated near the village."""
    d2 = (x - VILLAGE_X) ** 2 + (y - VILLAGE_Y) ** 2
    return 0.001 + np.exp(-d2 / (2.0 * SIGMA**2))


def village_density_points(points):
    """Adapter: evaluate ``density`` on an (n, 2) point array."""
    return density(points[:, 0], points[:, 1])


def reference_integral(grid_size=650):
    """Numerical reference value for the integral over the unit square."""
    axis = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    return float(np.mean(density(xx, yy)))


def _normalize(func, bounds):
    """Default to the original 2D village problem when no function is given."""
    if func is None:
        func = village_density_points
    if bounds is None:
        bounds = UNIT_SQUARE
    bounds = [[float(lo), float(hi)] for lo, hi in bounds]
    for lo, hi in bounds:
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"Invalid integration bounds: {bounds}")
    return func, bounds


def run_plain_monte_carlo(n, seed, func=None, bounds=None):
    """Uniform sampling over the complete integration region.

    func   : callable taking an (n, dim) array and returning n values.
             Default: the original 2D village density.
    bounds : sequence of (low, high) per dimension. Default: unit square.
    """
    func, bounds = _normalize(func, bounds)
    dim = len(bounds)
    lows = np.array([b[0] for b in bounds])
    widths = np.array([b[1] - b[0] for b in bounds])
    volume = float(np.prod(widths))

    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    points = lows + widths * rng.random((n, dim))
    values = func(points)
    estimate = volume * float(np.mean(values))

    elapsed = time.perf_counter() - start
    return {
        "points": points,
        "values": values,
        "estimate": estimate,
        "elapsed": elapsed,
        "evaluations": int(n),
        "dim": dim,
        "bounds": bounds,
    }


# ---------------------------------------------------------------- MISER helpers
def _box_volume(box):
    volume = 1.0
    for lo, hi in box:
        volume *= hi - lo
    return volume


def _halves(box, axis):
    """Split ``box`` at the midpoint of ``axis`` into (left, right, midpoint)."""
    midpoint = (box[axis][0] + box[axis][1]) / 2.0
    left_box = [list(b) for b in box]
    right_box = [list(b) for b in box]
    left_box[axis][1] = midpoint
    right_box[axis][0] = midpoint
    return left_box, right_box, midpoint


def _sample_std(values):
    """Sample standard deviation (ddof=1), or None if it cannot be estimated."""
    if len(values) < MIN_PILOT_PER_SIDE:
        return None
    with np.errstate(all="ignore"):
        s = float(np.std(values, ddof=1))
    return s if np.isfinite(s) else None


def evaluate_split(box, axis, pool_points, pool_values):
    """Score a midpoint split of ``box`` along ``axis`` from the pilot pool.

    The pilot pool is uniform over ``box``, so the pilots falling in each half
    are a uniform sample of that half and give an estimate of its std.

    Split cost (Press & Farrar 1990): with N_i ∝ (V_i·σ_i)^(2/3) the variance
    of the integral estimate is ∝ (Σ_i (V_i·σ_i)^(2/3))^3 / N², so the axis
    with the smallest cost is the one whose split reduces the variance most.

    Returns a dict; ``valid`` is False when either half has too few pilots
    (or a non-finite std), in which case the axis cannot be judged.
    """
    left_box, right_box, midpoint = _halves(box, axis)
    left_mask = pool_points[:, axis] < midpoint
    left_vals = pool_values[left_mask]
    right_vals = pool_values[~left_mask]
    s_left = _sample_std(left_vals)
    s_right = _sample_std(right_vals)
    v_left = _box_volume(left_box)
    v_right = _box_volume(right_box)

    result = {
        "axis": axis,
        "midpoint": midpoint,
        "left_box": left_box,
        "right_box": right_box,
        "pilot_left": int(left_mask.sum()),
        "pilot_right": int(len(pool_values) - left_mask.sum()),
        "vol_left": v_left,
        "vol_right": v_right,
        "valid": s_left is not None and s_right is not None,
    }
    if result["valid"]:
        w_left = (v_left * s_left) ** ALLOC_EXPONENT
        w_right = (v_right * s_right) ** ALLOC_EXPONENT
        result.update(
            sigma_left=s_left, sigma_right=s_right,
            var_left=s_left**2, var_right=s_right**2,
            w_left=w_left, w_right=w_right,
            cost=(w_left + w_right) ** 3,
        )
    return result


def allocate(remaining, w_left, w_right, v_left, v_right, min_each):
    """Split ``remaining`` production samples between the two halves.

    n_left / remaining = w_left / (w_left + w_right), w_i = (V_i·σ_i)^(2/3).
    If both weights are 0 (or not finite) the samples follow the volumes.
    Each half gets at least ``min_each``; n_left + n_right == remaining.
    """
    w_sum = w_left + w_right
    if np.isfinite(w_sum) and w_sum > 0.0:
        n_left = int(round(remaining * w_left / w_sum))
    else:
        n_left = int(round(remaining * v_left / (v_left + v_right)))
    n_left = max(min_each, min(n_left, remaining - min_each))
    return n_left, remaining - n_left


# ---------------------------------------------------------------- MISER
def run_miser(n, seed, leaf_size=300, func=None, bounds=None,
              pilot_ratio=PILOT_RATIO, max_depth=MAX_DEPTH,
              reuse_parent_pilots=REUSE_PARENT_PILOTS):
    """
    MISER recursive stratified Monte Carlo (Press & Farrar 1990).

    At each node with budget N (function evaluations):
      0) Leaf test: if N <= 2*leaf_size + P (P = pilot target below) or the
         depth limit is reached, the node is integrated with plain MC.
      1) Pilot pool of P = max(2·dim·MIN_PILOT_PER_SIDE, round(pilot_ratio·N))
         points, uniform over the node. Pilots inherited from the parent that
         lie in this node are reused; only the shortfall is newly evaluated.
      2) Every axis is scored on the *same* pool (``evaluate_split``).
         Axes whose halves have < MIN_PILOT_PER_SIDE pilots are skipped. If no
         axis can be scored, the missing pilots (at most 2·MIN_PILOT_PER_SIDE)
         are drawn once inside the deficient half of the closest axis; if the
         budget does not allow that, the node becomes a plain-MC leaf.
         Ties (e.g. all σ = 0) go to the longest side, then the lower axis.
      3) The remaining N - (new pilots) samples are allocated with
         w_i = (V_i·σ_i)^(2/3) (``allocate``), at least leaf_size per half.
      4) Both halves recurse, each receiving the pilots that lie inside it
         (only if ``reuse_parent_pilots``; otherwise each node starts empty).

    Pilot points are used only for split decisions, never in the estimate:
    the leaf estimates use fresh production samples, so the estimate stays
    unbiased even though the partition depends on the pilot values.
    Every function evaluation (pilot or production) counts toward ``n``.

    ``variance`` = Σ_leaf V² s² / N_leaf (independent strata), conditional on
    the chosen partition; ``std_error`` = sqrt(variance).
    """
    func, bounds = _normalize(func, bounds)
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    dim = len(bounds)
    min_pilot_total = 2 * dim * MIN_PILOT_PER_SIDE

    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    all_points = []
    all_kinds = []  # 0 = pilot, 1 = final/production
    split_events = []
    nodes = []  # instrumentation only: one record per recursive call
    evaluation_count = 0

    def draw(box, count):
        pts = np.empty((count, dim), dtype=float)
        for axis in range(dim):
            lo, hi = box[axis]
            pts[:, axis] = rng.uniform(lo, hi, count)
        return pts

    def add_points(points, kind):
        nonlocal evaluation_count
        all_points.append(points)
        all_kinds.append(np.full(len(points), kind, dtype=np.int8))
        evaluation_count += len(points)

    def leaf_estimate(box, count, node, reason):
        node["eval_start"] = evaluation_count  # instrumentation
        pts = draw(box, count)
        vals = func(pts)
        add_points(pts, 1)
        volume = _box_volume(box)
        result = volume * float(np.mean(vals))
        s2 = float(np.var(vals, ddof=1)) if count > 1 else float("nan")
        variance = volume**2 * s2 / count
        # Instrumentation (does not affect the estimate). For a leaf "count"
        # is its production samples (= budget unless pilots were already
        # spent in this node before it fell back to plain MC).
        node["count"] = int(count)
        node["leaf"] = True
        node["leaf_reason"] = reason
        node["estimate"] = result
        node["sample_var"] = s2 if count > 1 else 0.0
        node["estimate_var"] = variance
        return result, variance

    def recurse(box, count, depth=0, parent=None, pilot_var=None,
                pool_points=None, pool_values=None):
        if pool_points is None:
            pool_points = np.empty((0, dim))
            pool_values = np.empty(0)

        node = {
            "id": len(nodes),
            "parent": parent,
            "depth": depth,
            "box": [list(b) for b in box],
            "count": int(count),
            "budget": int(count),  # evaluations assigned to this node
            "pilot_var": pilot_var,
            "leaf": False,
            "eval_at_create": evaluation_count,  # instrumentation
            "pilot_inherited": int(len(pool_values)),
            "pilot_new": 0,
            "pilot_topup": 0,
        }
        nodes.append(node)

        pilot_target = max(min_pilot_total, int(round(count * pilot_ratio)))

        # 0) Too small for a reliable split, or too deep: plain MC.
        if depth >= max_depth:
            return leaf_estimate(box, count, node, "max_depth")
        if count <= 2 * leaf_size + pilot_target:
            return leaf_estimate(box, count, node, "budget")

        # 1) Pilot pool = inherited pilots + new uniform pilots in this node.
        n_new = max(0, pilot_target - len(pool_values))
        if n_new:
            new_pts = draw(box, n_new)
            new_vals = func(new_pts)
            add_points(new_pts, 0)
            pool_points = np.vstack([pool_points, new_pts])
            pool_values = np.concatenate([pool_values, new_vals])
        node["pilot_new"] = n_new

        # 2) Score every axis on the same pool.
        tests = [evaluate_split(box, a, pool_points, pool_values)
                 for a in range(dim)]
        valid = [t for t in tests if t["valid"]]

        n_topup = 0
        if not valid:
            # Pilots are bunched on one side for every axis: top up the
            # deficient half of the axis closest to being valid, once.
            t = max(tests, key=lambda t: (min(t["pilot_left"], t["pilot_right"]),
                                          -t["axis"]))
            need_l = max(0, MIN_PILOT_PER_SIDE - t["pilot_left"])
            need_r = max(0, MIN_PILOT_PER_SIDE - t["pilot_right"])
            if count - n_new - (need_l + need_r) <= 2 * leaf_size:
                # No budget for the top-up: integrate the rest with plain MC.
                node["pilot_cost"] = n_new
                return leaf_estimate(box, count - n_new, node, "no_valid_split")
            extra = [draw(t["left_box"], need_l), draw(t["right_box"], need_r)]
            extra_pts = np.vstack(extra)
            extra_vals = func(extra_pts)
            add_points(extra_pts, 0)
            n_topup = need_l + need_r
            pool_points = np.vstack([pool_points, extra_pts])
            pool_values = np.concatenate([pool_values, extra_vals])
            t = evaluate_split(box, t["axis"], pool_points, pool_values)
            if not t["valid"]:  # e.g. f returned NaN / inf
                node["pilot_topup"] = n_topup
                node["pilot_cost"] = n_new + n_topup
                return leaf_estimate(box, count - n_new - n_topup, node,
                                     "no_valid_split")
            valid = [t]
            tests[t["axis"]] = t
        node["pilot_topup"] = n_topup

        best = min(valid, key=lambda t: (t["cost"],
                                         -(box[t["axis"]][1] - box[t["axis"]][0]),
                                         t["axis"]))
        pilot_cost = n_new + n_topup
        remaining = count - pilot_cost

        # 3) Allocation with w_i = (V_i σ_i)^(2/3).
        n_left, n_right = allocate(remaining, best["w_left"], best["w_right"],
                                   best["vol_left"], best["vol_right"],
                                   leaf_size)

        split_events.append({
            "axis": best["axis"],
            "midpoint": best["midpoint"],
            "box": [list(b) for b in box],
            "depth": depth,
            "after_eval": evaluation_count,
            # Instrumentation fields
            "node_id": node["id"],
            "pilot_cost": pilot_cost,
            "pilot_pool": int(len(pool_values)),
            "pilot_inherited": node["pilot_inherited"],
            "pilot_new": n_new,
            "pilot_topup": n_topup,
            "var_left": best["var_left"],
            "var_right": best["var_right"],
            "sigma_left": best["sigma_left"],
            "sigma_right": best["sigma_right"],
            "w_left": best["w_left"],
            "w_right": best["w_right"],
            "axis_costs": [t["cost"] if t["valid"] else None for t in tests],
            "n_left": int(n_left),
            "n_right": int(n_right),
        })
        node["split_axis"] = best["axis"]
        node["split_point"] = best["midpoint"]
        node["pilot_cost"] = pilot_cost

        # 4) Recurse; each half inherits the pilots that lie inside it.
        left_mask = pool_points[:, best["axis"]] < best["midpoint"]
        if reuse_parent_pilots:
            left_pool = (pool_points[left_mask], pool_values[left_mask])
            right_pool = (pool_points[~left_mask], pool_values[~left_mask])
        else:
            left_pool = right_pool = (None, None)
        left_est, left_var = recurse(
            best["left_box"], n_left, depth + 1, node["id"], best["var_left"],
            *left_pool)
        right_est, right_var = recurse(
            best["right_box"], n_right, depth + 1, node["id"], best["var_right"],
            *right_pool)
        return left_est + right_est, left_var + right_var

    initial_box = [list(b) for b in bounds]
    estimate, variance = recurse(initial_box, n)

    points = np.vstack(all_points)
    kinds = np.concatenate(all_kinds)
    values = func(points)

    elapsed = time.perf_counter() - start

    # Defensive assertion: every requested function evaluation is counted.
    assert evaluation_count == n, (evaluation_count, n)

    return {
        "points": points,
        "values": values,
        "kinds": kinds,
        "estimate": float(estimate),
        "variance": float(variance),
        "std_error": float(np.sqrt(variance)),
        "elapsed": elapsed,
        "splits": split_events,
        "evaluations": evaluation_count,
        "nodes": nodes,
        "dim": dim,
        "bounds": bounds,
    }
