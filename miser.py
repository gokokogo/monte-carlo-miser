"""
Monte Carlo / MISER integration algorithms.

이 파일의 알고리즘은 원래 app.py 안에 있던 run_plain_monte_carlo / run_miser를
UI 코드와 분리해서 옮긴 것이다. 계산 로직(파일럿 샘플 수, 분할 비용, 샘플 배분,
재귀 종료 조건, 난수 소비 순서)은 그대로 두었다.

원본 대비 확장 사항 (모두 하위 호환):
  * ``func`` / ``bounds`` 선택 인자 추가 -> 1D 함수와 [a, b] 구간 지원.
    생략하면 원래의 2D village density를 [0,1]^2 에서 적분하며, 결과는
    원본 코드와 비트 단위로 동일하다 (tests/test_algorithms.py 참고).
  * MISER의 축 탐색 루프를 (0, 1) -> range(dim) 으로 일반화하고,
    파일럿 비용을 4*m -> 2*dim*m 으로 일반화했다 (dim=2 이면 동일).
  * 시각화를 위한 기록만 반환 dict 에 *키를 추가* 했다:
      - "nodes": 재귀 호출(영역)별 경계, 깊이, 배정 샘플 수, 파일럿 분산 추정치,
                 생성 시점(eval_at_create), 리프 샘플 시작 인덱스(eval_start)
      - "splits" 항목에 var_left / var_right / n_left / n_right 등 추가 필드
      - "dim", "bounds", "evaluations"(MC 에도 추가)
    기존 키("points", "values", "estimate", "elapsed", "kinds", "splits",
    "evaluations")의 의미는 바뀌지 않았다.
"""

import time

import numpy as np

VILLAGE_X = 0.70
VILLAGE_Y = 0.30
VILLAGE_RADIUS = 0.15
SIGMA = 0.055

UNIT_SQUARE = ((0.0, 1.0), (0.0, 1.0))


def density(x, y):
    """Smooth house-density model concentrated near the village."""
    d2 = (x - VILLAGE_X) ** 2 + (y - VILLAGE_Y) ** 2
    return 0.001 + np.exp(-d2 / (2.0 * SIGMA**2))


def village_density_points(points):
    """Adapter: evaluate ``density`` on an (n, 2) point array."""
    return density(points[:, 0], points[:, 1])


def in_village(points):
    d2 = (
        (points[:, 0] - VILLAGE_X) ** 2
        + (points[:, 1] - VILLAGE_Y) ** 2
    )
    return d2 <= VILLAGE_RADIUS**2


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


def run_miser(n, seed, leaf_size=300, func=None, bounds=None):
    """
    MISER-style recursive stratified Monte Carlo.

    At each internal node:
      1) Pilot samples estimate variance in both halves of every axis.
      2) The axis with the lower estimated split cost is selected.
      3) The remaining samples are allocated according to
         subregion area * standard deviation.
      4) The two subregions are integrated recursively.

    Pilot evaluations count toward the total evaluation budget.
    The independent production samples are used for the integral estimate.
    """
    func, bounds = _normalize(func, bounds)
    dim = len(bounds)

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

    def box_area(box):
        area = 1.0
        for lo, hi in box:
            area *= hi - lo
        return area

    def add_points(points, kind):
        nonlocal evaluation_count
        all_points.append(points)
        all_kinds.append(np.full(len(points), kind, dtype=np.int8))
        evaluation_count += len(points)

    def leaf_estimate(box, count, node):
        node["eval_start"] = evaluation_count  # instrumentation
        pts = draw(box, count)
        vals = func(pts)
        add_points(pts, 1)
        result = box_area(box) * float(np.mean(vals))
        # Instrumentation (does not affect the estimate).
        node["leaf"] = True
        node["estimate"] = result
        node["sample_var"] = float(np.var(vals, ddof=1)) if count > 1 else 0.0
        return result

    def recurse(box, count, depth=0, parent=None, pilot_var=None):
        nonlocal evaluation_count

        node = {
            "id": len(nodes),
            "parent": parent,
            "depth": depth,
            "box": [list(b) for b in box],
            "count": int(count),
            "pilot_var": pilot_var,
            "leaf": False,
            "eval_at_create": evaluation_count,  # instrumentation
        }
        nodes.append(node)

        PILOT_RATIO = 0.10
        pilot_groups = 2 * dim
        min_pilot_per_group = 2
     
        m = max(
            min_pilot_per_group,
            round(count * PILOT_RATIO / pilot_groups)
        )
        
        pilot_cost = pilot_groups * m
     
        # Stop if the region is too small for a reliable split.
        if count <= 2 * leaf_size + pilot_cost or depth >= 12:
            return leaf_estimate(box, count, node)

        axis_tests = []

        for axis in range(dim):
            midpoint = (box[axis][0] + box[axis][1]) / 2.0

            left_box = [list(b) for b in box]
            right_box = [list(b) for b in box]
            left_box[axis][1] = midpoint
            right_box[axis][0] = midpoint

            left_pilot = draw(left_box, m)
            right_pilot = draw(right_box, m)
            add_points(left_pilot, 0)
            add_points(right_pilot, 0)

            left_values = func(left_pilot)
            right_values = func(right_pilot)

            var_left = float(np.var(left_values, ddof=1))
            var_right = float(np.var(right_values, ddof=1))

            # Equal-area halves: compare the variance-based split cost.
            split_cost = (var_left ** (1 / 3) + var_right ** (1 / 3)) ** 3

            axis_tests.append({
                "axis": axis,
                "midpoint": midpoint,
                "left_box": left_box,
                "right_box": right_box,
                "var_left": var_left,
                "var_right": var_right,
                "cost": split_cost,
            })

        best = min(axis_tests, key=lambda item: item["cost"])
        remaining = count - pilot_cost

        # Allocate more production samples to the half with larger
        # estimated standard deviation (the two halves have equal area).
        weight_left = np.sqrt(max(best["var_left"], 0.0))
        weight_right = np.sqrt(max(best["var_right"], 0.0))
        weight_sum = weight_left + weight_right

        if weight_sum <= 1e-15:
            n_left = remaining // 2
        else:
            n_left = int(round(remaining * weight_left / weight_sum))

        n_left = max(leaf_size, min(n_left, remaining - leaf_size))
        n_right = remaining - n_left

        split_events.append({
            "axis": best["axis"],
            "midpoint": best["midpoint"],
            "box": [list(b) for b in box],
            "depth": depth,
            "after_eval": evaluation_count,
            # Instrumentation fields
            "node_id": node["id"],
            "pilot_cost": pilot_cost,
            "var_left": best["var_left"],
            "var_right": best["var_right"],
            "n_left": int(n_left),
            "n_right": int(n_right),
        })
        node["split_axis"] = best["axis"]
        node["split_point"] = best["midpoint"]
        node["pilot_cost"] = pilot_cost

        left_integral = recurse(
            best["left_box"], n_left, depth + 1, node["id"], best["var_left"]
        )
        right_integral = recurse(
            best["right_box"], n_right, depth + 1, node["id"], best["var_right"]
        )
        return left_integral + right_integral

    initial_box = [list(b) for b in bounds]
    estimate = recurse(initial_box, n)

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
        "elapsed": elapsed,
        "splits": split_events,
        "evaluations": evaluation_count,
        "nodes": nodes,
        "dim": dim,
        "bounds": bounds,
    }
