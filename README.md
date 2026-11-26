# Monte Carlo vs MISER Integration

Interactive Visualization and Performance Comparison of Monte Carlo Integration
Algorithms — a Streamlit dashboard for a university algorithms team project.

The app runs the project's own **Plain (standard) Monte Carlo** and **MISER**
(recursive stratified sampling) implementations on the same integrand with the
same function-evaluation budget, replays how each one placed its samples, shows
how MISER partitioned the domain, and compares accuracy and runtime.

## 1. The two algorithms

| | Plain Monte Carlo | MISER |
|---|---|---|
| Sampling | Uniform random samples over the whole region | Recursively bisects the region; samples each sub-region separately |
| Adaptation | None | Pilot samples estimate the standard deviation σ of each half; more samples go where f varies most |
| Estimate | Volume × mean of f | Σ sub-region volume × mean of f in that sub-region |
| Standard error | V · s / √N | √(Σ V_leaf² s_leaf² / N_leaf) for the chosen partition |
| Overhead | Minimal | Pilot samples (part of the budget) + recursion |

MISER aims to reduce the estimator's variance. It is **not** guaranteed to be
more accurate or faster for every integrand, sample size, or random run.

### MISER as implemented (`miser.run_miser`)

At each region with budget N function evaluations:

1. **Stop** and integrate the region with plain MC if N ≤ 2·leaf_size + P
   (P = pilot target below) or the depth limit (12) is reached.
2. **Pilot pool** of P = max(8, round(0.10 · N)) points (`PILOT_RATIO`), uniform
   over the region. Pilots inherited from the parent region that lie inside it
   are reused; only the shortfall is evaluated (`REUSE_PARENT_PILOTS`).
3. **Choose the split axis** on that one pool: split at the midpoint of each
   axis, estimate σ of both halves, and take the axis with the smallest cost
   (Σᵢ (Vᵢ·σᵢ)^(2/3))³ (Press & Farrar 1990). Ties (e.g. all σ = 0) go to the
   longest side.
4. **Pilots bunched on one side**: an axis whose halves have fewer than 2 pilots
   is skipped. If no axis can be scored, at most 2 missing pilots per half are
   drawn once in the deficient half (within the budget); otherwise the region
   falls back to plain MC. No retries, the budget is never exceeded.
5. **Allocate** the remaining samples with Nᵢ ∝ (Vᵢ·σᵢ)^(2/3), at least
   leaf_size per half; integer rounding never loses or adds samples.
6. **Recurse** into both halves (left/lower first) and add their estimates and
   variances.

Pilot points only guide the splits; they are never part of the estimate, so it
stays unbiased. Every evaluation (pilot or sampling point) counts toward the
budget, and `run_miser` asserts that exactly N evaluations were made.

Changes compared with the team's original MISER (kept as history in git): one
shared pilot pool per region instead of separate pilots per axis, pilot reuse,
the bounded top-up for bunched pilots, (V·σ)^(2/3) allocation instead of ∝ σ,
and the `variance` / `std_error` result keys. Plain Monte Carlo is unchanged
(a regression test pins its values).

## 2. Scenarios

All scenarios live in `scenarios.py` and integrate over the unit square [0, 1]².

| Scenario (key) | f(x, y) | Kind | Reference value |
|---|---|---|---|
| Circle area (`circle`) | 1 inside the circle at (0.70, 0.30), r = 0.15, else 0 | circle (0/1) | πr² = 0.0706858 |
| Joint default of two firms (`joint_default`) | 1 if both firms default (PD = 2 %, ρ = 0.30, Gaussian copula) | binary region (0/1) | 1-D quadrature, 0.0016644 |
| Interference fringes (`interference`) | 1 + cos(100(x + y)) | continuous | analytic, 1.0000237 |
| Starlight on a sensor (`peak`) | exp(−((x−0.4)² + (y−0.4)²) / (2·0.03²)) | continuous, 3-D view | erf product over [0,1]², 0.0056549 |

Each scenario's `kind` decides the wording in one place (`experiments.LABELS`):
circle → *Inside / Outside circle*, other 0/1 regions → *Inside / Outside
region*, continuous functions → *Function value* (coloured by f, never
inside/outside). `view3d` enables the 3-D view.

## 3. Features

- **Sidebar**: scenario, number of samples (1,000–100,000), random seed, MISER
  leaf threshold (under *Advanced*), and **Run Comparison** — the only control
  that runs the algorithms. Choosing a scenario loads its demo seed/sample size.
- **Replay player** (Plain MC left, MISER right) with play / pause / resume,
  restart, speed 0.5× · 1× · 2× · 4×, a progress bar, and a 2-D / 3-D switch for
  the Gaussian peak.
- **Comparison statistics** after the first replay finishes: final estimate,
  reference, absolute / relative / standard error, algorithm time, total
  samples, pilot samples / sampling points, inside counts (0/1 scenarios),
  MISER splits and depth, error/time bar charts.
- **More analysis** tabs: convergence over sample sizes; error analysis with
  optional repeated experiments (MAE, RMSE, std. deviation, mean reported SE);
  performance (time, accuracy vs time); MISER partition by depth; an algorithm
  overview.

## 4. Using the simulation

1. Pick a scenario, the number of samples and a seed in the sidebar.
2. Click **Run Comparison**. Both algorithms run once (MC with `seed`, MISER
   with `seed + 1`) with the same budget.
3. The replay starts automatically. When it finishes the first time, the
   statistics appear below it.
4. Changing a sidebar setting does not re-run anything: a warning says the
   results are from the previous run until you click **Run Comparison** again.

### Replay

The replay **re-plays the computed results in evaluation order**; it does not
run the algorithms again.

- Samples appear in the exact order each algorithm evaluated them (`points`,
  `values`, `kinds`) and stay on the plot: sampling points as dots, MISER
  **pilot samples as ✕**. Nothing is reordered, subsampled or skipped.
- A MISER split line appears once the pilot samples that decided it have been
  drawn (the `after_eval` index recorded by the algorithm); the region currently
  being sampled is outlined.
- **Timing**: each algorithm's replay length follows its measured `elapsed`
  time; both are scaled by one common factor so the slower one takes about 15 s
  at 1× (the faster one at least 0.05 s). The note above the plots shows the
  measured times, their ratio and the factor. Per-sample timestamps are not
  recorded, so within one algorithm samples are spread evenly over its replay.
- Pause keeps the position; Restart replays from the first sample; speed applies
  to both panels. None of these re-run the algorithms.
- Running estimate chips: MC = area × mean f of the shown samples; MISER =
  Σ finished regions + the current region (its sampling points only), i.e. the
  part of the area sampled so far — not the mean of the shown points. Finished
  panels show the algorithm's own final value.

### 2-D and 3-D views

- **2-D** (all scenarios): Plotly axes plus canvas layers; colour = inside /
  outside for 0/1 scenarios, function value (one colour range shared by both
  panels) for continuous ones. The circle boundary or the peak's 2σ guide is
  drawn as an overlay.
- **3-D** (Gaussian peak): a surface computed from the scenario's own `func`
  (`visualization/surface.py`; 101 uniform grid lines plus a ±4σ window at σ/8,
  157×157 nodes, cached), samples at height f with a shared colour bar, pilot
  samples as x markers, MISER splits on the floor. Rotate / zoom with the mouse,
  *Reset 3D view*, optional camera sync. Switching views keeps the replay
  position.

## 5. Performance comparison

The dashboard reports, per run, the estimate, absolute / relative error,
standard error, function evaluations (equal for both by construction),
algorithm time, and MISER splits and depth. Repeated experiments add MAE, RMSE,
the spread of the estimates and the mean reported SE over independent seeds.

One measurement on the Gaussian peak (`peak`, leaf threshold 300; 100 runs at
1k / 10k, 40 runs at 100k; one Windows laptop):

| N | Plain MC RMSE | MISER RMSE | Plain MC time | MISER time |
|---|---|---|---|---|
| 1,000 | 1.66e-3 | 1.49e-3 | 0.48 ms | 0.26 ms |
| 10,000 | 4.95e-4 | 4.65e-5 | 0.43 ms | 4.45 ms |
| 100,000 | 1.59e-4 | 3.81e-6 | 4.46 ms | 34.4 ms |

Results depend on the scenario, budget and seeds: on `interference` (variation
everywhere) plain MC was as accurate or better, and on `joint_default` MISER
occasionally misses the tiny non-zero region. Times depend on the machine.

## 6. Tech stack

Python, NumPy (algorithms), Streamlit (web app), Plotly (charts; `plotly.min.js`
from the installed package drives the browser-side player), pytest (tests).
Tested with Python 3.15, Streamlit 1.67, NumPy 2.5, Plotly 6.9.

## 7. Project structure

```text
├── app.py                  # Streamlit app (the only entry point)
├── miser.py                # run_plain_monte_carlo, run_miser (+ original village density)
├── scenarios.py            # scenario integrands, exact values, display metadata, demo presets
├── experiments.py          # Integrand registry, labels, run_pair, sweeps, repeated runs, errors / SE
├── visualization/
│   ├── live.py             # replay payload + Streamlit component wrapper
│   ├── surface.py          # 3-D surface grid from the scenario's own f (cached)
│   ├── plots.py            # Plotly figures for statistics and analysis tabs
│   └── live_component/
│       └── index.html      # replay player (2-D canvas + 3-D Plotly), plotly.min.js copied at startup
├── tests/
│   ├── test_algorithms.py
│   ├── test_miser_pilot.py
│   └── test_scenarios_integration.py
├── requirements.txt        # runtime dependencies
└── requirements-dev.txt    # + pytest
```

## 8. Setup and running

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

### Experiments from Python

```python
from experiments import SCENARIO_INTEGRANDS, make_bounds, run_pair, repeated, errors, standard_error

it = SCENARIO_INTEGRANDS["peak"]
bounds = make_bounds(it, 0, 1)
reference = it.reference(bounds)
mc, ms = run_pair(it, bounds, 10_000, 7, 300)          # n, seed, leaf threshold
for name, r in (("Plain MC", mc), ("MISER", ms)):
    print(name, r["estimate"], errors(r["estimate"], reference)[0], standard_error(r), r["evaluations"])
for row in repeated(it, bounds, 7, 300, repeats=10, sizes=(1_000, 10_000)):
    print(row["algorithm"], row["n"], row["rmse"])
```

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

The tests cover the algorithms (budgets, reproducibility, pilot handling,
allocation, unbiasedness, SE calibration), all scenarios and reference values,
the replay payload, and the Streamlit app (via `streamlit.testing`). The
browser-side player (JavaScript) is not covered by these tests.

## 9. Fairness and data notes

- Both algorithms get the same total function-evaluation budget; MISER's pilot
  samples are counted inside it.
- Seeds: Monte Carlo uses `seed`, MISER uses `seed + 1`; they never share samples.
- Every plotted point is a real evaluated sample; partition plots come from the
  regions MISER actually recorded.
- Reference values are analytic (or a fine 1-D quadrature for `joint_default`),
  over the finite unit square.
- Execution time is the `elapsed` value each algorithm measures around its own
  computation with `time.perf_counter()`. It excludes data transfer, rendering
  and the replay. MISER's time also includes one final evaluation of f over all
  its points to return their values.

## 10. Limitations

- MISER's pilot can miss a very small feature (e.g. the `joint_default` corner);
  the region then looks flat, gets few samples, and the reported SE (computed
  for the chosen partition) underestimates the real error. Reusing parent
  pilots (`REUSE_PARENT_PILOTS = True`) is more accurate on smooth problems but
  made such misses somewhat more frequent in our tests; `False` gives fresh
  pilots at every region (as in Numerical Recipes).
- Over the whole recursion, pilots use about 10–40 % of the budget (the
  10 % ratio applies per region).
- The replay reproduces evaluation order, not real-time timing of individual
  samples.
- 3-D view: Plotly re-processes every point of a scatter3d trace on each update
  (≈10 µs per point measured), so for very large runs the 3-D view refreshes in
  batches and single updates can pause the page for 1–4 s near the end of a
  100,000-sample replay. The 2-D view plays 100,000 samples at about 60 fps.
- Times are wall-clock times of a Python implementation and depend on the machine.
