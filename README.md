# Monte Carlo vs MISER Integration

Interactive Visualization and Performance Comparison of Monte Carlo Integration
Algorithms — a Streamlit dashboard for a university algorithms team project.

It runs the project's own **Standard Monte Carlo** and **MISER** (recursive
stratified sampling) implementations under the same sample budget and visualizes
how each one places samples, how MISER partitions the domain, and how accuracy and
runtime compare.

## The two algorithms

| | Standard Monte Carlo | MISER |
|---|---|---|
| Sampling | Uniform random samples over the whole region | Recursively bisects the region; samples each sub-region separately |
| Adaptation | None | Pilot samples estimate the variance of each half; more samples go to the higher-variance half |
| Estimate | Volume × mean of f | Sum of the sub-region estimates |
| Overhead | Minimal | Pilot samples (part of the budget) + recursion |

MISER aims to reduce the estimator's variance. It is **not** guaranteed to be
more accurate or faster for every integrand, sample size, or random run.

## Project structure

```text
├── app.py                  # Streamlit UI only
├── miser.py                # Monte Carlo + MISER algorithms (+ original village density)
├── experiments.py          # Integrands, reference values, sweeps, repeated runs
├── visualization/
│   ├── __init__.py
│   ├── plots.py            # Plotly figure builders
│   ├── live.py             # live replay component (Python side)
│   └── live_component/     # live replay player (HTML/JS)
├── tests/test_algorithms.py
├── requirements.txt
└── requirements-dev.txt    # adds pytest
```

## Features

The app estimates the **area of a circle** with both algorithms. The circle is
the one from the original demo: centre (0.70, 0.30), radius 0.15, inside the
unit square. The integrand is its indicator,
`f(x, y) = 1 if (x − 0.70)² + (y − 0.30)² ≤ 0.15² else 0` (the original
`in_village` test), so the integral is the circle's area, πr² ≈ 0.070686.

1. **Sidebar**: number of samples (1,000–100,000), random seed, the MISER leaf
   threshold (under *Advanced*), and **Run Comparison**. This button is the only
   run control.
2. **Sampling animation** (top of the page, Plain Monte Carlo left and MISER
   right). It starts automatically when *Run Comparison* is clicked, and both
   panels start together and advance by the same number of samples. Points
   appear in the exact order each algorithm evaluated them and stay on the
   plot. Points inside the circle are coloured blue (MC) or orange (MISER), and
   points outside are gray. Inside/outside comes from the algorithm's own f
   value. MISER's split lines appear, thin and translucent, once the pilot
   samples that decided them have been evaluated, and the region being sampled
   is outlined. Each panel shows samples, progress, inside and outside counts,
   inside ratio and the current estimate, plus the current depth for MISER.
   The whole run replays in about 15 s.
   - Plain MC estimate = inside / total × 1.
   - MISER estimate = Σ region area × inside-fraction of that region. It is
     **not** the overall inside ratio. While playing, it shows the part of the
     area already sampled.
3. **Comparison statistics** (below the animation) appear when the animation
   finishes. They include final estimate, absolute and relative error,
   algorithm execution time, total samples, samples inside the circle, inside
   ratio, MISER splits and maximum depth, a comparison table, and
   error/time bar charts.
4. **More analysis** tabs: convergence; error analysis with optional
   **repeated experiments** (MAE, RMSE, standard deviation); performance
   (time, accuracy vs time); the MISER partition by depth; and an algorithm
   overview.

How the player works: it runs in the browser
(`visualization/live_component/index.html`). Plotly draws the axes, and only
the new points are painted on canvas layers each frame, so 100,000 samples play
at about 60 fps and the page stays responsive. A new run restarts the player.
Other reruns, such as changing a setting, leave it running. When the playback
ends, the player reports its run id back to Streamlit, and that is what reveals
the statistics. `plotly.min.js` is copied from the installed `plotly` package
at startup, so it works offline.

Colors are consistent across all charts: Monte Carlo is blue, MISER is orange,
and the reference is gray.

### Fairness and data notes

- Both algorithms get the same total function-evaluation budget. MISER's pilot
  samples are counted inside it, and the implementation asserts this.
- Seeds: Monte Carlo uses `seed` and MISER uses `seed + 1` (the original project
  convention). The two algorithms never share samples.
- Every plotted point is a real evaluated sample, and all of them are drawn.
- Partition plots come from the regions MISER actually recorded while it ran.
- Reference values are analytic for every integrand. The village integral
  uses the erf closed form of the Gaussian bump.
- Execution time is measured inside each algorithm with `time.perf_counter()`
  and does not include rendering. The original MISER code evaluates f once more
  over all points after integrating, to fill the returned `values`. That pass
  is part of its measured time.

### Changes to the original algorithm code

The algorithms were moved from `app.py` to `miser.py` and their calculation
logic was not changed. Two backward-compatible generalizations were added:

- Optional `func` / `bounds` arguments support 1D functions and arbitrary
  intervals. Without them, results are bit-identical to the original 2D village
  version, and a regression test checks this.
- MISER's axis loop goes over `range(dim)`, and its pilot cost is `2·dim·m`.
  For `dim = 2` both are the same as the original.
- New instrumentation keys (`nodes`, extra split fields) were added to the
  returned dicts. Existing keys keep their meaning.

## Local setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

