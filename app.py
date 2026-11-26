"""
Monte Carlo vs MISER Integration — Streamlit UI.

UI 전용 파일. 알고리즘은 miser.py, 시나리오 등록/기준값/실험 반복/표시 라벨은
experiments.py (+ scenarios.py), 그래프는 visualization/plots.py, 샘플링 재생기
(재생·일시정지·속도·2D/3D)는 visualization/live.py (+ live_component/index.html).

적분 대상: scenarios.SCENARIOS 의 시나리오 (원 넓이, 이진 영역, 연속 함수/가우시안
봉우리). 모든 시나리오는 run_plain_monte_carlo / run_miser 하나씩으로 실행된다.

실행 흐름: 사이드바 Run Comparison → 두 알고리즘 실행 → 새 run_id 로 재생 자동 시작
→ 재생이 처음 끝나면 컴포넌트가 run_id 를 돌려줌 → 최종 통계 표시. 재생 제어는
브라우저 안에서만 동작하므로 알고리즘을 다시 실행하지 않는다.
"""

import streamlit as st

from experiments import (
    SCENARIO_INTEGRANDS,
    SWEEP_SIZES,
    errors,
    labels_for,
    make_bounds,
    regions_at_depth,
    repeated,
    run_pair,
    seeds_for,
    standard_error,
    sweep,
)
# 원 하나 대신 시나리오들을 쓰기 위해 추가 (시나리오 → Integrand 변환은 experiments 공통)
from scenarios import DEMO as PRESETS
from scenarios import SCENARIOS
from visualization import live, plots

DEMO_KEYS = tuple(SCENARIOS)
DEMO = SCENARIO_INTEGRANDS

st.set_page_config(page_title="Monte Carlo vs MISER", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px; padding: 8px 12px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Monte Carlo vs MISER Integration")
st.caption("Interactive Visualization and Performance Comparison of "
           "Monte Carlo Integration Algorithms")

# ============================================================ sidebar
# 시나리오별 시연용 Seed·샘플 수를 자동으로 넣기 위해 추가
def apply_demo_preset():
    preset = PRESETS[st.session_state["scenario"]]["risk"]
    st.session_state["n_samples"] = preset["n"]
    st.session_state["seed"] = preset["seed"]


if "scenario" not in st.session_state:
    st.session_state["scenario"] = DEMO_KEYS[0]
    apply_demo_preset()

with st.sidebar:
    st.header("Experiment Settings")
    scenario = st.selectbox("Scenario", DEMO_KEYS, key="scenario",
                            on_change=apply_demo_preset,
                            format_func=lambda k: SCENARIOS[k]["label"])
    n_samples = st.number_input(
        "Number of Samples", min_value=1_000, max_value=100_000,
        step=1_000, key="n_samples",
        help="Total function-evaluation budget given to EACH algorithm. "
             "MISER's pilot (variance-estimation) samples are included.",
    )
    seed = st.number_input("Random Seed", min_value=0, max_value=999_999,
                           step=1, key="seed")
    with st.expander("Advanced"):
        leaf_size = st.slider(
            "MISER leaf sample threshold", 100, 800, 300, 50,
            help="A region stops splitting when its budget is at most "
                 "2 × threshold + pilot cost. Smaller values → deeper recursion.",
        )
    run_clicked = st.button("Run Comparison", type="primary", width="stretch")
    INTEGRAND = DEMO[scenario]  # 사이드바에서 고른 시나리오
    BOUNDS = make_bounds(INTEGRAND, 0, 1)
    REFERENCE = INTEGRAND.reference(BOUNDS)
    st.caption(
        f"Target: {SCENARIOS[scenario]['description']} Exact integral = {REFERENCE:.6f}. "
        f"At 1× the slower algorithm replays in about {live.PLAYBACK_SECONDS} s."
    )

settings = (scenario, int(n_samples), int(seed), int(leaf_size))

# ============================================================ run
if run_clicked:
    with st.spinner("Running Plain Monte Carlo and MISER..."):
        mc, ms = run_pair(INTEGRAND, BOUNDS, int(n_samples), int(seed), int(leaf_size))
        sweep_rows = sweep(INTEGRAND, BOUNDS, int(seed), int(leaf_size))
    run_id = st.session_state.get("run_counter", 0) + 1
    st.session_state["run_counter"] = run_id
    st.session_state["run"] = {
        "settings": settings, "mc": mc, "ms": ms, "sweep": sweep_rows,
        "run_id": run_id,
        "payload": live.build_payload(run_id, INTEGRAND, mc, ms, REFERENCE),
    }
    st.session_state.pop("repeated", None)

run = st.session_state.get("run")

if run and run["settings"] != settings:
    st.warning("Sidebar settings changed since the last run. The animation and "
               "statistics still show the previous run — click **Run Comparison** "
               "to apply the new settings.")

# ============================================================ animation (top)
payload = (run["payload"] if run else
           live.blank_payload(INTEGRAND, REFERENCE))
finished_id = live.live_sampling(payload)
if run and finished_id == run["run_id"]:
    st.session_state["anim_done"] = run["run_id"]

if run is None:
    st.info("Choose a scenario and the number of samples in the sidebar, then click "
            "**Run Comparison**. Both algorithms run once; their samples are then "
            "replayed side by side in the order they were evaluated (MISER pilot "
            "samples as ✕), with play / pause / speed controls above the plots.")
    st.stop()

done = st.session_state.get("anim_done") == run["run_id"]

# ============================================================ statistics (bottom)
st.header("Comparison Statistics")
if not done:
    st.info("Replay in progress… the final statistics (computed when you clicked "
            "Run Comparison) appear here as soon as the replay finishes.")
    st.stop()

r_scenario, r_n, r_seed, r_leaf = run["settings"]
# 통계를 실행한 시나리오 기준으로 계산하기 위해 추가
INTEGRAND = DEMO[r_scenario]
BOUNDS = make_bounds(INTEGRAND, 0, 1)
REFERENCE = INTEGRAND.reference(BOUNDS)
mc, ms = run["mc"], run["ms"]
mc_abs, mc_rel = errors(mc["estimate"], REFERENCE)
ms_abs, ms_rel = errors(ms["estimate"], REFERENCE)
mc_in = int(mc["values"].sum())
ms_in = int(ms["values"].sum())
ms_pilot = int((ms["kinds"] == 0).sum())
ms_prod = int((ms["kinds"] == 1).sum())
ms_in_prod = int(ms["values"][ms["kinds"] == 1].sum())
ms_depth = max(nd["depth"] for nd in ms["nodes"])
ms_leaves = sum(nd["leaf"] for nd in ms["nodes"])
mc_se, ms_se = standard_error(mc), standard_error(ms)
# 연속 함수(가우시안 등)는 "원 안의 점 개수"가 아니라 함수값의 적분이므로 라벨을 분기.
# 문구는 시나리오 메타데이터(kind)에서 experiments.labels_for 한 곳으로 결정.
L = labels_for(INTEGRAND)
BINARY = L["kind"] in ("circle", "binary")

# 결과(Absolute error) 하이라이트를 위해 추가
st.markdown(
    """<style>
    .st-key-hlmcabs div[data-testid="stMetric"], .st-key-hlmsabs div[data-testid="stMetric"] {
        background: rgba(245, 158, 11, 0.18); border: 2px solid #f59e0b;
    }
    </style>""",
    unsafe_allow_html=True,
)
c_mc, c_ms = st.columns(2)
with c_mc:
    st.markdown('<h4 style="color:#1f77b4">Plain Monte Carlo</h4>', unsafe_allow_html=True)
    a = st.columns(4)
    a[0].metric("Final estimate", f"{mc['estimate']:.6f}")
    with a[1].container(key="hlmcabs"):
        st.metric("Absolute error", f"{mc_abs:.2e}")
    a[2].metric("Relative error", f"{mc_rel * 100:.3f} %")
    a[3].metric("Execution time", f"{mc['elapsed'] * 1000:.2f} ms")
    b = st.columns(4)
    b[0].metric("Total samples", f"{mc['evaluations']:,}")
    if BINARY:
        b[1].metric(L["one"], f"{mc_in:,}")
        b[2].metric("Inside ratio", f"{100 * mc_in / mc['evaluations']:.2f} %")
        b[3].metric("Pilot samples", "0",
                    help=f"Plain MC has no pilot phase: estimate = {L['how_mc']}.")
    else:
        b[1].metric("Standard error", f"{mc_se:.2e}",
                    help="V · s / √N from the sampled f values.")
        b[2].metric("Mean f (samples)", f"{float(mc['values'].mean()):.4g}",
                    help="Estimate = square area (1) × mean f.")
        b[3].metric("Pilot samples", "0", help="Plain MC has no pilot phase.")
with c_ms:
    st.markdown('<h4 style="color:#ff7f0e">MISER Algorithm</h4>', unsafe_allow_html=True)
    a = st.columns(4)
    a[0].metric("Final estimate", f"{ms['estimate']:.6f}")
    with a[1].container(key="hlmsabs"):
        st.metric("Absolute error", f"{ms_abs:.2e}",
                  delta=f"{ms_abs - mc_abs:+.1e}", delta_color="inverse",
                  help="Delta = MISER − Plain MC (negative is better).")
    a[2].metric("Relative error", f"{ms_rel * 100:.3f} %")
    a[3].metric("Execution time", f"{ms['elapsed'] * 1000:.2f} ms",
                delta=f"{(ms['elapsed'] - mc['elapsed']) * 1000:+.1f} ms",
                delta_color="inverse",
                help="Delta = MISER − Plain MC (negative is faster).")
    b = st.columns(4)
    b[0].metric("Total samples", f"{ms['evaluations']:,}",
                help=f"{ms_pilot:,} pilot samples + {ms_prod:,} sampling points.")
    if BINARY:
        b[1].metric(L["one"], f"{ms_in:,}")
        b[2].metric("Inside ratio", f"{100 * ms_in / ms['evaluations']:.2f} %",
                    help="Not the MISER estimate: MISER samples non-uniformly.")
    else:
        b[1].metric("Standard error", f"{ms_se:.2e}",
                    help="sqrt(Σ V² s² / N) over the leaf regions, for the "
                         "partition MISER chose.")
        b[2].metric("Pilot samples", f"{ms_pilot:,}",
                    help="Used only to choose splits; not in the estimate.")
    b[3].metric("Splits / depth", f"{len(ms['splits'])} / {ms_depth}",
                help="Region splits / maximum recursion depth.")

t1, t2 = st.columns([1.15, 1])
with t1:
    head = [
        {"Metric": L["estimate"],
         "Plain Monte Carlo": f"{mc['estimate']:.6f}", "MISER": f"{ms['estimate']:.6f}"},
        {"Metric": L["reference"],
         "Plain Monte Carlo": f"{REFERENCE:.6f}", "MISER": f"{REFERENCE:.6f}"},
    ]
    tail = [
        {"Metric": f"Samples {L['one'].lower()}",
         "Plain Monte Carlo": f"{mc_in:,}",
         "MISER": f"{ms_in:,} (production {ms_in_prod:,})"},
        {"Metric": "Inside ratio",
         "Plain Monte Carlo": f"{100 * mc_in / mc['evaluations']:.2f} %",
         "MISER": f"{100 * ms_in / ms['evaluations']:.2f} %"},
    ] if BINARY else []
    tail += [
        {"Metric": "How the area is computed" if BINARY else "How the integral is computed",
         "Plain Monte Carlo": L["how_mc"], "MISER": L["how_ms"]},
    ]
    st.dataframe(
        head + [
            {"Metric": "Absolute error",
             "Plain Monte Carlo": f"{mc_abs:.3e}", "MISER": f"{ms_abs:.3e}"},
            {"Metric": "Relative error",
             "Plain Monte Carlo": f"{mc_rel * 100:.3f} %", "MISER": f"{ms_rel * 100:.3f} %"},
            {"Metric": "Standard error",
             "Plain Monte Carlo": f"{mc_se:.3e}", "MISER": f"{ms_se:.3e}"},
            {"Metric": "Execution time (algorithm only)",
             "Plain Monte Carlo": f"{mc['elapsed'] * 1000:.3f} ms",
             "MISER": f"{ms['elapsed'] * 1000:.3f} ms"},
            {"Metric": "Total samples",
             "Plain Monte Carlo": f"{mc['evaluations']:,}",
             "MISER": f"{ms['evaluations']:,} (pilot {ms_pilot:,})"},
            {"Metric": "Pilot samples / sampling points",
             "Plain Monte Carlo": f"0 / {mc['evaluations']:,}",
             "MISER": f"{ms_pilot:,} / {ms_prod:,}"},
        ] + tail + [
            {"Metric": "Region splits / max depth / leaf regions",
             "Plain Monte Carlo": "—",
             "MISER": f"{len(ms['splits'])} / {ms_depth} / {ms_leaves}"},
        ],
        hide_index=True,
    )
with t2:
    st.plotly_chart(plots.final_comparison_figure(mc_abs, ms_abs, mc["elapsed"],
                                                  ms["elapsed"]),
                    config={"displaylogo": False}, key="final-bars")

more_acc = "MISER" if ms_abs < mc_abs else "Plain Monte Carlo"
faster = "MISER" if ms["elapsed"] < mc["elapsed"] else "Plain Monte Carlo"
WHERE = {
    "circle": "around the circle's edge (where the indicator changes)",
    "binary": "around the edge of the f = 1 region (where the indicator changes)",
    "continuous": "where f varies most (for a peak, its flanks)",
}[L["kind"]]
st.caption(
    f"In this run, **{more_acc}** was closer to the reference value and **{faster}** "
    f"computed faster. MISER puts more samples {WHERE}, but it also spends part "
    "of the budget on pilot samples, and a pilot that misses a small feature sees "
    "almost no variance there, so whether this pays off depends on the problem and "
    "budget. One random run cannot decide which method is better — "
    "see *Error analysis → Repeated experiments* for RMSE over many runs. "
    "Execution time is the `elapsed` value each algorithm measures around its own "
    "computation (time.perf_counter); it excludes data transfer, rendering and the "
    "replay. MISER's time also includes one final evaluation of f over all its "
    "points to return their values."
)

# ============================================================ more analysis
st.subheader("More analysis")
rows = run["sweep"]
tab_conv, tab_err, tab_perf, tab_part, tab_about = st.tabs([
    "Convergence", "Error analysis", "Performance", "MISER partition",
    "About the algorithms"])

with tab_conv:
    st.plotly_chart(plots.convergence_figure(rows, REFERENCE),
                    config={"displaylogo": False}, key="conv")
    st.caption("One independent run of each algorithm at every sample size "
               f"({', '.join(f'{n:,}' for n in SWEEP_SIZES)}), same seeds as the "
               "main run. Single runs fluctuate randomly.")

with tab_err:
    st.plotly_chart(plots.error_figure(rows), config={"displaylogo": False}, key="err")
    st.markdown("**Repeated experiments**")
    sweep_time = sum(r["time"] for r in rows)
    ra, rb = st.columns([1, 2])
    with ra:
        repeats = st.select_slider("Repetitions per sample size",
                                   options=[5, 10, 20], value=10)
        st.caption(f"Cost: {repeats} × {len(SWEEP_SIZES)} sizes × 2 algorithms = "
                   f"**{repeats * len(SWEEP_SIZES) * 2} runs**, roughly "
                   f"{repeats * sweep_time:.1f} s of compute.")
        rep_clicked = st.button("Run Repeated Experiments")
    if rep_clicked:
        with st.spinner("Running repeated experiments..."):
            st.session_state["repeated"] = {
                "settings": run["settings"], "repeats": repeats,
                "rows": repeated(INTEGRAND, BOUNDS, r_seed, r_leaf, repeats),
            }
    rep = st.session_state.get("repeated")
    if rep and rep["settings"] == run["settings"]:
        rrows = rep["rows"]
        st.success(f"{rep['repeats']} independent runs per sample size "
                   "(different seeds from the single run).")
        g1, g2 = st.columns(2)
        g1.plotly_chart(plots.repeated_figure(rrows, "rmse", "RMSE vs number of samples",
                                              "RMSE (log)"),
                        config={"displaylogo": False}, key="rmse")
        g2.plotly_chart(plots.repeated_figure(rrows, "std", "Std. deviation of estimates",
                                              "Standard deviation (log)"),
                        config={"displaylogo": False}, key="std")
        st.dataframe(
            [{
                "Algorithm": r["algorithm"], "Samples": r["n"],
                "Mean estimate": f"{r['mean_estimate']:.6f}",
                "Mean Absolute Error": f"{r['mae']:.3e}",
                "RMSE": f"{r['rmse']:.3e}",
                "Std of estimates": f"{r['std']:.3e}",
                "Mean reported SE": f"{r['mean_std_error']:.3e}",
                "Mean time (ms)": f"{r['mean_time'] * 1000:.3f}",
            } for r in sorted(rrows, key=lambda r: (r["n"], r["algorithm"]))],
            hide_index=True,
        )
    else:
        with rb:
            st.info("Optional: compute Mean Absolute Error, RMSE and the standard "
                    "deviation over independent runs.")

with tab_perf:
    rep = st.session_state.get("repeated")
    has_rep = bool(rep and rep["settings"] == run["settings"])
    p1, p2 = st.columns(2)
    with p1:
        fig = (plots.time_figure(rep["rows"], key="mean_time",
                                 title=f"Mean execution time ({rep['repeats']} runs)")
               if has_rep else plots.time_figure(rows))
        st.plotly_chart(fig, config={"displaylogo": False}, key="time")
    with p2:
        fig = (plots.accuracy_time_figure(rep["rows"], "mean_time", "rmse",
                                          "Accuracy vs execution time (repeated)",
                                          "RMSE (log)")
               if has_rep else
               plots.accuracy_time_figure(rows, "time", "abs_error",
                                          "Accuracy vs execution time (single run)",
                                          "Absolute error (log)"))
        st.plotly_chart(fig, config={"displaylogo": False}, key="acc-time")
    st.caption("Further down-left is better (lower error for less time). "
               + ("Based on repeated experiments." if has_rep else
                  "Single runs — run the repeated experiments for RMSE."))

with tab_part:
    nodes = ms["nodes"]
    depth_names = {0: "Initial Region", 1: "First Split", 2: "Recursive Subdivision"}
    labels = [f"Depth {d}: {depth_names.get(d, 'Further Subdivision')}"
              for d in range(ms_depth + 1)]
    if ms_depth == 0:
        st.info("MISER did not split the region in this run (budget too small for "
                "the leaf threshold).")
        depth = 0
    else:
        depth = labels.index(st.select_slider("Recursion depth to display",
                                              options=labels,
                                              value=labels[min(3, ms_depth)]))
    regions = regions_at_depth(nodes, depth)
    max_density = max(nd["count"] / ((nd["box"][0][1] - nd["box"][0][0])
                                     * (nd["box"][1][1] - nd["box"][1][0]))
                      for nd in regions)
    q1, q2 = st.columns([1.6, 1])
    with q1:
        st.plotly_chart(plots.partition_figure_2d(
                            regions, max_density, INTEGRAND.circle,
                            overlay_name=("Circle boundary" if L["kind"] == "circle"
                                          else "Reference guide (2σ)")),
                        config={"displaylogo": False}, key="part")
        st.caption("Each rectangle is an actual MISER recursive call. Shade = samples "
                   "allocated per unit area, relative to the densest region at this "
                   "depth. Hover for the allocated samples and pilot variance.")
    with q2:
        st.plotly_chart(plots.depth_summary_figure(nodes),
                        config={"displaylogo": False}, key="depthsum")

with tab_about:
    ab1, ab2 = st.columns(2)
    with ab1:
        st.markdown(
            """
            **Plain Monte Carlo**
            - Drops points **uniformly** over the whole square.
            - Integral ≈ square area × mean of f over the points
              (for a 0/1 indicator: the fraction of points inside the region).
            - Error shrinks like **1/√N**; no adaptation, minimal overhead.
            """)
    with ab2:
        st.markdown(
            """
            **MISER (recursive stratified sampling)**
            - **Recursively bisects** the square at the midpoint of the axis whose
              split lowers the expected variance most.
            - Spends about **10 % of each region's budget on pilot samples** (✕ in
              the replay), drawn uniformly in that region, to estimate the
              standard deviation σ of each half. Pilots guide the splits only;
              they are not part of the estimate.
            - Gives the halves samples in proportion to **(volume × σ)^(2/3)**, so
              more go where f varies most (a circle's edge, a peak's flanks).
            - Integral = Σ region area × mean f of that region's sampling points.
            - Not guaranteed to win every run or every problem.
            """)
    st.caption("Replay: samples appear in their real evaluation order. The two replay "
               "lengths follow the measured algorithm times, scaled by one common "
               "factor (shown above the plots); per-sample timestamps are not "
               "recorded, so within a run samples are spread evenly in time.")
    mc_seed, miser_seed = seeds_for(r_seed)
    st.caption(f"Run settings: {r_n:,} samples per algorithm, seeds {mc_seed} (MC) / "
               f"{miser_seed} (MISER), leaf threshold {r_leaf}.")
