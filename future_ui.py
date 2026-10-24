"""Future-race predictor tab. Only mock scenarios, no claim of future accuracy."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from future_predictor import season_calendar, prior_circuit_event, trial_strategies
from simulator import simulate_strategy
from tyre_estimator import fetch_race_laps, estimate_tyres

@st.cache_data(ttl=7200, show_spinner=False)
def _calendar(year):
    return season_calendar(year)

@st.cache_data(ttl=86400, show_spinner=False)
def _history(year, circuit):
    return prior_circuit_event(year, circuit)

@st.cache_data(ttl=86400, show_spinner=False)
def _laps(year, round_number):
    return fetch_race_laps(year, round_number)

def _plan(prefix, laps, default_stops):
    stops = st.selectbox("Pit stops", [1, 2, 3], index=default_stops-1, key=f"{prefix}_stops")
    cols = st.columns(stops)
    pit_laps = []
    for j in range(stops):
        minimum = (pit_laps[-1]+1) if pit_laps else 1
        maximum = laps - (stops-j)
        default = max(minimum, min(maximum, round((j+1)*laps/(stops+1))))
        with cols[j]:
            pit_laps.append(st.number_input(f"Pit after lap {j+1}", min_value=minimum,
                                            max_value=maximum, value=default, key=f"{prefix}_pit_{j}"))
    compounds = []
    default_tyres = (["Medium", "Hard"] if stops == 1 else
                     ["Medium", "Hard", "Soft"] if stops == 2 else
                     ["Soft", "Medium", "Hard", "Soft"])
    for i in range(stops+1):
        compounds.append(st.selectbox(f"Stint {i+1}", ["Soft", "Medium", "Hard"],
                        index=["Soft", "Medium", "Hard"].index(default_tyres[i]),
                        key=f"{prefix}_compound_{i}"))
    return pit_laps, compounds

def render_future_predictor(season, driver_standings):
    st.subheader("🔮 Future Race Predictor — Pit Wall Forecast")
    st.warning("EXPERIMENTAL WHAT-IF SIMULATION: these are NOT official predictions. "
               "Tyre performance for future races is unknown. Probability values describe "
               "only the hypothetical uncertainty assumptions chosen below.")
    try:
        schedule = _calendar(season)
    except Exception as exc:
        st.error(str(exc)); return
    if not schedule:
        st.info("No upcoming races found for this selected season. Choose a season with future races.")
        return
    choices = {f'{r["round"]} · {r["name"]} · {r["date"]}': r for r in schedule}
    chosen = choices[st.selectbox("Upcoming Grand Prix", list(choices), key="future_gp")]
    if not driver_standings.empty:
        st.selectbox("Your driver (context only; no driver-specific forecast)",
                     driver_standings["driver"].tolist(), key="future_driver")
    st.caption(f'Circuit: {chosen["circuit"]}. Past event data may be unavailable or use a different car/compound allocation.')

    past = _history(season, chosen["circuit_id"])
    if past:
        st.info(f"Historical reference: {past[2]} {past[0]} (round {past[1]}). Different regulations and conditions can make it a poor predictor.")
        if st.button("Estimate tyres from last available race at this circuit", key="future_estimate"):
            try:
                with st.spinner("Loading historical OpenF1 laps and stints…"):
                    data = _laps(past[0], past[1])
                estimates = []
                for driver_code in sorted(data["Driver"].dropna().unique()):
                    result = estimate_tyres(data, driver_code)
                    for compound, entry in result.get("estimates", {}).items():
                        estimates.append({"compound": compound, **entry})
                if estimates:
                    summary = pd.DataFrame(estimates).groupby("compound", as_index=False).agg(
                        pace_s=("pace_s", "median"),
                        degradation_s_per_lap=("degradation_s_per_lap", "median"),
                        drivers=("pace_s", "count"))
                    st.session_state["future_historical"] = {"circuit": chosen["circuit_id"],
                        "year": past[0], "rows": summary.to_dict("records")}
                else:
                    st.warning("Not enough clean historical laps to estimate tyre performance.")
            except Exception as exc:
                st.warning(f"Historical data unavailable: {exc}. Manual assumptions are still available.")
    else:
        st.info("No earlier race at this circuit found. Enter assumptions manually.")
    record = st.session_state.get("future_historical", {})
    measured = {r["compound"]: r for r in record.get("rows", [])} if record.get("circuit") == chosen["circuit_id"] else {}
    if measured:
        st.caption(f"Historical medians from {record['year']} — not current-season estimates; different cars, track and tyre compounds.")
        st.dataframe(pd.DataFrame(record["rows"]), hide_index=True, width="stretch")

    st.markdown("#### Set your race assumptions")
    c1,c2,c3 = st.columns(3)
    with c1:
        laps = st.slider("Race distance (laps)", 15, 90, 55, key="future_laps")
        baseline = st.number_input("Base lap pace, seconds", min_value=35.0, max_value=180.0,
                                   value=90.0, step=0.1, key="future_base")
    with c2:
        pit_loss = st.slider("Pit-lane time loss (seconds)", 10.,45.,22.,0.5,key="future_pit")
        sc_prob = st.slider("Safety-car chance (assumption)",0.,1.,0.25,.05,key="future_sc")
    with c3:
        pace_sigma = st.slider("Pace uncertainty ± (standard deviation, s)",0.,2.,.35,.05,key="future_pace_unc")
        deg_sigma = st.slider("Degradation uncertainty (s/lap)",0.,.2,.025,.005,key="future_deg_unc")
    ref_compound = "Medium" if "Medium" in measured else (next(iter(measured)) if measured else None)
    reference = measured[ref_compound]["pace_s"] if ref_compound else None
    st.caption("Historical compound offsets are relative to the historical reference; the base lap pace above is still manually chosen. Compounds absent from data use editable illustrative assumptions.")
    options={"Soft": (-0.7, .12), "Medium": (0., .07), "Hard": (.55, .04)}
    cols=st.columns(3)
    offsets={}; deg={}
    for i,compound in enumerate(("Soft","Medium","Hard")):
        value=measured.get(compound)
        suggested_offset = (value["pace_s"]-reference) if value and reference is not None else options[compound][0]
        suggested_deg = value["degradation_s_per_lap"] if value else options[compound][1]
        with cols[i]:
            st.markdown(f"**{compound}** {'(historical)' if value else '(manual default)'}")
            offsets[compound] = st.number_input(f"Pace offset {compound} (s)",
                value=float(round(suggested_offset,3)),step=.05,
                key=f"future_off_{chosen['circuit_id']}_{record.get('year') if value else 'manual'}_{compound}")
            deg[compound]=st.number_input(f"Degradation {compound} (s/lap)",
                min_value=0.,max_value=1.0,value=float(round(max(0,suggested_deg),4)),step=.01,
                key=f"future_deg_{chosen['circuit_id']}_{record.get('year') if value else 'manual'}_{compound}")
    st.markdown("#### Your competing pit-wall plans")
    a,b=st.columns(2)
    with a:
        st.markdown("**Strategy A**")
        plan_a=_plan("future_A",laps,1)
    with b:
        st.markdown("**Strategy B**")
        plan_b=_plan("future_B",laps,2)
    st.markdown("#### 🎲 Monte Carlo: what does it mean?")
    st.write(
        "Instead of predicting just one race, the simulator repeats your two pit-stop plans "
        "under many slightly different conditions. Each scenario uses the **same conditions "
        "for both strategies**, making the comparison fair. The results show how often A "
        "or B would finish the *simulated race* faster — not which driver would win the Grand Prix."
    )
    runs = st.select_slider("Number of simulated races", options=[100, 300, 500, 1000], value=500)
    if st.button("🏁 Compare strategies", type="primary"):
        try:
            result = trial_strategies(
                laps, baseline, pit_loss, {"A": plan_a, "B": plan_b}, offsets, deg,
                n=runs, pace_sigma=pace_sigma, degradation_sigma=deg_sigma,
                sc_probability=sc_prob,
            )
            # Use the same settings for the deterministic lap-by-lap illustration.
            # Monte Carlo uses its own varied scenarios; this plot is not a sampled trial.
            lap_a = simulate_strategy(laps, baseline, pit_loss, plan_a[1], plan_a[0],
                deg["Soft"], deg["Medium"], deg["Hard"], offsets_override=offsets)
            lap_b = simulate_strategy(laps, baseline, pit_loss, plan_b[1], plan_b[0],
                deg["Soft"], deg["Medium"], deg["Hard"], offsets_override=offsets)
            st.session_state["v46_lap_comparison"] = {"A": lap_a["laps"], "B": lap_b["laps"]}
            st.session_state["v44_forecast_result"] = result
            st.session_state["v44_forecast_race"] = f'{season} · {chosen["name"]}'
        except Exception as exc:
            st.error(f"Could not run simulation: {exc}")

    result = st.session_state.get("v44_forecast_result")
    if result:
        st.caption(
            f'Last simulation: {st.session_state.get("v44_forecast_race", "unknown race")}. '
            "If you change tyre settings or strategies, click **Compare strategies** again to update the charts."
        )
        samples = pd.DataFrame(result["samples"])
        diff = samples["B_minus_A_s"]  # IMPORTANT: positive => A has smaller elapsed time
        avg = float(diff.mean())
        med = float(diff.median())
        winner = "Strategy A" if med > 0 else ("Strategy B" if med < 0 else "Neither strategy")
        difference = abs(med)
        st.markdown("### 🏎️ Lap time + pit-stop loss by lap")
        st.caption(
            "Lines show predicted driving lap time including tyre wear and fuel effect. "
            "Vertical dashed spikes show the **extra time lost in the pits** on that lap. "
            "The top of each spike is that lap's total elapsed time (driving + pit loss). "
            "This example uses your chosen assumptions, **without a safety car**; "
            "Monte Carlo results below include varied conditions."
        )
        lap_data = st.session_state.get("v46_lap_comparison")
        if lap_data:
            fig_laps = go.Figure()
            for name, color in (("A", "#16836B"), ("B", "#D94848")):
                frame = pd.DataFrame(lap_data[name])
                fig_laps.add_trace(go.Scatter(
                    x=frame["lap"], y=frame["lap_time_s"], mode="lines",
                    name=f"Strategy {name} · driving lap",
                    line=dict(color=color, width=2.5),
                    customdata=frame[["compound", "tyre_age"]].to_numpy(),
                    hovertemplate=("Lap %{x}<br>Driving: %{y:.2f} s<br>"
                                   "Tyre: %{customdata[0]} (age %{customdata[1]})<extra></extra>"),
                ))
                stops = frame[frame["pit_loss_s"] > 0]
                if not stops.empty:
                    # Each spike connects the normal lap-time to lap time plus pit loss.
                    spike_x, spike_y = [], []
                    for r in stops.itertuples():
                        spike_x.extend([r.lap, r.lap, None])
                        spike_y.extend([r.lap_time_s, r.lap_time_s + r.pit_loss_s, None])
                    fig_laps.add_trace(go.Scatter(
                        x=spike_x, y=spike_y, mode="lines",
                        name=f"Strategy {name} · pit loss",
                        line=dict(color=color, width=3, dash="dash"),
                        hoverinfo="skip",
                    ))
                    fig_laps.add_trace(go.Scatter(
                        x=stops["lap"],
                        y=stops["lap_time_s"] + stops["pit_loss_s"],
                        mode="markers", showlegend=False,
                        marker=dict(symbol="diamond", size=10, color=color),
                        customdata=stops[["lap_time_s", "pit_loss_s"]].to_numpy(),
                        hovertemplate=("Pit on lap %{x}<br>Driving: %{customdata[0]:.2f} s"
                                       "<br>Pit loss: +%{customdata[1]:.2f} s"
                                       "<br>Total lap: %{y:.2f} s<extra></extra>"),
                    ))
            fig_laps.update_layout(
                xaxis_title="Race lap", yaxis_title="Lap time incl. pit loss (seconds)",
                hovermode="closest", margin=dict(t=15, b=30),
                legend=dict(orientation="h", y=1.12, x=0),
            )
            fig_laps.update_xaxes(dtick=max(1, laps // 20), range=[1, laps])
            st.plotly_chart(fig_laps, width="stretch", key="v46_lap_pit_chart")
            total_a = sum(r["lap_time_s"] + r["pit_loss_s"] for r in lap_data["A"])
            total_b = sum(r["lap_time_s"] + r["pit_loss_s"] for r in lap_data["B"])
            lap_delta = total_b - total_a
            st.caption(
                f"For these fixed assumptions: A = {total_a:.1f} s, B = {total_b:.1f} s. "
                + (f"A is {lap_delta:.1f} s faster. " if lap_delta > 0 else
                   f"B is {-lap_delta:.1f} s faster. " if lap_delta < 0 else "Strategies tie. ")
                + "A pit stop is modelled after the specified lap; a fresh tyre starts the next lap."
            )
        else:
            st.info("Click Compare strategies again to generate the lap-by-lap chart.")
        st.markdown("### 🏆 Results at a glance")
        if med != 0:
            st.success(f"**{winner} is typically {difference:.2f} seconds faster** in these simulations (median difference).")
        else:
            st.info("The median simulated difference is zero seconds. The strategies are approximately tied.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Strategy A faster", f'{result["a_win_pct"]:.1f}% of scenarios')
        m2.metric("Strategy B faster", f'{result["b_win_pct"]:.1f}% of scenarios')
        m3.metric("Average B − A", f"{avg:+.2f} s")
        st.caption(
            "**How to read B − A:** +10 s means A finishes 10 seconds sooner; "
            "−10 s means B finishes 10 seconds sooner. "
            "A result of 0 s is a tie."
        )
        st.markdown("#### Time difference between strategies")
        st.write(
            "Each bar counts simulated races with a similar time difference. "
            "**Left of zero = B faster**; **right of zero = A faster**. "
            "The dashed vertical line marks an exact tie."
        )
        fig = go.Figure()
        neg = diff[diff < 0]
        pos = diff[diff > 0]
        fig.add_trace(go.Histogram(x=neg, name="B faster", marker_color="#D94848", opacity=0.85,
                                   xbins=dict(size=2)))
        fig.add_trace(go.Histogram(x=pos, name="A faster", marker_color="#16836B", opacity=0.85,
                                   xbins=dict(size=2)))
        fig.update_layout(
            barmode="overlay", bargap=0.04,
            xaxis_title="B race time − A race time (seconds)",
            yaxis_title="Number of simulated races",
            legend_title_text="Faster strategy", margin=dict(t=25, b=50),
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=2)
        st.plotly_chart(fig, width="stretch", key="forecast_diff_hist")

        low = result["difference_p10_s"]
        high = result["difference_p90_s"]
        st.write(
            f"**Middle 80% of simulated differences (10th–90th percentile):** "
            f"{low:+.2f} to {high:+.2f} seconds (B − A). "
            "A range crossing zero means either strategy could be quicker depending on the assumed conditions."
        )
        st.markdown("#### Side-by-side race times")
        times = samples[["simulation", "A", "B"]].melt(
            id_vars="simulation", var_name="Strategy", value_name="Simulated race time (minutes)"
        )
        fig2 = px.box(
            times, x="Strategy", y="Simulated race time (minutes)", color="Strategy",
            points=False, color_discrete_map={"A": "#16836B", "B": "#D94848"},
            title="Typical race time and variability for each strategy",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, width="stretch", key="forecast_box")
        with st.expander("Show first 20 simulated races"):
            st.dataframe(samples.head(20).rename(columns={
                "simulation": "Scenario", "A": "Strategy A (min)",
                "B": "Strategy B (min)", "B_minus_A_s": "B − A (seconds)",
            }), hide_index=True, width="stretch")
        st.download_button("Download all simulated races (CSV)", samples.to_csv(index=False),
                           file_name=f'f1_strategy_{season}_{chosen["round"]}.csv', mime="text/csv")
        st.caption(
            "These are **what-if comparisons** based on your input assumptions, not verified "
            "F1 forecasts or genuine race-win probabilities. The model simplifies traffic, pit stops and safety cars."
        )
    st.caption("Both strategies face the same random pace, degradation, pit-loss and safety-car scenarios. "
       "This simplified model does not include traffic, rival strategies, weather, or real race positions. "
       "Historical tyre data can be unavailable or incomparable across seasons.")
