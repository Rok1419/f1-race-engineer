"""Streamlit view for hypothetical championship and race strategy."""
import pandas as pd
import plotly.express as px
import streamlit as st

from simulator import RACE_POINTS, championship_projection, simulate_strategy
from tyre_estimator import fetch_race_laps, estimate_tyres


def render_championship_simulator(driver_standings, constructor_standings, season):
    st.subheader("🏆 Championship What-If Simulator")
    st.info("SIMULATION — not official results. Choose a hypothetical finishing order for ONE future Grand Prix. Uses 25–18–15–12–10–8–6–4–2–1 points; no sprint, penalties or fastest-lap point. Existing standings are the starting point.")
    options = driver_standings["driver"].tolist()
    n = min(10, len(options))
    st.caption("Choose a driver for each finishing place. A driver can appear only once; unselected drivers receive zero new points.")
    choices = {}
    first = st.columns(5)
    for i in range(n):
        pos = i + 1
        with first[i % 5]:
            selected = st.selectbox(f"P{pos} (+{RACE_POINTS[i]} pts)", ["— None —"] + options,
                                    index=i + 1, key=f"v4_finish_{season}_{pos}")
        if selected != "— None —":
            if selected in choices:
                st.error(f"{selected} is assigned twice. Each driver may finish only once.")
                return
            choices[selected] = pos
    projected_drivers, projected_teams = championship_projection(
        driver_standings.to_dict("records"), constructor_standings.to_dict("records"), choices
    )
    st.markdown("#### Projected Drivers’ Championship")
    drv = pd.DataFrame(projected_drivers)
    drv["projected_rank"] = range(1, len(drv) + 1)
    st.dataframe(drv[["projected_rank", "driver", "constructor", "points", "simulated_points", "projected_points"]],
                 hide_index=True, width="stretch")
    st.plotly_chart(px.bar(drv.head(12).sort_values("projected_points"), y="driver", x="projected_points",
                           orientation="h", labels={"projected_points": "Projected points", "driver": "Driver"}),
                    width="stretch")
    st.markdown("#### Projected Constructors’ Championship")
    if projected_teams:
        team_df = pd.DataFrame(projected_teams)
        team_df["projected_rank"] = range(1, len(team_df) + 1)
        st.dataframe(team_df[["projected_rank", "constructor", "points", "simulated_points", "projected_points"]],
                     hide_index=True, width="stretch")
        st.plotly_chart(px.bar(team_df.sort_values("projected_points"), x="projected_points", y="constructor",
                               orientation="h"), width="stretch")
    st.caption("Championship ties are displayed in existing order; official countback tie-break rules are not simulated. Team transfers within a season may affect attribution.")


def _scenario(label, key, race_laps, base, pit_loss, safety_start, safety_len):
    st.markdown(f"**{label}**")
    stops = st.slider("Number of pit stops", 0, 3, 1, key=f"{key}_stops")
    pit_laps = []
    for i in range(stops):
        low = 1 if i == 0 else pit_laps[i-1] + 1
        default = min(race_laps - (stops - i), max(low, round(race_laps * (i + 1) / (stops + 1))))
        pit_laps.append(st.number_input(f"Pit after lap {i+1}", min_value=low,
                                        max_value=race_laps - (stops - i), value=default,
                                        step=1, key=f"{key}_pit_{i}"))
    compounds = []
    for i in range(stops + 1):
        recommended = ["Medium", "Hard", "Medium", "Soft"][i]
        compounds.append(st.selectbox(f"Stint {i+1} tyres", ["Soft", "Medium", "Hard"],
                                      index=["Soft", "Medium", "Hard"].index(recommended), key=f"{key}_tyre_{i}"))
    return pit_laps, compounds


def render_race_engineer(driver_standings, race_results, season):
    st.subheader("🎧 Race Engineer Mode — Pit Wall")
    st.info("EXPERIMENTAL SIMULATION — lap times, degradation and safety-car effects use adjustable assumptions, not a validated F1 predictive model. Run different strategies to see hypothetical time differences.")
    driver = st.selectbox("Your driver", driver_standings["driver"].tolist(), key=f"v4_engineer_driver_{season}")
    st.markdown("### 🛞 Auto-Estimate Tyres — experimental")
    st.caption("Uses historical race laps from OpenF1, not an official tyre model. Choose a completed race and driver, then import estimates into the simulator. The selected driver above does not constrain OpenF1 driver codes.")
    race_options = (race_results[["round", "race"]].drop_duplicates().sort_values("round") if not race_results.empty else pd.DataFrame())
    estimate = None
    import_values = st.session_state.get("v41_import_values")
    if not race_options.empty:
        labels = {f'{int(row["round"])} — {row["race"]}': int(row["round"]) for _, row in race_options.iterrows()}
        selected_label = st.selectbox("Completed Grand Prix for tyre estimation", list(labels), index=len(labels)-1, key="v41_est_race")
        round_num = labels[selected_label]
        fuel_correction = st.number_input("Assumed fuel-load effect (seconds per earlier lap)", min_value=0.0, max_value=0.12, value=0.035, step=0.005, format="%.3f", key="v41_fuel_corr")
        @st.cache_data(ttl=86400, show_spinner=False)
        def _load_ff1(year, rnd):
            return fetch_race_laps(year, rnd)
        if st.button("Load real race lap data", key="v41_load"):
            st.session_state["v41_loaded"] = (season, round_num)
        if st.session_state.get("v41_loaded") == (season, round_num):
            try:
                with st.spinner("Reading race laps from OpenF1..."):
                    ff1_laps = _load_ff1(season, round_num)
                if ff1_laps.empty:
                    st.warning("No race lap data available for this Grand Prix.")
                else:
                    driver_codes = sorted(ff1_laps["Driver"].dropna().unique().tolist())
                    code = st.selectbox("Driver code from session", driver_codes, key="v41_driver_code")
                    estimate = estimate_tyres(ff1_laps, code, fuel_correction)
                    if estimate.get("rows"):
                        st.dataframe(pd.DataFrame(estimate["rows"]), hide_index=True, width="stretch")
                        st.caption("Confidence is heuristic; even Moderate is NOT a reliable prediction. Missing compounds cannot be inferred. Fuel, traffic and track evolution can distort degradation.")
                        if st.button("Use these estimates in Race Engineer", key="v41_apply"):
                            st.session_state["v41_import_values"] = estimate
                            st.rerun()
                    else:
                        st.warning("Insufficient clean laps for estimation. Try another driver or race.")
            except ImportError as exc:
                st.error(f"A required dependency is missing: {exc}. Please reinstall requirements if needed.")
            except Exception as exc:
                st.error(f"Could not load race data: {exc}")
    else:
        st.info("No completed races available for estimation.")
    if import_values and import_values.get("estimates"):
        st.success(f"Imported estimates from driver {import_values['driver']}. Missing compounds use manual defaults.")
    fitted = import_values.get("estimates", {}) if import_values else {}
    known = [name for name in ("Soft", "Medium", "Hard") if name in fitted]
    # Preserve a user-editable pace reference while importing real measurements.
    if import_values and known:
        reference = "Medium" if "Medium" in fitted else known[0]
        ref_pace = fitted[reference]["pace_s"]
        # If Medium is missing, avoid pretending its pace is known.
        estimated_offsets = {name: fitted[name]["pace_s"] - ref_pace for name in known}
    else:
        reference, ref_pace, estimated_offsets = "Medium", 90.0, {}
    if import_values and known:
        st.caption(f"Baseline reference: {reference} ({ref_pace:.3f} s, estimated at tyre age 1 after assumed fuel correction). Unmeasured compounds retain illustrative offsets.")
    c1, c2, c3 = st.columns(3)
    with c1:
        race_laps = st.slider("Race distance (laps)", 10, 100, 55, key="v4_laps")
        base = st.number_input("Baseline medium-tyre lap (seconds)", min_value=30.0, max_value=300.0,
                               value=float(ref_pace), step=0.1, key=f"v41_base_{import_values['driver']}_{import_values['race_laps']}" if import_values and known else "v4_base")
    with c2:
        pit_loss = st.slider("Normal pit-stop time loss (seconds)", 0.0, 50.0, 22.0, 0.5, key="v4_pitloss")
        deg_soft = st.slider("Soft degradation (s/lap)", 0.0, 1.0, float(min(1.0, fitted.get("Soft", {}).get("degradation_s_per_lap", 0.12))), 0.005, key=f"v41_degs_{import_values['driver']}_{import_values['race_laps']}" if import_values else "v4_degs")
    with c3:
        deg_medium = st.slider("Medium degradation (s/lap)", 0.0, 1.0, float(min(1.0, fitted.get("Medium", {}).get("degradation_s_per_lap", 0.07))), 0.005, key=f"v41_degm_{import_values['driver']}_{import_values['race_laps']}" if import_values else "v4_degm")
        deg_hard = st.slider("Hard degradation (s/lap)", 0.0, 1.0, float(min(1.0, fitted.get("Hard", {}).get("degradation_s_per_lap", 0.04))), 0.005, key=f"v41_degh_{import_values['driver']}_{import_values['race_laps']}" if import_values else "v4_degh")
    soft_offset = st.number_input("Soft initial pace difference vs reference (s)", value=float(estimated_offsets.get("Soft", -0.7)), step=0.05, key=f"v41_softoffset_{import_values['driver']}_{import_values['race_laps']}" if import_values else "v41_so_default")
    medium_offset = st.number_input("Medium initial pace difference vs reference (s)", value=float(estimated_offsets.get("Medium", 0.0)), step=0.05, key=f"v41_mediumoffset_{import_values['driver']}_{import_values['race_laps']}" if import_values else "v41_me_default")
    hard_offset = st.number_input("Hard initial pace difference vs reference (s)", value=float(estimated_offsets.get("Hard", 0.55)), step=0.05, key=f"v41_hardoffset_{import_values['driver']}_{import_values['race_laps']}" if import_values else "v41_ha_default")
    offsets = {"Soft": soft_offset, "Medium": medium_offset, "Hard": hard_offset}
    model_fuel = import_values.get("fuel_effect", 0.035) if import_values else 0.035
    sc_on = st.checkbox("Simulate a safety car", value=False, key="v4_sc")
    if sc_on:
        c4, c5 = st.columns(2)
        safety_start = c4.slider("Safety-car first lap", 1, race_laps, min(20, race_laps), key="v4_sc_start")
        safety_len = c5.slider("Safety-car duration (laps)", 1, min(10, race_laps), 4, key="v4_sc_len")
    else:
        safety_start, safety_len = 0, 0
    left, right = st.columns(2)
    with left:
        a_pits, a_tyres = _scenario("Strategy A", "v4_a", race_laps, base, pit_loss, safety_start, safety_len)
    with right:
        b_pits, b_tyres = _scenario("Strategy B", "v4_b", race_laps, base, pit_loss, safety_start, safety_len)
    try:
        a = simulate_strategy(race_laps, base, pit_loss, a_tyres, a_pits,
                              deg_soft, deg_medium, deg_hard, safety_start, safety_len,
                              fuel_effect=model_fuel, offsets_override=offsets)
        b = simulate_strategy(race_laps, base, pit_loss, b_tyres, b_pits,
                              deg_soft, deg_medium, deg_hard, safety_start, safety_len,
                              fuel_effect=model_fuel, offsets_override=offsets)
    except ValueError as exc:
        st.error(str(exc))
        return
    delta = b["total_seconds"] - a["total_seconds"]
    winner = "A" if delta > 0.005 else "B" if delta < -0.005 else "Tie"
    m1, m2, m3 = st.columns(3)
    m1.metric("Strategy A time", f"{a['total_seconds']/60:.2f} min")
    m2.metric("Strategy B time", f"{b['total_seconds']/60:.2f} min")
    m3.metric("Faster strategy", winner, f"{abs(delta):.2f} s difference")
    curves = pd.concat([pd.DataFrame(a["laps"]).assign(strategy="A"),
                        pd.DataFrame(b["laps"]).assign(strategy="B")], ignore_index=True)
    st.plotly_chart(px.line(curves, x="lap", y="cumulative_s", color="strategy",
                            labels={"lap":"Race lap", "cumulative_s":"Simulated elapsed race time (s)"},
                            title=f"{driver} — cumulative race-time comparison"), width="stretch")
    st.plotly_chart(px.line(curves, x="lap", y="lap_time_s", color="strategy",
                            labels={"lap":"Race lap", "lap_time_s":"Simulated lap time (s)"}), width="stretch")
    st.dataframe(curves[["strategy", "lap", "compound", "tyre_age", "lap_time_s", "pit_loss_s", "safety_car"]],
                 hide_index=True, width="stretch")
    st.caption("Model assumptions: tyre pace offsets are editable and may come from estimates; missing values are illustrative. Degradation is linear per tyre lap; heavier fuel costs 0.035 s per earlier lap. A safety car adds 18 s to each affected lap and reduces pit loss by 40% if pitting then. The model does not simulate opponents, gaps, traffic, overtakes, weather or real position.")
