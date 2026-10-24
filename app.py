import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from simulator import RACE_POINTS, championship_projection, simulate_strategy
from simulator_ui import render_championship_simulator, render_race_engineer
from future_ui import render_future_predictor

from f1_data import (
    F1DataError,
    get_constructor_standings,
    get_driver_standings,
    get_race_results,
    get_sprint_results,
    get_qualifying_results,
    build_teammate_matchups,
    build_driver_metrics,
    build_points_progression,
    build_constructor_metrics,
    build_constructor_progression,
    get_race_pitstops,
)

st.set_page_config(
    page_title="F1 World Championship Analysis",
    page_icon="🏁",
    layout="wide",
)

# Use Streamlit theme variables instead of forcing dark-only colours.
# This makes all text and panels readable in either theme.
st.markdown("""<style>
:root { --f1-accent: #e10600; }
[data-testid="stMetric"] {
 background: var(--secondary-background-color);
 border: 1px solid var(--secondary-background-color);
 border-left: 3px solid var(--f1-accent);
 padding: 14px; border-radius: 10px;
 color: var(--text-color);
}
[data-testid="stSidebar"] {background: var(--secondary-background-color);}
h1,h2,h3 {letter-spacing: -0.025em; color: var(--text-color);}
.stTabs [data-baseweb="tab"] {font-weight: 650;}
</style>""", unsafe_allow_html=True)
st.title("🏎️ F1 WORLD CHAMPIONSHIP • ANALYTICS")
st.caption("Interactive championship dashboard powered by the Jolpica F1 API.")

current_year = dt.date.today().year
season = st.sidebar.selectbox(
    "Season",
    options=list(range(current_year, 1949, -1)),
    index=0,
)

st.sidebar.markdown(
    """
    **Version 4.2**
    - Championship overview
    - Driver analytics
    - Driver comparison
    - Constructor standings
    - Race explorer
    - Qualifying analysis
    - Teammate battles
    - Manufacturer / constructor insights
    - Pit-stop analysis
    - OpenF1 lap and tyre strategy analysis
    - Championship what-if simulator
    - Race Engineer strategy simulator
    """
)

@st.cache_data(ttl=3600, show_spinner=False)
def load_season_data(selected_season: int):
    drivers = get_driver_standings(selected_season)
    constructors = get_constructor_standings(selected_season)
    races = get_race_results(selected_season)
    sprints = get_sprint_results(selected_season)
    qualifying = get_qualifying_results(selected_season)
    metrics = build_driver_metrics(drivers, races)
    progression = build_points_progression(races, sprints)
    return drivers, constructors, races, sprints, metrics, progression, qualifying


try:
    with st.spinner(f"Loading {season} Formula 1 data..."):
        (
            driver_standings,
            constructor_standings,
            race_results,
            sprint_results,
            driver_metrics,
            points_progression,
            qualifying_results,
        ) = load_season_data(season)
except F1DataError as exc:
    st.error(str(exc))
    st.info(
        "Check your internet connection and try another season. "
        "The dashboard downloads public F1 data when a season is selected."
    )
    st.stop()

if driver_standings.empty:
    st.warning(f"No championship standings are available for {season}.")
    st.stop()

completed_races = int(race_results["round"].nunique()) if not race_results.empty else 0
leader = driver_standings.iloc[0]
leading_team = (
    constructor_standings.iloc[0]["constructor"]
    if not constructor_standings.empty
    else "N/A"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Season", str(season))
m2.metric("Completed races", completed_races)
m3.metric("Championship leader", leader["driver"])
m4.metric("Leading constructor", leading_team)

tabs = st.tabs(
    [
        "Championship",
        "Driver analysis",
        "Driver comparison",
        "Constructors",
        "Race explorer",
        "Qualifying",
        "Teammate battles",
        "Manufacturer analysis",
        "Pit stops",
        "Lap times & tyres",
        "🏆 Championship simulator",
        "🎧 Race Engineer Mode",
        "🔮 Future Race Predictor",
    ]
)

with tabs[0]:
    st.subheader("Driver championship standings")

    chart = px.bar(
        driver_standings.sort_values("points"),
        x="points",
        y="driver",
        orientation="h",
        hover_data=["position", "wins", "constructor"],
        labels={"points": "Championship points", "driver": "Driver"},
    )
    chart.update_layout(yaxis_title=None)
    st.plotly_chart(chart, width="stretch")

    display_cols = ["position", "driver", "constructor", "points", "wins", "nationality"]
    st.dataframe(
        driver_standings[display_cols],
        hide_index=True,
        width="stretch",
    )

    if not points_progression.empty:
        st.subheader("Points progression")
        default_drivers = driver_standings["driver"].head(5).tolist()
        selected = st.multiselect(
            "Drivers to show",
            driver_standings["driver"].tolist(),
            default=default_drivers,
            key="championship_progression",
        )
        progress_view = points_progression[
            points_progression["driver"].isin(selected)
        ]
        fig = px.line(
            progress_view,
            x="round",
            y="cumulative_points",
            color="driver",
            markers=True,
            hover_data=["race", "event_points"],
            labels={
                "round": "Round",
                "cumulative_points": "Recorded event points",
                "driver": "Driver",
            },
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Progression is calculated from recorded race and sprint points. "
            "Official championship totals can differ if penalties or later adjustments apply."
        )

with tabs[1]:
    st.subheader("Driver performance analytics")

    if driver_metrics.empty:
        st.info("Race-level results are not available for this season.")
    else:
        left, right = st.columns([1, 2])

        with left:
            chosen_driver = st.selectbox(
                "Driver",
                driver_metrics["driver"].tolist(),
                key="driver_analysis_select",
            )
            d = driver_metrics.loc[
                driver_metrics["driver"] == chosen_driver
            ].iloc[0]

            a, b = st.columns(2)
            a.metric("Championship rank", int(d["position"]))
            b.metric("Championship points", f'{d["points"]:.1f}')
            a.metric("Wins", int(d["wins"]))
            b.metric("Podiums", int(d["podiums"]))
            a.metric("Average grid", f'{d["avg_grid"]:.2f}')
            b.metric("Average finish", f'{d["avg_finish"]:.2f}')
            a.metric("DNFs", int(d["dnfs"]))
            b.metric("Race-session points", f'{d["race_points"]:.1f}')

        with right:
            driver_races = race_results[
                race_results["driver"] == chosen_driver
            ].sort_values("round")

            if not driver_races.empty:
                finish_fig = px.line(
                    driver_races,
                    x="round",
                    y="finish_position",
                    markers=True,
                    hover_data=["race", "grid", "status", "points"],
                    labels={
                        "round": "Round",
                        "finish_position": "Finish position",
                    },
                )
                finish_fig.update_yaxes(autorange="reversed", dtick=1)
                st.plotly_chart(finish_fig, width="stretch")

        st.subheader("Season performance table")
        metric_cols = [
            "position",
            "driver",
            "constructor",
            "points",
            "wins",
            "podiums",
            "dnfs",
            "avg_grid",
            "avg_finish",
            "race_points",
        ]
        st.dataframe(
            driver_metrics[metric_cols],
            hide_index=True,
            width="stretch",
        )

with tabs[2]:
    st.subheader("Head-to-head driver comparison")

    driver_options = driver_metrics["driver"].tolist()
    if len(driver_options) < 2:
        st.info("At least two drivers are required for comparison.")
    else:
        c1, c2 = st.columns(2)
        driver_a = c1.selectbox(
            "Driver A",
            driver_options,
            index=0,
            key="driver_a",
        )
        driver_b = c2.selectbox(
            "Driver B",
            driver_options,
            index=1 if len(driver_options) > 1 else 0,
            key="driver_b",
        )

        comparison = driver_metrics[
            driver_metrics["driver"].isin([driver_a, driver_b])
        ].copy()

        compare_cols = [
            "driver",
            "position",
            "points",
            "wins",
            "podiums",
            "dnfs",
            "avg_grid",
            "avg_finish",
            "race_points",
        ]
        st.dataframe(
            comparison[compare_cols].set_index("driver").T,
            width="stretch",
        )

        if not points_progression.empty:
            comparison_progress = points_progression[
                points_progression["driver"].isin([driver_a, driver_b])
            ]
            fig = px.line(
                comparison_progress,
                x="round",
                y="cumulative_points",
                color="driver",
                markers=True,
                hover_data=["race", "event_points"],
                labels={
                    "round": "Round",
                    "cumulative_points": "Recorded event points",
                    "driver": "Driver",
                },
            )
            st.plotly_chart(fig, width="stretch")

        race_compare = race_results[
            race_results["driver"].isin([driver_a, driver_b])
        ]
        if not race_compare.empty:
            finish_compare = px.line(
                race_compare,
                x="round",
                y="finish_position",
                color="driver",
                markers=True,
                hover_data=["race", "grid", "status"],
                labels={
                    "round": "Round",
                    "finish_position": "Finish position",
                    "driver": "Driver",
                },
            )
            finish_compare.update_yaxes(autorange="reversed", dtick=1)
            st.plotly_chart(finish_compare, width="stretch")

with tabs[3]:
    st.subheader("Constructor championship")

    if constructor_standings.empty:
        st.info("Constructor standings are not available for this season.")
    else:
        fig = px.bar(
            constructor_standings.sort_values("points"),
            x="points",
            y="constructor",
            orientation="h",
            hover_data=["position", "wins", "nationality"],
            labels={
                "points": "Championship points",
                "constructor": "Constructor",
            },
        )
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            constructor_standings[
                ["position", "constructor", "points", "wins", "nationality"]
            ],
            hide_index=True,
            width="stretch",
        )

with tabs[4]:
    st.subheader("Race explorer")

    if race_results.empty:
        st.info("Race results are not available for this season.")
    else:
        races = (
            race_results[["round", "race"]]
            .drop_duplicates()
            .sort_values("round")
        )
        race_labels = {
            f'{int(row["round"])} — {row["race"]}': int(row["round"])
            for _, row in races.iterrows()
        }

        chosen_race_label = st.selectbox(
            "Race",
            list(race_labels.keys()),
            index=len(race_labels) - 1,
        )
        chosen_round = race_labels[chosen_race_label]

        selected_race = race_results[
            race_results["round"] == chosen_round
        ].sort_values("finish_position")

        race_table = selected_race[
            [
                "finish_position",
                "driver",
                "constructor",
                "grid",
                "points",
                "status",
                "laps",
            ]
        ].rename(
            columns={
                "finish_position": "finish",
            }
        )
        st.dataframe(race_table, hide_index=True, width="stretch")

        positions = selected_race[
            ["driver", "grid", "finish_position"]
        ].melt(
            id_vars="driver",
            value_vars=["grid", "finish_position"],
            var_name="measure",
            value_name="position",
        )
        fig = px.scatter(
            positions,
            x="position",
            y="driver",
            symbol="measure",
            labels={"position": "Position", "driver": "Driver"},
        )
        fig.update_xaxes(autorange="reversed", dtick=1)
        st.plotly_chart(fig, width="stretch")

st.divider()
st.caption(
    "Data source: Jolpica F1. This project is intended for analysis and learning."
)


with tabs[5]:
    st.subheader("Qualifying performance")
    if qualifying_results.empty:
        st.info("No qualifying sessions are available for this season in Jolpica. Coverage varies by era.")
    else:
        qdrivers = sorted(qualifying_results["driver"].unique())
        selected_q = st.multiselect("Drivers", qdrivers, default=qdrivers[:5], key="q_driver")
        q_view = qualifying_results[qualifying_results["driver"].isin(selected_q)]
        fig = px.line(q_view, x="round", y="quali_position", color="driver", markers=True,
                      hover_data=["race", "constructor", "Q1", "Q2", "Q3"],
                      labels={"round":"Round", "quali_position":"Qualifying position"})
        fig.update_yaxes(autorange="reversed", dtick=1)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(q_view[["round", "race", "driver", "constructor", "quali_position", "Q1", "Q2", "Q3"]], hide_index=True, width="stretch")
        st.caption("Qualifying classification, not necessarily Sunday's starting grid: penalties and sprint formats may change the grid.")

        if not race_results.empty:
            st.subheader("Qualifying position versus race finishing position")
            merged = qualifying_results.merge(
                race_results[["round", "driver_id", "finish_position", "grid", "status"]],
                on=["round", "driver_id"], how="inner")
            classified = merged[merged["status"].fillna("").eq("Finished") | merged["status"].fillna("").str.startswith("+")]
            classified = classified[classified["quali_position"].notna() & classified["finish_position"].notna()]
            if not classified.empty:
                fig = px.scatter(classified, x="quali_position", y="finish_position", color="driver",
                                 hover_data=["race", "round", "grid"],
                                 labels={"quali_position":"Qualifying position", "finish_position":"Classified race finish"})
                st.plotly_chart(fig, width="stretch")
            st.caption("Only finishers are included in the scatter chart. Starting grid can differ from qualifying due to penalties.")

with tabs[6]:
    st.subheader("Teammate head-to-head")
    matchup = build_teammate_matchups(race_results, qualifying_results)
    if matchup.empty:
        st.info("No two-driver same-team race pairings were found.")
    else:
        teams = sorted(matchup["constructor"].unique())
        team = st.selectbox("Team", teams, key="teammate_team")
        team_rows = matchup[matchup["constructor"] == team].sort_values("round")
        names = sorted(set(team_rows["driver_a"]).union(team_rows["driver_b"]))
        if len(names) >= 2:
            selected_names = st.multiselect("Select two teammates", names, default=names[:2], max_selections=2)
            if len(selected_names) == 2:
                a, b = selected_names
                pair = team_rows[((team_rows["driver_a"] == a) & (team_rows["driver_b"] == b)) |
                                 ((team_rows["driver_a"] == b) & (team_rows["driver_b"] == a))]
                quali = pair["quali_winner"].value_counts()
                race = pair["race_winner"].value_counts()
                a_points = pair.apply(lambda r: r["a_points"] if r["driver_a"] == a else r["b_points"], axis=1).sum()
                b_points = pair.apply(lambda r: r["a_points"] if r["driver_a"] == b else r["b_points"], axis=1).sum()
                c1,c2,c3 = st.columns(3)
                c1.metric("Shared race weekends", len(pair))
                c2.metric("Qualifying head-to-head", f"{quali.get(a, 0)} – {quali.get(b, 0)}")
                c3.metric("Race head-to-head", f"{race.get(a, 0)} – {race.get(b, 0)}")
                st.write(f"**Race points in shared rounds:** {a}: {a_points:g} | {b}: {b_points:g}")
                st.dataframe(pair[["round", "race", "quali_winner", "race_winner"]], hide_index=True, width="stretch")
                st.caption("Teammates are compared only in rounds when both drove for the same constructor. Race H2H excludes rounds where neither was classified; qualifying H2H excludes missing or tied positions. Sprint results are not included.")
            else:
                st.info("Select two drivers to compare.")
        else:
            st.info("Not enough teammates for a head-to-head comparison.")


# --- V3 advanced views ---
with tabs[7]:
    st.subheader("🏭 Manufacturer / constructor analysis")
    st.caption("F1 awards the championship to constructors (teams). Engine manufacturers are not separately ranked by this data source.")
    constructor_metrics = build_constructor_metrics(constructor_standings, race_results)
    if constructor_metrics.empty:
        st.info("No race results available for team analysis.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(constructor_metrics.sort_values("race_points"), x="race_points", y="constructor", orientation="h", labels={"race_points":"Race points", "constructor":"Team"}), width="stretch")
        with c2:
            st.plotly_chart(px.scatter(constructor_metrics, x="mean_grid", y="mean_finish", size="starts", color="constructor", hover_data=["podiums", "dnfs"], labels={"mean_grid":"Average grid (excluding pit-lane/zero)", "mean_finish":"Average classified result position"}), width="stretch")
        view_columns = [c for c in ["position", "constructor", "points", "race_points", "race_wins", "podiums", "dnfs", "starts", "finish_rate_pct", "mean_grid", "mean_finish"] if c in constructor_metrics.columns]
        st.dataframe(constructor_metrics[view_columns], hide_index=True, width="stretch")
        progress = build_constructor_progression(race_results, sprint_results)
        if not progress.empty:
            options = list(constructor_metrics["constructor"])
            selected = st.multiselect("Teams to compare", options, default=options[:5], key="v3_team_progress")
            st.plotly_chart(px.line(progress[progress["constructor"].isin(selected)], x="round", y="cumulative_points", color="constructor", markers=True, hover_data=["race", "event_points"], labels={"cumulative_points":"Recorded race + sprint points"}), width="stretch")
        teams = sorted(race_results["constructor"].dropna().unique())
        if teams:
            chosen_team = st.selectbox("Team driver contributions", teams, key="v3_team_contributions")
            per_driver = race_results[race_results["constructor"] == chosen_team].groupby("driver", as_index=False)["points"].sum()
            st.plotly_chart(px.bar(per_driver, x="driver", y="points", labels={"points":"Race points only"}), width="stretch")
            st.caption("Contributions show race-session points only. The progression includes sprint points when a matching race-team entry exists. Official totals may reflect penalties or corrections.")

with tabs[8]:
    st.subheader("🔧 Pit-stop analysis")
    if race_results.empty:
        st.info("No completed races available.")
    else:
        rr = race_results[["round", "race"]].drop_duplicates().sort_values("round")
        selection = st.selectbox("Grand Prix", rr["round"].tolist(), format_func=lambda n: f"Round {n} — {rr.loc[rr['round'] == n, 'race'].iloc[0]}", key="v3_pit_round")
        @st.cache_data(ttl=3600, show_spinner=False)
        def load_pits(year, round_num):
            return get_race_pitstops(year, round_num)
        pits_key = f"v3_pits_{season}_{selection}"
        if st.button("Load pit-stop records", key="v3_load_pits") or pits_key in st.session_state:
            try:
                with st.spinner("Downloading pit stops..."):
                    stops = load_pits(season, selection)
                    st.session_state[pits_key] = True
                if stops.empty:
                    st.warning("No pit-stop records for this race. Historical coverage varies.")
                else:
                    names = race_results[race_results["round"] == selection][["driver_id", "driver", "constructor"]].drop_duplicates()
                    stops = stops.merge(names, on="driver_id", how="left")
                    st.dataframe(stops.sort_values(["lap", "driver_id"]), hide_index=True, width="stretch")
                    clean = stops.dropna(subset=["duration_s"])
                    clean = clean[clean["duration_s"].between(1, 60)]
                    if not clean.empty:
                        fig = px.scatter(clean, x="lap", y="duration_s", color="constructor", hover_data=["driver", "stop"], labels={"duration_s":"Recorded stop duration (s)"})
                        st.plotly_chart(fig, width="stretch")
                    st.caption("Pit-stop duration is the value reported by Jolpica, not necessarily total pit-lane time. Outlier filtering applies to the plot only.")
            except F1DataError as exc:
                st.warning(f"Pit-stop data unavailable: {exc}")
        st.info("Press ‘Load pit-stop records’ to download this race's data.")

with tabs[9]:
    st.subheader("⏱️ Lap times, race pace and tyre strategy")
    st.caption("OpenF1 historical data from 2023 onward is loaded only when requested; no F1 login required.")
    if race_results.empty:
        st.info("No completed races available.")
    else:
        available_rounds = race_results[["round", "race"]].drop_duplicates().sort_values("round")
        round_num = st.selectbox("Grand Prix", available_rounds["round"].tolist(), format_func=lambda n: f"Round {n} — {available_rounds.loc[available_rounds['round'] == n, 'race'].iloc[0]}", key="v3_fast_round")
        @st.cache_data(ttl=86400, show_spinner=False)
        def load_fastf1_laps(year: int, rnd: int):
            from tyre_estimator import fetch_race_laps
            return fetch_race_laps(year, rnd)
        fast_key = f"v3_fast_laps_{season}_{round_num}"
        if st.button("Load OpenF1 lap and tyre data", key="v3_fast_load") or fast_key in st.session_state:
            try:
                with st.spinner("Downloading lap data (this may take a while on first use)..."):
                    laps = load_fastf1_laps(season, round_num)
                    st.session_state[fast_key] = True
                if laps.empty:
                    st.warning("No lap data was returned for this race.")
                else:
                    drivers = sorted(laps["Driver"].dropna().unique().tolist())
                    choices = st.multiselect("Drivers to compare", drivers, default=drivers[:4], key=f"v3_lap_drivers_{season}_{round_num}")
                    sub = laps[laps["Driver"].isin(choices)].copy()
                    if "IsAccurate" in sub.columns:
                        sub = sub[sub["IsAccurate"].fillna(False)]
                    sub = sub[sub["lap_time_s"].notna() & sub["lap_time_s"].gt(0)]
                    if not sub.empty:
                        st.plotly_chart(px.line(sub, x="LapNumber", y="lap_time_s", color="Driver", markers=False, hover_data=["Compound", "Stint"], labels={"lap_time_s":"Lap time (seconds)", "LapNumber":"Lap"}), width="stretch")
                        st.plotly_chart(px.box(sub, x="Driver", y="lap_time_s", color="Driver", points=False, labels={"lap_time_s":"Lap time (seconds)"}), width="stretch")
                        st.caption("Lap-time distributions include accurate laps but can still be affected by traffic, safety cars and race circumstances; they are not a pure measure of pace.")
                    if "Stint" in laps.columns and "Compound" in laps.columns:
                        clean_stints = laps.dropna(subset=["Stint", "Compound"]).groupby(["Driver", "Stint", "Compound"], as_index=False).agg(start_lap=("LapNumber", "min"), end_lap=("LapNumber", "max"))
                        clean_stints = clean_stints[clean_stints["Driver"].isin(choices)]
                        if not clean_stints.empty:
                            clean_stints["laps_in_stint"] = clean_stints["end_lap"] - clean_stints["start_lap"] + 1
                            st.subheader("Tyre stints")
                            st.dataframe(clean_stints.sort_values(["Driver", "Stint"]), hide_index=True, width="stretch")
                            st.plotly_chart(px.bar(clean_stints, x="laps_in_stint", y="Driver", color="Compound", orientation="h", hover_data=["start_lap", "end_lap", "Stint"], labels={"laps_in_stint":"Recorded laps in stint"}), width="stretch")
            except ImportError:
                st.error(r"OpenF1 data loading failed; check your network connection")
            except Exception as exc:
                st.warning(f"Could not load this OpenF1 session: {exc}")
        st.info("Press ‘Load OpenF1 lap and tyre data’ to download the selected session.")


with tabs[10]:
    render_championship_simulator(
        driver_standings, constructor_standings, season
    )

with tabs[11]:
    render_race_engineer(
        driver_standings, race_results, season
    )

with tabs[12]:
    render_future_predictor(season, driver_standings)
