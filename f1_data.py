from __future__ import annotations

from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
TIMEOUT = 20

class F1DataError(RuntimeError):
    """Raised when F1 data cannot be downloaded or parsed."""


def _get_json(path: str, params: dict[str, Any] | None = None) -> dict:
    url = f"{BASE_URL}/{path.lstrip('/')}"
    try:
        response = requests.get(
            url,
            params=params,
            timeout=TIMEOUT,
            headers={"User-Agent": "f1-world-championship-analysis/1.0"},
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise F1DataError(
            f"Could not load data from Jolpica F1: {exc}"
        ) from exc



def _get_all_races(season: int, endpoint: str) -> list[dict]:
    """Fetch all driver-result rows, not just the API's first 100.

    Jolpica counts individual driver results for pagination, and may split
    a single Grand Prix across two consecutive pages.
    """
    page_size = 100
    offset = 0
    races = []

    while True:
        data = _get_json(
            f"{season}/{endpoint}.json",
            params={"limit": page_size, "offset": offset},
        )
        metadata = data.get("MRData", {})
        page = metadata.get("RaceTable", {}).get("Races", [])
        races.extend(page)

        try:
            total = int(metadata.get("total", 0))
            actual_limit = int(metadata.get("limit", page_size))
        except (TypeError, ValueError) as exc:
            raise F1DataError("Invalid pagination response from Jolpica F1") from exc

        if actual_limit <= 0:
            raise F1DataError("Invalid page size returned by Jolpica F1")

        offset += actual_limit
        if offset >= total:
            break
        if not page:
            raise F1DataError("Jolpica F1 returned an empty page before all results were downloaded")

    return races


def _to_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_driver_standings(season: int) -> pd.DataFrame:
    data = _get_json(f"{season}/driverstandings.json")
    lists = (
        data.get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )
    if not lists:
        return pd.DataFrame()

    rows = []
    for item in lists[0].get("DriverStandings", []):
        driver = item.get("Driver", {})
        constructors = item.get("Constructors", [])
        rows.append(
            {
                "position": _to_int(item.get("position")),
                "driver_id": driver.get("driverId"),
                "driver": " ".join(
                    part
                    for part in [
                        driver.get("givenName", ""),
                        driver.get("familyName", ""),
                    ]
                    if part
                ),
                "nationality": driver.get("nationality", ""),
                "constructor": ", ".join(
                    c.get("name", "") for c in constructors if c.get("name")
                ),
                "points": _to_float(item.get("points")),
                "wins": _to_int(item.get("wins"), 0),
            }
        )

    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)


def get_constructor_standings(season: int) -> pd.DataFrame:
    data = _get_json(f"{season}/constructorstandings.json")
    lists = (
        data.get("MRData", {})
        .get("StandingsTable", {})
        .get("StandingsLists", [])
    )
    if not lists:
        return pd.DataFrame()

    rows = []
    for item in lists[0].get("ConstructorStandings", []):
        constructor = item.get("Constructor", {})
        rows.append(
            {
                "position": _to_int(item.get("position")),
                "constructor_id": constructor.get("constructorId"),
                "constructor": constructor.get("name", ""),
                "nationality": constructor.get("nationality", ""),
                "points": _to_float(item.get("points")),
                "wins": _to_int(item.get("wins"), 0),
            }
        )

    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)


def get_race_results(season: int) -> pd.DataFrame:
    races = _get_all_races(season, "results")

    rows = []
    for race in races:
        for result in race.get("Results", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            rows.append(
                {
                    "round": _to_int(race.get("round")),
                    "race": race.get("raceName", ""),
                    "date": race.get("date"),
                    "driver_id": driver.get("driverId"),
                    "driver": " ".join(
                        part
                        for part in [
                            driver.get("givenName", ""),
                            driver.get("familyName", ""),
                        ]
                        if part
                    ),
                    "constructor": constructor.get("name", ""),
                    "grid": _to_int(result.get("grid")),
                    "finish_position": _to_int(result.get("position")),
                    "position_text": result.get("positionText", ""),
                    "points": _to_float(result.get("points")),
                    "status": result.get("status", ""),
                    "laps": _to_int(result.get("laps"), 0),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["round", "finish_position"]).reset_index(drop=True)
    return df


def get_sprint_results(season: int) -> pd.DataFrame:
    races = _get_all_races(season, "sprint")

    rows = []
    for race in races:
        for result in race.get("SprintResults", []):
            driver = result.get("Driver", {})
            rows.append(
                {
                    "round": _to_int(race.get("round")),
                    "race": race.get("raceName", ""),
                    "driver_id": driver.get("driverId"),
                    "driver": " ".join(
                        part
                        for part in [
                            driver.get("givenName", ""),
                            driver.get("familyName", ""),
                        ]
                        if part
                    ),
                    "points": _to_float(result.get("points")),
                }
            )
    return pd.DataFrame(rows)


def build_driver_metrics(
    standings: pd.DataFrame,
    race_results: pd.DataFrame,
) -> pd.DataFrame:
    if standings.empty:
        return pd.DataFrame()

    if race_results.empty:
        result = standings.copy()
        result["podiums"] = 0
        result["dnfs"] = 0
        result["avg_grid"] = pd.NA
        result["avg_finish"] = pd.NA
        result["race_points"] = 0.0
        return result

    race = race_results.copy()
    race["podium"] = race["finish_position"].between(1, 3, inclusive="both")

    status = race["status"].fillna("")
    race["dnf"] = ~(
        status.eq("Finished")
        | status.str.startswith("+")
    )

    metrics = (
        race.groupby(["driver_id", "driver"], as_index=False)
        .agg(
            podiums=("podium", "sum"),
            dnfs=("dnf", "sum"),
            avg_grid=("grid", "mean"),
            avg_finish=("finish_position", "mean"),
            race_points=("points", "sum"),
        )
    )

    result = standings.merge(
        metrics,
        on=["driver_id", "driver"],
        how="left",
    )

    for col in ["podiums", "dnfs"]:
        result[col] = result[col].fillna(0).astype(int)
    for col in ["avg_grid", "avg_finish", "race_points"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result.sort_values("position").reset_index(drop=True)


def build_points_progression(
    race_results: pd.DataFrame,
    sprint_results: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    if not race_results.empty:
        race_points = race_results[
            ["round", "race", "driver_id", "driver", "points"]
        ].copy()
        frames.append(race_points)

    if not sprint_results.empty:
        sprint_points = sprint_results[
            ["round", "race", "driver_id", "driver", "points"]
        ].copy()
        frames.append(sprint_points)

    if not frames:
        return pd.DataFrame()

    all_points = pd.concat(frames, ignore_index=True)
    progression = (
        all_points.groupby(
            ["round", "race", "driver_id", "driver"],
            as_index=False,
        )["points"]
        .sum()
        .rename(columns={"points": "event_points"})
        .sort_values(["driver", "round"])
    )
    progression["cumulative_points"] = (
        progression.groupby("driver_id")["event_points"].cumsum()
    )
    return progression.reset_index(drop=True)


def get_qualifying_results(season: int) -> pd.DataFrame:
    """Fetch ALL qualifying results with the same paginated API helper."""
    races = _get_all_races(season, "qualifying")
    rows = []
    for race in races:
        for result in race.get("QualifyingResults", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            rows.append({
                "round": _to_int(race.get("round")),
                "race": race.get("raceName", ""),
                "driver_id": driver.get("driverId"),
                "driver": " ".join(filter(None, [driver.get("givenName"), driver.get("familyName")])),
                "constructor": constructor.get("name", ""),
                "quali_position": _to_int(result.get("position")),
                "Q1": result.get("Q1", ""),
                "Q2": result.get("Q2", ""),
                "Q3": result.get("Q3", ""),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["round", "driver_id"]).sort_values(["round", "quali_position"]).reset_index(drop=True)
    return df


def build_teammate_matchups(race_results: pd.DataFrame, qualifying: pd.DataFrame) -> pd.DataFrame:
    """Compare only drivers who share a constructor and enter the SAME round."""
    if race_results.empty:
        return pd.DataFrame()
    race = race_results.copy()
    race["classified"] = race["status"].fillna("").eq("Finished") | race["status"].fillna("").str.startswith("+")
    if not qualifying.empty:
        race = race.merge(qualifying[["round", "driver_id", "quali_position"]], on=["round", "driver_id"], how="left")
    else:
        race["quali_position"] = float("nan")
    entries = []
    for (round_num, team), group in race.groupby(["round", "constructor"]):
        if len(group) != 2:
            continue
        a, b = group.iloc[0], group.iloc[1]
        entries.append({"round": round_num, "race": a["race"], "constructor": team,
            "driver_a": a["driver"], "driver_b": b["driver"],
            "quali_winner": a["driver"] if pd.notna(a["quali_position"]) and pd.notna(b["quali_position"]) and a["quali_position"] < b["quali_position"] else b["driver"] if pd.notna(a["quali_position"]) and pd.notna(b["quali_position"]) and b["quali_position"] < a["quali_position"] else None,
            "race_winner": a["driver"] if a["classified"] and (not b["classified"] or a["finish_position"] < b["finish_position"]) else b["driver"] if b["classified"] and (not a["classified"] or b["finish_position"] < a["finish_position"]) else None,
            "a_points": a["points"], "b_points": b["points"],
        })
    return pd.DataFrame(entries)


# --- V3: constructor/manufacturer metrics and on-demand race pit stops ---
def build_constructor_metrics(standings: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    race = results.copy()
    race["podium"] = race["finish_position"].between(1, 3)
    race["win"] = race["finish_position"].eq(1)
    race["finish"] = race["status"].fillna("").eq("Finished") | race["status"].fillna("").str.startswith("+")
    race["dnf"] = ~race["finish"]
    metrics = race.groupby("constructor", as_index=False).agg(
        race_points=("points", "sum"), race_wins=("win", "sum"),
        podiums=("podium", "sum"), dnfs=("dnf", "sum"),
        starts=("driver_id", "count"), mean_finish=("finish_position", "mean"),
        mean_grid=("grid", lambda values: values[values > 0].mean()),
    )
    metrics["finish_rate_pct"] = (100 * (metrics["starts"] - metrics["dnfs"]) / metrics["starts"]).round(1)
    if standings.empty:
        return metrics.sort_values("race_points", ascending=False)
    return standings.merge(metrics, on="constructor", how="left").sort_values("position")


def build_constructor_progression(results: pd.DataFrame, sprints: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    race = results[["round", "race", "constructor", "points"]].copy()
    # Sprint classifications do not contain a reliable team field in every era;
    # use the same driver's constructor from the race in that round when available.
    if not sprints.empty:
        mapping = results[["round", "driver_id", "constructor"]].drop_duplicates()
        sp = sprints.merge(mapping, on=["round", "driver_id"], how="inner")
        if not sp.empty:
            race = pd.concat([race, sp[["round", "race", "constructor", "points"]]], ignore_index=True)
    sums = race.groupby(["round", "race", "constructor"], as_index=False)["points"].sum()
    sums = sums.rename(columns={"points": "event_points"}).sort_values(["constructor", "round"])
    sums["cumulative_points"] = sums.groupby("constructor")["event_points"].cumsum()
    return sums


def get_race_pitstops(season: int, round_number: int) -> pd.DataFrame:
    # Pitstop data in the Ergast-compatible endpoint exists only for supported eras.
    races = _get_all_races_for_path(f"{season}/{round_number}/pitstops")
    rows = []
    for race in races:
        for stop in race.get("PitStops", []):
            rows.append({"driver_id": stop.get("driverId", ""),
                         "stop": _to_int(stop.get("stop")),
                         "lap": _to_int(stop.get("lap")),
                         "duration_s": _to_float(stop.get("duration"), float("nan")),
                         "time": stop.get("time", "")})
    return pd.DataFrame(rows)


def _get_all_races_for_path(path: str) -> list[dict]:
    offset = 0
    rows = []
    while True:
        data = _get_json(path + ".json", {"limit": 100, "offset": offset})
        meta = data.get("MRData", {})
        page = meta.get("RaceTable", {}).get("Races", [])
        rows.extend(page)
        total = int(meta.get("total", 0))
        step = int(meta.get("limit", 100))
        if step < 1 or (not page and offset < total):
            raise F1DataError("Pit stop pagination failed")
        offset += step
        if offset >= total: return rows
