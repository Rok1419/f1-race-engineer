"""OpenF1 historical tyre estimator; observational, not a definitive degradation value."""
from pathlib import Path
import statistics
import math
import json
import hashlib
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import requests

import numpy as np
import pandas as pd

COMPOUNDS = ("Soft", "Medium", "Hard")


def fetch_race_laps(year: int, round_number: int) -> pd.DataFrame:
    """Load historical race laps and tyre stints without F1 authentication.

    Matches a Jolpica calendar event to its OpenF1 Race session by date.
    OpenF1 historical race data is available from 2023 onwards.
    """
    if year < 2023:
        raise ValueError("OpenF1 historical tyre data starts in 2023. Choose 2023 or newer.")
    headers = {"User-Agent": "F1-Race-Engineer/4.5 (personal dashboard)"}
    # Disk cache survives Streamlit restarts and avoids repeated API calls.
    cache_dir = Path(__file__).resolve().parent / "openf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get(url, **params):
        signature = json.dumps([url, sorted(params.items())], separators=(",", ":"))
        cache_path = cache_dir / (hashlib.sha256(signature.encode()).hexdigest() + ".json")
        if cache_path.exists():
            try:
                # Historical snapshots can be reused for one month.
                if time.time() - cache_path.stat().st_mtime < 30 * 86400:
                    return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass

        for attempt in range(3):
            try:
                # Stay under the free-tier request-per-second limit even if the
                # dashboard needs multiple endpoints for one race.
                time.sleep(0.6)
                response = requests.get(url, params=params, headers=headers, timeout=30)
                if response.status_code == 429:
                    if attempt == 2:
                        raise RuntimeError(
                            "OpenF1 rate limit (HTTP 429). No further automatic requests "
                            "will be made. Try again later; already downloaded data is cached."
                        )
                    retry_header = response.headers.get("Retry-After", "")
                    delay = 15.0 * (2 ** attempt)
                    if retry_header:
                        try:
                            delay = float(retry_header)
                        except ValueError:
                            try:
                                date = parsedate_to_datetime(retry_header)
                                delay = (date - datetime.now(timezone.utc)).total_seconds()
                            except (TypeError, ValueError, OverflowError):
                                pass
                    time.sleep(min(60.0, max(1.0, delay)))
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, (dict, list)):
                    raise ValueError("Unexpected data format")
                tmp = cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.replace(cache_path)
                return data
            except requests.RequestException as exc:
                raise RuntimeError(f"Data request failed: {exc}") from exc
            except ValueError as exc:
                raise RuntimeError(f"Could not read API response: {exc}") from exc
        raise RuntimeError("Could not load historical data")

    calendar = get(f"https://api.jolpi.ca/ergast/f1/{year}/{round_number}.json")
    events = calendar.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not events:
        raise ValueError("Selected race not found in Jolpica calendar")
    event_date = pd.to_datetime(events[0]["date"], utc=True)
    sessions = get("https://api.openf1.org/v1/sessions", year=year, session_name="Race")
    races = [x for x in sessions if x.get("session_name") == "Race" and not x.get("is_cancelled")]
    if not races:
        raise ValueError("OpenF1 has no completed races for this season yet")
    match = min(races, key=lambda x: abs((pd.to_datetime(x["date_start"], utc=True) - event_date).total_seconds()))
    day_delta = abs((pd.to_datetime(match["date_start"], utc=True) - event_date).total_seconds()) / 86400
    if day_delta > 2.5:
        raise ValueError("Matching OpenF1 race session not found (race may not have completed yet)")
    key = match["session_key"]
    laps = get("https://api.openf1.org/v1/laps", session_key=key)
    stints = get("https://api.openf1.org/v1/stints", session_key=key)
    # Driver names are optional: avoid losing useful lap/tyre data if this
    # additional OpenF1 endpoint is temporarily rate-limited.
    try:
        drivers = get("https://api.openf1.org/v1/drivers", session_key=key)
    except RuntimeError:
        drivers = []
    if not laps or not stints:
        raise ValueError("No OpenF1 lap or tyre-stint data available for this race")
    driver_info = {int(d["driver_number"]): d for d in drivers if d.get("driver_number") is not None}
    stint_info = {}
    for stint in stints:
        if stint.get("lap_start") is None or stint.get("lap_end") is None:
            continue
        number = int(stint["driver_number"])
        for lap in range(int(stint["lap_start"]), int(stint["lap_end"])+1):
            stint_info[number, lap] = stint

    rows=[]
    for entry in laps:
        number, lap = entry.get("driver_number"), entry.get("lap_number")
        if number is None or lap is None:
            continue
        number, lap = int(number), int(lap)
        stint = stint_info.get((number, lap))
        if stint is None:
            continue
        d = driver_info.get(number, {})
        compound = str(stint.get("compound") or "UNKNOWN").upper()
        tyre_age = int(stint.get("tyre_age_at_start") or 0) + lap - int(stint["lap_start"]) + 1
        duration = pd.to_numeric(entry.get("lap_duration"), errors="coerce")
        valid = pd.notna(duration) and 30 < duration < 240 and not bool(entry.get("is_pit_out_lap"))
        rows.append({
            "Driver": d.get("name_acronym") or str(number),
            "Team": d.get("team_name") or "Unknown",
            "LapNumber": lap,
            "LapTime": pd.to_timedelta(duration, unit="s") if pd.notna(duration) else pd.NaT,
            "lap_time_s": float(duration) if pd.notna(duration) else float("nan"),
            "Compound": compound,
            "TyreLife": tyre_age,
            "Stint": int(stint.get("stint_number") or 1),
            "IsAccurate": bool(valid),
            "PitInTime": None,
            "PitOutTime": "pit" if entry.get("is_pit_out_lap") else None,
            "TrackStatus": None,  # OpenF1 lap API does not label green-flag laps.
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Could not match laps to tyre stints")
    return result.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def estimate_tyres(laps: pd.DataFrame, driver: str, fuel_effect: float = 0.035) -> dict:
    """Fit lap seconds = compound intercept + degradation*tyre_age + fuel_effect*(remaining race laps).

    Calculate per-stint fits and pool slope/intercept using medians. A fixed fuel
    effect is user-adjustable; this is not a causal estimate of degradation.
    """
    needed = {"Driver", "LapNumber", "Compound", "TyreLife", "Stint", "lap_time_s"}
    if not needed.issubset(laps.columns):
        return {"estimates": {}, "reason": "Required lap data is missing."}
    raw = laps[laps["Driver"] == driver].copy()
    if raw.empty:
        return {"estimates": {}, "reason": "No laps for this driver."}
    max_lap = pd.to_numeric(raw["LapNumber"], errors="coerce").max()
    for field in ("LapNumber", "TyreLife", "Stint", "lap_time_s"):
        raw[field] = pd.to_numeric(raw[field], errors="coerce")
    raw = raw.dropna(subset=["LapNumber", "TyreLife", "Stint", "lap_time_s", "Compound"])
    raw = raw[raw["Compound"].isin([c.upper() for c in COMPOUNDS])]
    raw = raw[(raw["TyreLife"] >= 1) & (raw["lap_time_s"] > 30) & (raw["lap_time_s"] < 240)]
    if "IsAccurate" in raw:
        raw = raw[raw["IsAccurate"].eq(True)]
    if "PitInTime" in raw:
        raw = raw[raw["PitInTime"].isna()]
    if "PitOutTime" in raw:
        raw = raw[raw["PitOutTime"].isna()]
    if "TrackStatus" in raw and raw["TrackStatus"].notna().any():
        raw = raw[raw["TrackStatus"].fillna("").eq("1")]
    estimates = {}
    details = []
    for name in COMPOUNDS:
        compound_laps = raw[raw["Compound"] == name.upper()]
        samples = []
        for _, part in compound_laps.groupby("Stint"):
            if len(part) < 5 or part["TyreLife"].nunique() < 4:
                continue
            x = part["TyreLife"].to_numpy(dtype=float) - 1
            y = (part["lap_time_s"] - fuel_effect * (max_lap - part["LapNumber"])).to_numpy(dtype=float)
            # Robust first pass eliminates large traffic/inconsistent laps.
            slope, intercept = np.polyfit(x, y, 1)
            residuals = y - (intercept + slope*x)
            mad = np.median(np.abs(residuals - np.median(residuals)))
            threshold = max(0.7, 3.0 * 1.4826 * mad)
            mask = np.abs(residuals - np.median(residuals)) <= threshold
            if mask.sum() >= 5 and np.unique(x[mask]).size >= 4:
                x, y = x[mask], y[mask]
                slope, intercept = np.polyfit(x, y, 1)
            slope = float(slope)
            intercept = float(intercept)
            if not (math.isfinite(slope) and math.isfinite(intercept)):
                continue
            samples.append((slope, intercept, len(x)))
        if samples:
            slopes = [s[0] for s in samples]
            intercepts = [s[1] for s in samples]
            count = sum(s[2] for s in samples)
            slope = float(statistics.median(slopes))
            pace = float(statistics.median(intercepts))
            # Negative apparent degradation can arise due to fuel, traffic or car pace.
            use_slope = max(0.0, slope)
            confidence = "Moderate" if len(samples) >= 2 and count >= 12 else "Low"
            estimates[name] = {"pace_s": round(pace, 3), "degradation_s_per_lap": round(use_slope, 4), "raw_slope": round(slope, 4), "laps_used": count, "stints_used": len(samples), "confidence": confidence}
            details.append({"compound": name, **estimates[name]})
    return {"estimates": estimates, "rows": details, "driver": driver, "race_laps": int(max_lap), "fuel_effect": fuel_effect, "clean_laps": int(len(raw))}
