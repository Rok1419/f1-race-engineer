"""Future-race discovery and paired Monte Carlo strategy experiments.

All forecasts are hypothetical; no true future lap timings are known.
"""
import datetime as dt
import math
import numpy as np
import requests
from simulator import simulate_strategy

API = "https://api.jolpi.ca/ergast/f1"

def season_calendar(year: int, today=None):
    today = today or dt.datetime.now(dt.timezone.utc).date()
    try:
        r = requests.get(f"{API}/{year}/races.json", params={"limit": 100},
                         headers={"User-Agent": "F1RaceEngineer/4.2"}, timeout=20)
        r.raise_for_status()
        races = r.json()["MRData"]["RaceTable"]["Races"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise RuntimeError(f"Cannot load F1 calendar: {exc}") from exc
    return [dict(round=int(r["round"]), name=r["raceName"],
                 date=r["date"], circuit_id=r["Circuit"]["circuitId"],
                 circuit=r["Circuit"]["circuitName"])
            for r in races if dt.date.fromisoformat(r["date"]) >= today]

def prior_circuit_event(year: int, circuit_id: str):
    """Lookup latest prior-year edition held at the same circuit."""
    for candidate in range(year - 1, max(2017, year - 6), -1):
        try:
            r = requests.get(f"{API}/{candidate}/circuits/{circuit_id}/races.json",
                             params={"limit": 100},
                             headers={"User-Agent": "F1RaceEngineer/4.2"}, timeout=15)
            r.raise_for_status()
            events = r.json()["MRData"]["RaceTable"]["Races"]
            if events:
                return (candidate, int(events[-1]["round"]), events[-1]["raceName"])
        except (requests.RequestException, KeyError, ValueError):
            continue
    return None

def trial_strategies(laps, base, pit_loss, plans, offsets, degradation,
                     n=500, pace_sigma=0.35, pit_sigma=1.5, degradation_sigma=0.025,
                     sc_probability=0.25, seed=42):
    """Paired simulations: A and B receive the SAME random race conditions.

    Uncertainty distributions are selected by user, not calibrated probabilities.
    Strategy score is isolated elapsed race time, not finishing position.
    """
    rng = np.random.default_rng(seed)
    scores = {"A": [], "B": []}
    for _ in range(n):
        common_base = max(30.0, base + rng.normal(0, pace_sigma))
        common_pit = max(0.0, pit_loss + rng.normal(0, pit_sigma))
        common_deg = {tyre: max(0.0, degradation[tyre] + rng.normal(0, degradation_sigma))
                      for tyre in ("Soft", "Medium", "Hard")}
        sc_lap = int(rng.integers(3, laps - 2)) if rng.random() < sc_probability else 0
        sc_len = 3 if sc_lap else 0
        for name in ("A", "B"):
            pits, compounds = plans[name]
            result = simulate_strategy(laps, common_base, common_pit, compounds, pits,
                common_deg["Soft"], common_deg["Medium"], common_deg["Hard"],
                sc_lap, sc_len, offsets_override=offsets)
            scores[name].append(result["total_seconds"])
    a, b = np.array(scores["A"]), np.array(scores["B"])
    diff = b - a
    return {"a_win_pct": round(float(np.mean(diff > 0) * 100),1),
            "b_win_pct": round(float(np.mean(diff < 0) * 100),1),
            "tie_pct": round(float(np.mean(diff == 0) * 100),1),
            "median_advantage_a_s": round(float(np.median(diff)),2),
            "difference_p10_s": round(float(np.quantile(diff, .1)),2),
            "difference_p90_s": round(float(np.quantile(diff, .9)),2),
            "a_median_min": round(float(np.median(a)/60),2),
            "b_median_min": round(float(np.median(b)/60),2),
            "samples": [{"simulation": i+1, "A": round(float(a[i])/60,3),
                         "B": round(float(b[i])/60,3), "B_minus_A_s": round(float(diff[i]),3)}
                        for i in range(n)]}
