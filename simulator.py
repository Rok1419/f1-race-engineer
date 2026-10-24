"""Deterministic what-if tools. NOT a race prediction model."""

from __future__ import annotations

import math

RACE_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)


def championship_projection(drivers, constructors, predicted_positions):
    """Return projected standings. Inputs are lists of dicts and driver->position map.

    Each hypothetical race awards standard GP points for P1-P10 only; sprint
    points and penalties are not forecast. Existing official totals are retained.
    """
    if len(set(predicted_positions.values())) != len(predicted_positions):
        raise ValueError("Two drivers cannot be assigned the same finishing position.")
    if any(not isinstance(v, int) or v < 1 for v in predicted_positions.values()):
        raise ValueError("Finishing positions must be positive integers.")

    awarded = {
        driver: RACE_POINTS[pos - 1] if 1 <= pos <= 10 else 0
        for driver, pos in predicted_positions.items()
    }
    new_drivers = []
    team_delta = {}
    for row in drivers:
        key = row["driver"]
        earned = awarded.get(key, 0)
        team = str(row.get("constructor", "")).split(",")[0].strip()
        team_delta[team] = team_delta.get(team, 0) + earned
        new_drivers.append({**row, "simulated_points": earned, "projected_points": float(row["points"]) + earned})
    new_drivers.sort(key=lambda d: (-d["projected_points"], int(d["position"])))

    new_teams = [
        {**row, "simulated_points": team_delta.get(row["constructor"], 0),
         "projected_points": float(row["points"]) + team_delta.get(row["constructor"], 0)}
        for row in constructors
    ]
    new_teams.sort(key=lambda d: (-d["projected_points"], int(d["position"])))
    return new_drivers, new_teams


def simulate_strategy(laps: int, base_lap: float, pit_loss: float,
                      compounds: list[str], pit_laps: list[int],
                      deg_soft: float, deg_medium: float, deg_hard: float,
                      safety_car_lap: int = 0, safety_car_duration: int = 0,
                      fuel_effect: float = 0.035, offsets_override: dict | None = None):
    """Toy race-time model; user-defined inputs, not calibrated telemetry.

    Pit stops occur AFTER the specified lap; next lap starts on fresh tyres.
    An optional safety-car period adds a fixed delta per lap, while a stop
    during the SC window uses 60% of normal pit-loss. These are assumptions.
    """
    if not 10 <= laps <= 100:
        raise ValueError("Race length must be between 10 and 100 laps.")
    if not math.isfinite(base_lap) or base_lap <= 0 or not math.isfinite(pit_loss) or pit_loss < 0:
        raise ValueError("Lap time must be positive and pit loss non-negative.")
    if len(compounds) != len(pit_laps) + 1:
        raise ValueError("One compound is needed for each stint.")
    if any(p not in ("Soft", "Medium", "Hard") for p in compounds):
        raise ValueError("Unsupported tyre compound.")
    if sorted(set(pit_laps)) != pit_laps or any(p <= 0 or p >= laps for p in pit_laps):
        raise ValueError("Pit laps must be unique, ascending and within race distance.")
    for val in (deg_soft, deg_medium, deg_hard, fuel_effect):
        if not math.isfinite(val) or val < 0:
            raise ValueError("Degradation and fuel effects must be non-negative.")
    degradation = {"Soft": deg_soft, "Medium": deg_medium, "Hard": deg_hard}
    offsets = {"Soft": -0.7, "Medium": 0.0, "Hard": 0.55}
    if offsets_override is not None:
        offsets.update(offsets_override)
    stint_idx = 0
    tyre_age = 0
    total = 0.0
    rows = []
    for lap in range(1, laps + 1):
        tyre_age += 1
        tyre = compounds[stint_idx]
        normal = base_lap + offsets[tyre] + degradation[tyre] * (tyre_age - 1)
        normal += fuel_effect * (laps - lap)  # heavier early-race fuel load
        sc_active = safety_car_lap > 0 and safety_car_lap <= lap < safety_car_lap + safety_car_duration
        lap_time = normal + (18.0 if sc_active else 0.0)
        stopping = lap in pit_laps
        stop_loss = pit_loss * (0.6 if sc_active else 1.0) if stopping else 0.0
        total += lap_time + stop_loss
        rows.append({"lap": lap, "compound": tyre, "tyre_age": tyre_age,
                     "lap_time_s": round(lap_time, 3), "pit_loss_s": round(stop_loss, 3),
                     "cumulative_s": round(total, 3), "stint": stint_idx + 1,
                     "safety_car": sc_active, "pit": stopping})
        if stopping:
            stint_idx += 1
            tyre_age = 0
    return {"total_seconds": total, "laps": rows, "stops": len(pit_laps)}
