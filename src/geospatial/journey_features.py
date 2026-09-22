from __future__ import annotations

import math

from src.geospatial.h3_encoder import haversine_km
from src.geospatial.journey_builder import Journey


def calculate_journey_features(journey: Journey) -> dict[str, float]:
    if len(journey.points) < 2:
        return {
            "journey_distance_km": 0.0,
            "max_speed_kmh": 0.0,
            "location_continuity": 0.5,
            "journey_plausibility": 0.5,
            "impossible_travel": 0.0,
        }

    distances: list[float] = []
    speeds: list[float] = []
    for prev, current in zip(journey.points[:-1], journey.points[1:], strict=True):
        distance = haversine_km(prev.latitude, prev.longitude, current.latitude, current.longitude)
        hours = max((current.timestamp - prev.timestamp).total_seconds() / 3600.0, 1 / 3600)
        distances.append(distance)
        speeds.append(distance / hours)

    total = sum(distances)
    max_speed = max(speeds)
    impossible = float(max_speed > 900)
    continuity = math.exp(-max(0.0, max_speed - 130) / 250)
    plausibility = max(0.0, min(1.0, 0.75 * continuity + 0.25 * (1 - impossible)))
    return {
        "journey_distance_km": float(total),
        "max_speed_kmh": float(max_speed),
        "location_continuity": float(continuity),
        "journey_plausibility": float(plausibility),
        "impossible_travel": impossible,
    }
