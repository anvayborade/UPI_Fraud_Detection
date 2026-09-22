from __future__ import annotations

import math

try:
    import h3
except ImportError:  # pragma: no cover
    h3 = None


def latlon_to_cell(latitude: float, longitude: float, resolution: int = 8) -> str:
    if h3 is not None:
        return h3.latlng_to_cell(latitude, longitude, resolution)
    # Deterministic fallback for environments where h3 is not installed.
    scale = 10 ** max(1, min(resolution - 4, 4))
    return f"grid:{round(latitude * scale)}:{round(longitude * scale)}:{resolution}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))
