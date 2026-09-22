from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class JourneyPoint:
    timestamp: datetime
    latitude: float
    longitude: float
    h3_cell: str


@dataclass
class Journey:
    user_id: str
    points: list[JourneyPoint] = field(default_factory=list)

    def append(self, point: JourneyPoint, max_points: int = 20) -> None:
        self.points.append(point)
        self.points.sort(key=lambda p: p.timestamp)
        if len(self.points) > max_points:
            self.points[:] = self.points[-max_points:]
