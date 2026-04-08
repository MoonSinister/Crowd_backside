"""
simulation/environment.py — City simulation environment.

Wraps the city's spatial model (POI grid, distance matrix) and provides
the shared state that all agents query during simulation:
  - Agent position registry (for nearby-agent counting)
  - POI spatial grid
  - Simulation clock
  - Event log (trajectory records written per step)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.poi import POI, SpatialGrid


@dataclass
class SimulationClock:
    """Discrete time management for the simulation."""

    step_minutes: int = 10   # Each time step represents this many real minutes
    current_slot: int = 0    # 0-143 within a day
    current_day: int = 0

    @property
    def total_slot(self) -> int:
        return self.current_day * 144 + self.current_slot

    def advance(self) -> None:
        self.current_slot += 1
        if self.current_slot >= 144:
            self.current_slot = 0
            self.current_day += 1

    def time_str(self) -> str:
        h = (self.current_slot * self.step_minutes) // 60
        m = (self.current_slot * self.step_minutes) % 60
        return f"Day {self.current_day}  {h:02d}:{m:02d}"


@dataclass
class TrajectoryRecord:
    """A single movement event recorded during simulation."""
    agent_id: str
    day: int
    time_slot: int
    lat: float
    lon: float
    poi_id: str
    poi_type: str
    intent_category: str
    intent_explanation: str
    intent_path: str   # "fast" | "slow"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "day": self.day,
            "time_slot": self.time_slot,
            "lat": self.lat,
            "lon": self.lon,
            "poi_id": self.poi_id,
            "poi_type": self.poi_type,
            "intent_category": self.intent_category,
            "intent_explanation": self.intent_explanation,
            "intent_path": self.intent_path,
        }


class CityEnvironment:
    """
    Shared simulation environment accessible by all agents.

    Responsibilities:
    - Provide spatial queries (POI grid, radius search)
    - Maintain agent position registry for social perception
    - Log all trajectory records
    - Advance the simulation clock
    """

    def __init__(
        self,
        spatial_grid: SpatialGrid,
        step_minutes: int = 10,
    ) -> None:
        self._grid = spatial_grid
        self.clock = SimulationClock(step_minutes=step_minutes)
        # agent_id -> (lat, lon) live positions
        self._agent_positions: Dict[str, Tuple[float, float]] = {}
        self._trajectory_log: List[TrajectoryRecord] = []

    # ── Spatial ────────────────────────────────────────────────────────────

    @property
    def spatial_grid(self) -> SpatialGrid:
        return self._grid

    def count_agents_near(self, lat: float, lon: float, radius_m: float = 500.0) -> int:
        """Count agents within *radius_m* of (lat, lon)."""
        from geopy.distance import geodesic
        count = 0
        for agent_lat, agent_lon in self._agent_positions.values():
            if geodesic((lat, lon), (agent_lat, agent_lon)).meters <= radius_m:
                count += 1
        return count

    # ── Agent registry ─────────────────────────────────────────────────────

    def register_agent(self, agent_id: str, lat: float, lon: float) -> None:
        self._agent_positions[agent_id] = (lat, lon)

    def update_agent_position(self, agent_id: str, lat: float, lon: float) -> None:
        self._agent_positions[agent_id] = (lat, lon)

    def get_agent_position(self, agent_id: str) -> Optional[Tuple[float, float]]:
        return self._agent_positions.get(agent_id)

    # ── Logging ────────────────────────────────────────────────────────────

    def log_move(self, record: TrajectoryRecord) -> None:
        self._trajectory_log.append(record)

    def get_trajectory_log(self) -> List[TrajectoryRecord]:
        return list(self._trajectory_log)

    def save_trajectory_log(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in self._trajectory_log],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved {len(self._trajectory_log)} trajectory records to {path}")

    # ── Clock ──────────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the simulation clock by one time step."""
        self.clock.advance()

    @property
    def current_slot(self) -> int:
        return self.clock.current_slot

    @property
    def current_day(self) -> int:
        return self.clock.current_day
