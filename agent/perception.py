"""
agent/perception.py — P: Perception module.

The perception module collects the agent's local observation at each time step:
  - Current time and location
  - Nearby reachable POIs (within a configurable radius)
  - Observable information about other agents in the vicinity
  - A structured perception dict used as input to memory retrieval and intent generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from data.poi import POI, POICategory, SpatialGrid
from agent.state import AgentState


@dataclass
class PerceptionContext:
    """
    Structured representation of an agent's local observation at time t.

    This is the *pₜ* in the paper's notation.
    """

    time_slot: int
    hour: int
    day: int
    current_lat: float
    current_lon: float
    current_poi_id: str
    current_poi_type: str
    # Nearby reachable POIs, sorted by distance (closest first)
    nearby_pois: List[Dict[str, Any]] = field(default_factory=list)
    # Basic stats about agents in the vicinity (anonymised)
    nearby_agent_count: int = 0
    # Embedding vector for memory similarity search (set externally)
    embedding: Optional[List[float]] = None

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_prompt_text(self, max_pois: int = 10) -> str:
        """Render perception as structured text for LLM prompt injection."""
        lines = [
            f"Time: {self.hour:02d}:{(self.time_slot * 10 % 60):02d}  Day: {self.day}",
            f"Location: ({self.current_lat:.4f}, {self.current_lon:.4f})"
            f"  POI type: {self.current_poi_type or 'unknown'}",
            f"Nearby agents: {self.nearby_agent_count}",
            "Reachable POIs:",
        ]
        for entry in self.nearby_pois[:max_pois]:
            lines.append(
                f"  - {entry['name']} [{entry['category']}]"
                f"  dist={entry['distance_m']:.0f}m"
                f"  id={entry['poi_id']}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_slot": self.time_slot,
            "hour": self.hour,
            "day": self.day,
            "current_lat": self.current_lat,
            "current_lon": self.current_lon,
            "current_poi_id": self.current_poi_id,
            "current_poi_type": self.current_poi_type,
            "nearby_pois": self.nearby_pois,
            "nearby_agent_count": self.nearby_agent_count,
        }

    def to_embedding_input(self) -> str:
        """Compact text representation used for embedding-based similarity search."""
        return (
            f"time={self.time_slot} "
            f"lat={self.current_lat:.3f} "
            f"lon={self.current_lon:.3f} "
            f"poi_type={self.current_poi_type} "
            f"hour={self.hour}"
        )


class PerceptionModule:
    """
    Builds a PerceptionContext for an agent at each simulation step.

    Parameters
    ----------
    spatial_grid : SpatialGrid
        Pre-built spatial index over city POIs.
    perception_radius_m : float
        Radius in metres within which POIs are considered reachable.
    max_nearby_pois : int
        Maximum number of nearby POIs to include in perception.
    """

    def __init__(
        self,
        spatial_grid: SpatialGrid,
        perception_radius_m: float = 3000.0,
        max_nearby_pois: int = 30,
    ) -> None:
        self._grid = spatial_grid
        self._radius = perception_radius_m
        self._max_pois = max_nearby_pois

    def perceive(
        self,
        state: AgentState,
        nearby_agent_count: int = 0,
    ) -> PerceptionContext:
        """
        Build the PerceptionContext for *state* at the current simulation step.

        Parameters
        ----------
        state : AgentState
            Current agent state.
        nearby_agent_count : int
            Number of other agents in the local vicinity (passed in by simulation).
        """
        nearby_pois = self._grid.query_radius(
            lat=state.current_lat,
            lon=state.current_lon,
            radius_m=self._radius,
        )
        # Sort by distance and cap
        nearby_pois.sort(
            key=lambda p: p.distance_to_coords(state.current_lat, state.current_lon)
        )
        nearby_pois = nearby_pois[: self._max_pois]

        nearby_poi_dicts = [
            {
                "poi_id": p.poi_id,
                "name": p.name,
                "lat": p.lat,
                "lon": p.lon,
                "category": p.category.value,
                "distance_m": p.distance_to_coords(state.current_lat, state.current_lon),
            }
            for p in nearby_pois
        ]

        return PerceptionContext(
            time_slot=state.time_slot,
            hour=state.hour,
            day=state.day,
            current_lat=state.current_lat,
            current_lon=state.current_lon,
            current_poi_id=state.current_poi_id,
            current_poi_type=state.current_poi_type,
            nearby_pois=nearby_poi_dicts,
            nearby_agent_count=nearby_agent_count,
        )
