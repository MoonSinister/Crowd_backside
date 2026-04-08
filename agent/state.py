"""
agent/state.py — S: Individual state layer.

Holds stable long-term attributes (role / profile) as well as dynamic
intra-simulation state (fatigue, current location, active schedule, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


SOCIAL_ROLES = [
    "student",
    "office_worker",
    "teacher",
    "night_shift_worker",
    "delivery_rider",
    "retail_employee",
]


@dataclass
class AgentProfile:
    """
    Long-term stable individual profile derived from historical trajectory data.
    Assigned once during initialisation via training.profile.

    Attributes
    ----------
    role : str
        Social role (one of SOCIAL_ROLES).
    top_categories : list[tuple[str, float]]
        Top-5 (category, fraction) pairs representing long-term POI preferences.
    hourly_distribution : list[float]
        Probability of being active at each hour of the day (length-24, sums to 1).
    centroid_lat : float
        Spatial centroid latitude of historical visits.
    centroid_lon : float
        Spatial centroid longitude of historical visits.
    description : str
        Natural-language summary of the profile (used in LLM prompts).
    """

    role: str
    top_categories: List[Tuple[str, float]] = field(default_factory=list)
    hourly_distribution: List[float] = field(default_factory=lambda: [1 / 24] * 24)
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0
    description: str = ""

    def to_prompt_text(self) -> str:
        """Render profile as structured text for LLM prompt injection."""
        cat_str = ", ".join(
            f"{cat} ({frac:.1%})" for cat, frac in self.top_categories
        )
        return (
            f"Role: {self.role}\n"
            f"Frequent activity types: {cat_str}\n"
            f"Usual home/centroid area: ({self.centroid_lat:.4f}, {self.centroid_lon:.4f})\n"
            f"Profile summary: {self.description}"
        )


@dataclass
class AgentState:
    """
    Dynamic runtime state of an agent during simulation.

    Attributes
    ----------
    agent_id : str
        Unique agent identifier.
    profile : AgentProfile
        Long-term profile (set at initialisation, read-only during sim).
    current_lat : float
        Current latitude.
    current_lon : float
        Current longitude.
    current_poi_id : str
        ID of the POI the agent is currently at.
    current_poi_type : str
        Raw type string of current POI.
    time_slot : int
        Current discrete time slot (0–143, each slot = 10 minutes).
    day : int
        Simulation day counter.
    fatigue : float
        Dynamic fatigue level in [0, 1] (higher = more fatigued).
    workload : float
        Current workload/busyness in [0, 1].
    extra : dict
        Extensible dict for role-specific state variables.
    """

    agent_id: str
    profile: AgentProfile
    current_lat: float = 0.0
    current_lon: float = 0.0
    current_poi_id: str = ""
    current_poi_type: str = ""
    time_slot: int = 0
    day: int = 0
    fatigue: float = 0.0
    workload: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── Derived helpers ────────────────────────────────────────────────────

    @property
    def hour(self) -> int:
        return (self.time_slot * 10) // 60

    @property
    def minute(self) -> int:
        return (self.time_slot * 10) % 60

    def time_str(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}"

    def to_prompt_text(self) -> str:
        """Render dynamic state as structured text for LLM prompt injection."""
        return (
            f"Current time: {self.time_str()} (day {self.day})\n"
            f"Current location: ({self.current_lat:.4f}, {self.current_lon:.4f})"
            f" — {self.current_poi_type or 'unknown POI'}\n"
            f"Fatigue: {self.fatigue:.2f}, Workload: {self.workload:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.profile.role,
            "current_lat": self.current_lat,
            "current_lon": self.current_lon,
            "current_poi_id": self.current_poi_id,
            "current_poi_type": self.current_poi_type,
            "time_slot": self.time_slot,
            "day": self.day,
            "fatigue": self.fatigue,
            "workload": self.workload,
            "extra": self.extra,
        }

    # ── State update helpers ───────────────────────────────────────────────

    def move_to(
        self,
        poi_id: str,
        lat: float,
        lon: float,
        poi_type: str,
    ) -> None:
        self.current_poi_id = poi_id
        self.current_lat = lat
        self.current_lon = lon
        self.current_poi_type = poi_type

    def advance_time(self) -> None:
        """Increment time slot by 1 (wraps at 144 to next day)."""
        self.time_slot += 1
        if self.time_slot >= 144:
            self.time_slot = 0
            self.day += 1
            # Natural fatigue recovery at day boundary
            self.fatigue = max(0.0, self.fatigue - 0.3)

    def update_fatigue(self, delta: float) -> None:
        self.fatigue = max(0.0, min(1.0, self.fatigue + delta))

    def update_workload(self, delta: float) -> None:
        self.workload = max(0.0, min(1.0, self.workload + delta))
