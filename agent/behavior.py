"""
agent/behavior.py — A: Behavior module.

Implements intent-conditioned location sampling via the gravity model from
the paper (Eq. 3-8):

    P(l_{t+1} | l_t, I_t) = rho(l, I_t) · d(l_t, l)^{-beta}
                             / sum_{l' in L_t(I_t)} rho(l', I_t) · d(l_t, l')^{-beta}

where:
  - rho(l, I_t)      semantic matching strength between POI l and intent I_t
  - d(l_t, l)        Haversine distance in metres
  - beta             distance decay coefficient (> 0)
  - L_t(I_t)         candidate POI set: POIs whose category matches I_t

The module writes a new memory entry after every move.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
from geopy.distance import geodesic

from data.poi import POI, POICategory, SpatialGrid
from agent.state import AgentState
from agent.perception import PerceptionContext
from agent.memory import EpisodicMemory
from agent.intent import Intent


# ─────────────────────────────────────────────────────────────────────────────
# Semantic compatibility
# ─────────────────────────────────────────────────────────────────────────────

# Intent category → set of compatible POI categories.
# A POI is included in L_t(I_t) if its category is compatible with the intent.
INTENT_TO_POI_CATEGORIES: dict = {
    POICategory.DINING:        {POICategory.DINING, POICategory.CONVENIENCE},
    POICategory.SHOPPING:      {POICategory.SHOPPING, POICategory.CONVENIENCE},
    POICategory.TRANSPORT:     {POICategory.TRANSPORT},
    POICategory.RESIDENTIAL:   {POICategory.RESIDENTIAL},
    POICategory.PARK:          {POICategory.PARK},
    POICategory.OFFICE:        {POICategory.OFFICE},
    POICategory.ENTERTAINMENT: {POICategory.ENTERTAINMENT, POICategory.PARK},
    POICategory.EDUCATION:     {POICategory.EDUCATION},
    POICategory.HEALTHCARE:    {POICategory.HEALTHCARE},
    POICategory.CONVENIENCE:   {POICategory.CONVENIENCE, POICategory.SHOPPING},
    POICategory.OTHER:         set(POICategory),   # No restriction
}


def semantic_match_strength(poi: POI, intent: Intent) -> float:
    """
    rho(l, I_t): semantic matching strength in [0, 1].

    Returns 1.0 if the POI category exactly matches the intent category,
    0.5 if it is a secondary compatible category, and 0.0 otherwise.
    """
    compatible = INTENT_TO_POI_CATEGORIES.get(intent.category, set())
    if poi.category == intent.category:
        return 1.0
    if poi.category in compatible:
        return 0.5
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Gravity model sampler
# ─────────────────────────────────────────────────────────────────────────────

class BehaviorModule:
    """
    A: Behavior module — intent-conditioned next-location sampling.

    Parameters
    ----------
    spatial_grid : SpatialGrid
        City-wide POI index.
    distance_decay_beta : float
        β in the gravity formula.
    max_candidate_pois : int
        Upper bound on |L_t(I_t)| for efficiency.
    min_distance_m : float
        Minimum distance to consider (avoids division by zero for same-POI).
    search_radius_m : float
        Radius within which candidate POIs are searched.  Falls back to
        category-wide search if no candidates found within radius.
    """

    def __init__(
        self,
        spatial_grid: SpatialGrid,
        distance_decay_beta: float = 2.0,
        max_candidate_pois: int = 30,
        min_distance_m: float = 50.0,
        search_radius_m: float = 5000.0,
    ) -> None:
        self._grid = spatial_grid
        self._beta = distance_decay_beta
        self._max_candidates = max_candidate_pois
        self._min_dist = min_distance_m
        self._radius = search_radius_m
        self._rng = random.Random()

    def select_destination(
        self,
        state: AgentState,
        intent: Intent,
    ) -> Optional[POI]:
        """
        Sample the next destination POI given *state* and *intent*.

        Returns None only if no compatible POI exists in the city
        (should not happen with a well-populated POI dataset).
        """
        candidates = self._build_candidate_set(state, intent)
        if not candidates:
            return None

        weights = self._compute_weights(state, candidates, intent)
        if weights.sum() == 0:
            return self._rng.choice(candidates)

        # Normalise and sample
        probs = weights / weights.sum()
        idx = self._rng.choices(range(len(candidates)), weights=probs.tolist(), k=1)[0]
        return candidates[idx]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_candidate_set(
        self,
        state: AgentState,
        intent: Intent,
    ) -> List[POI]:
        """
        L_t(I_t): POIs semantically compatible with the current intent.

        First tries POIs within the search radius; if fewer than 3 found,
        falls back to all POIs of matching categories in the city.
        """
        compatible_cats = INTENT_TO_POI_CATEGORIES.get(intent.category, set(POICategory))

        # Radius search
        nearby = self._grid.query_radius(
            lat=state.current_lat,
            lon=state.current_lon,
            radius_m=self._radius,
        )
        candidates = [p for p in nearby if p.category in compatible_cats]

        if len(candidates) < 3:
            # Fallback: city-wide search
            candidates = []
            for cat in compatible_cats:
                candidates.extend(self._grid.query_category(cat))

        # Remove current POI (no self-loop)
        candidates = [p for p in candidates if p.poi_id != state.current_poi_id]

        # Cap for efficiency
        if len(candidates) > self._max_candidates:
            # Keep the max_candidates closest ones (pre-sort by distance)
            candidates.sort(
                key=lambda p: geodesic(
                    (state.current_lat, state.current_lon), (p.lat, p.lon)
                ).meters
            )
            candidates = candidates[: self._max_candidates]

        return candidates

    def _compute_weights(
        self,
        state: AgentState,
        candidates: List[POI],
        intent: Intent,
    ) -> np.ndarray:
        """
        Compute un-normalised gravity model weights for each candidate POI.

            w(l) = rho(l, I_t) * d(l_t, l)^{-beta}
        """
        weights = np.zeros(len(candidates), dtype=np.float64)
        for i, poi in enumerate(candidates):
            dist_m = geodesic(
                (state.current_lat, state.current_lon), (poi.lat, poi.lon)
            ).meters
            dist_m = max(dist_m, self._min_dist)
            rho = semantic_match_strength(poi, intent)
            weights[i] = rho * (dist_m ** (-self._beta))
        return weights

    # ── Memory write-back ──────────────────────────────────────────────────

    def execute_and_update(
        self,
        state: AgentState,
        perception: PerceptionContext,
        intent: Intent,
        query_embedding: List[float],
        memory: EpisodicMemory,
    ) -> Optional[POI]:
        """
        Select destination, move the agent, and write a memory entry.

        Returns the selected POI (or None if no candidate found).
        """
        destination = self.select_destination(state, intent)
        if destination is None:
            return None

        # Update agent state
        state.move_to(
            poi_id=destination.poi_id,
            lat=destination.lat,
            lon=destination.lon,
            poi_type=destination.raw_type,
        )

        # Write memory entry (mₜ = (pₜ, sₜ, Iₜ))
        current_slot = perception.time_slot + perception.day * 144
        memory.write(
            perception=perception,
            embedding=query_embedding,
            state_summary=state.to_prompt_text(),
            intent=intent.category.value,
            current_time_slot=current_slot,
        )

        return destination
