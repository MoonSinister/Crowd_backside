"""
tests/test_behavior.py — Unit tests for the behavior module (gravity model).
"""

import pytest
from data.poi import POI, POICategory, SpatialGrid
from agent.behavior import (
    BehaviorModule,
    semantic_match_strength,
    INTENT_TO_POI_CATEGORIES,
)
from agent.intent import Intent, StubBackend
from agent.state import AgentProfile, AgentState
from agent.memory import EpisodicMemory


def _make_pois():
    return [
        POI("p1", "Ramen Restaurant",  35.001, 139.001, "ramen restaurant"),
        POI("p2", "Convenience Store", 35.002, 139.002, "convenience store"),
        POI("p3", "Park",              35.003, 139.003, "park"),
        POI("p4", "Office",            35.004, 139.004, "office"),
        POI("p5", "Supermarket",       35.005, 139.005, "supermarket"),
    ]


def _make_state(current_poi_id="p4"):
    profile = AgentProfile(
        role="office_worker",
        top_categories=[("dining", 0.4)],
        hourly_distribution=[1 / 24] * 24,
        centroid_lat=35.003,
        centroid_lon=139.003,
        description="Office worker",
    )
    return AgentState(
        agent_id="a0",
        profile=profile,
        current_lat=35.004,
        current_lon=139.004,
        current_poi_id=current_poi_id,
        current_poi_type="office",
        time_slot=72,  # 12:00
        day=0,
    )


def test_semantic_match_strength_exact():
    poi = POI("x", "R", 35.0, 139.0, "ramen restaurant")
    intent = Intent(POICategory.DINING)
    assert semantic_match_strength(poi, intent) == 1.0


def test_semantic_match_strength_secondary():
    poi = POI("x", "Convenience", 35.0, 139.0, "convenience store")
    intent = Intent(POICategory.DINING)
    assert semantic_match_strength(poi, intent) == 0.5


def test_semantic_match_strength_none():
    poi = POI("x", "Hospital", 35.0, 139.0, "hospital")
    intent = Intent(POICategory.DINING)
    assert semantic_match_strength(poi, intent) == 0.0


def test_gravity_model_selects_compatible_poi():
    pois = _make_pois()
    grid = SpatialGrid(pois, cell_size_m=200)
    behavior = BehaviorModule(grid, distance_decay_beta=2.0)
    state = _make_state()
    intent = Intent(POICategory.DINING, "hungry")

    destination = behavior.select_destination(state, intent)
    # Must be a dining or convenience store (compatible categories)
    assert destination is not None
    assert destination.category in INTENT_TO_POI_CATEGORIES[POICategory.DINING]


def test_gravity_model_never_returns_current_poi():
    pois = _make_pois()
    grid = SpatialGrid(pois, cell_size_m=200)
    behavior = BehaviorModule(grid, distance_decay_beta=2.0)
    state = _make_state(current_poi_id="p1")  # Currently at ramen restaurant
    intent = Intent(POICategory.DINING)

    # Run multiple times to check exclusion is stable
    for _ in range(20):
        dest = behavior.select_destination(state, intent)
        if dest is not None:
            assert dest.poi_id != "p1"


def test_execute_and_update_moves_agent():
    from agent.perception import PerceptionContext

    pois = _make_pois()
    grid = SpatialGrid(pois, cell_size_m=200)
    behavior = BehaviorModule(grid, distance_decay_beta=2.0)
    state = _make_state()
    intent = Intent(POICategory.DINING)

    perception = PerceptionContext(
        time_slot=72, hour=12, day=0,
        current_lat=state.current_lat, current_lon=state.current_lon,
        current_poi_id=state.current_poi_id, current_poi_type=state.current_poi_type,
    )
    memory = EpisodicMemory()
    llm = StubBackend()
    emb = llm.embed(perception.to_embedding_input())

    dest = behavior.execute_and_update(
        state=state,
        perception=perception,
        intent=intent,
        query_embedding=emb,
        memory=memory,
    )
    assert dest is not None
    # State should be updated to destination
    assert state.current_poi_id == dest.poi_id
    assert state.current_lat == dest.lat
    # Memory should have one entry
    assert len(memory) == 1
