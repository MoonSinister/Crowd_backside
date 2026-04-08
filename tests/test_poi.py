"""
tests/test_poi.py — Unit tests for the POI data layer.
"""

import pytest
from data.poi import (
    POI,
    POICategory,
    SpatialGrid,
    DistanceMatrix,
    raw_type_to_category,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_pois():
    return [
        POI("p1", "Ramen Restaurant", 35.0010, 139.0010, "ramen restaurant"),
        POI("p2", "Convenience Store", 35.0020, 139.0020, "convenience store"),
        POI("p3", "Park", 35.0030, 139.0030, "park"),
        POI("p4", "Office", 35.0040, 139.0040, "office"),
        POI("p5", "Supermarket", 35.0050, 139.0050, "supermarket"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_raw_type_to_category_known():
    assert raw_type_to_category("Ramen Restaurant") == POICategory.DINING
    assert raw_type_to_category("convenience store") == POICategory.CONVENIENCE
    assert raw_type_to_category("park") == POICategory.PARK
    assert raw_type_to_category("office") == POICategory.OFFICE


def test_raw_type_to_category_unknown():
    assert raw_type_to_category("spaceship") == POICategory.OTHER


def test_poi_category_assigned_in_post_init():
    poi = POI("x", "Test", 35.0, 139.0, "restaurant")
    assert poi.category == POICategory.DINING


def test_poi_distance():
    p1 = POI("a", "A", 35.0, 139.0, "office")
    p2 = POI("b", "B", 35.0, 139.0, "office")
    assert p1.distance_to(p2) == pytest.approx(0.0, abs=1.0)

    p3 = POI("c", "C", 35.01, 139.0, "park")
    # ~1.1 km
    assert 1000 < p1.distance_to(p3) < 1200


def test_spatial_grid_query_radius():
    pois = _make_pois()
    grid = SpatialGrid(pois, cell_size_m=200)
    # Query at p1's location with 5 km radius — should find all 5 POIs
    results = grid.query_radius(35.0010, 139.0010, radius_m=5000)
    assert len(results) == 5


def test_spatial_grid_query_radius_small():
    pois = _make_pois()
    grid = SpatialGrid(pois, cell_size_m=100)
    # Very small radius — should only find the POI at the exact location
    results = grid.query_radius(35.0010, 139.0010, radius_m=50)
    assert any(p.poi_id == "p1" for p in results)


def test_spatial_grid_query_category():
    pois = _make_pois()
    grid = SpatialGrid(pois, cell_size_m=200)
    parks = grid.query_category(POICategory.PARK)
    assert len(parks) == 1
    assert parks[0].poi_id == "p3"


def test_spatial_grid_nearest():
    pois = _make_pois()
    grid = SpatialGrid(pois, cell_size_m=200)
    nearest = grid.nearest(35.0010, 139.0010, k=3)
    assert len(nearest) == 3
    # Closest should be p1 itself (distance ≈ 0)
    assert nearest[0][0].poi_id == "p1"
    assert nearest[0][1] < 1.0


def test_distance_matrix():
    pois = _make_pois()
    dm = DistanceMatrix(pois)
    assert dm.get("p1", "p1") == pytest.approx(0.0, abs=0.1)
    d12 = dm.get("p1", "p2")
    d21 = dm.get("p2", "p1")
    assert d12 == pytest.approx(d21, rel=1e-4)
    assert d12 > 0
