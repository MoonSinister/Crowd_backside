"""
data/poi.py — POI data structures, semantic category system, and spatial grid.

POI semantic categories (aligned with paper's activity labels):
  dining, shopping, transport, residential, park, office, entertainment,
  education, healthcare, convenience

The spatial grid is a simple flat-earth grid used for fast neighbour lookup.
Exact great-circle distances use Haversine from geopy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic


# ─────────────────────────────────────────────────────────────────────────────
# Semantic category system
# ─────────────────────────────────────────────────────────────────────────────

class POICategory(str, Enum):
    """Top-level semantic categories for POI intent matching."""
    DINING = "dining"
    SHOPPING = "shopping"
    TRANSPORT = "transport"
    RESIDENTIAL = "residential"
    PARK = "park"
    OFFICE = "office"
    ENTERTAINMENT = "entertainment"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    CONVENIENCE = "convenience"
    OTHER = "other"


# Mapping from raw POI type strings (e.g. from OSM or survey datasets) to
# canonical POICategory values.  Extend as needed.
RAW_TYPE_TO_CATEGORY: Dict[str, POICategory] = {
    # Dining
    "restaurant": POICategory.DINING,
    "ramen restaurant": POICategory.DINING,
    "soba restaurant": POICategory.DINING,
    "italian restaurant": POICategory.DINING,
    "cafe": POICategory.DINING,
    "food": POICategory.DINING,
    # Shopping
    "convenience store": POICategory.CONVENIENCE,
    "supermarket": POICategory.SHOPPING,
    "farmers market": POICategory.SHOPPING,
    "mall": POICategory.SHOPPING,
    "shop": POICategory.SHOPPING,
    "retail": POICategory.SHOPPING,
    # Transport
    "train station": POICategory.TRANSPORT,
    "bus stop": POICategory.TRANSPORT,
    "public transport": POICategory.TRANSPORT,
    "subway": POICategory.TRANSPORT,
    # Residential
    "home": POICategory.RESIDENTIAL,
    "apartment": POICategory.RESIDENTIAL,
    "residential": POICategory.RESIDENTIAL,
    # Park / Open space
    "park": POICategory.PARK,
    "open space": POICategory.PARK,
    "garden": POICategory.PARK,
    # Office / Work
    "office": POICategory.OFFICE,
    "company": POICategory.OFFICE,
    "workplace": POICategory.OFFICE,
    # Entertainment
    "cinema": POICategory.ENTERTAINMENT,
    "theater": POICategory.ENTERTAINMENT,
    "museum": POICategory.ENTERTAINMENT,
    "sport": POICategory.ENTERTAINMENT,
    # Education
    "school": POICategory.EDUCATION,
    "university": POICategory.EDUCATION,
    "library": POICategory.EDUCATION,
    # Healthcare
    "hospital": POICategory.HEALTHCARE,
    "clinic": POICategory.HEALTHCARE,
    "pharmacy": POICategory.HEALTHCARE,
}


def raw_type_to_category(raw_type: str) -> POICategory:
    """Map a raw POI type string to a canonical POICategory."""
    return RAW_TYPE_TO_CATEGORY.get(raw_type.strip().lower(), POICategory.OTHER)


# ─────────────────────────────────────────────────────────────────────────────
# POI data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class POI:
    """A single Point of Interest."""
    poi_id: str
    name: str
    lat: float
    lon: float
    raw_type: str
    category: POICategory = field(init=False)

    def __post_init__(self) -> None:
        self.category = raw_type_to_category(self.raw_type)

    def distance_to(self, other: "POI") -> float:
        """Haversine distance in metres."""
        return geodesic((self.lat, self.lon), (other.lat, other.lon)).meters

    def distance_to_coords(self, lat: float, lon: float) -> float:
        """Haversine distance in metres to a coordinate pair."""
        return geodesic((self.lat, self.lon), (lat, lon)).meters

    def to_dict(self) -> dict:
        return {
            "poi_id": self.poi_id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "raw_type": self.raw_type,
            "category": self.category.value,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Spatial grid for fast neighbour lookup
# ─────────────────────────────────────────────────────────────────────────────

class SpatialGrid:
    """
    Flat-earth spatial grid index for O(1) cell lookup + cheap radius queries.

    The globe is approximated as a flat surface at the city's average latitude.
    Cells are roughly *cell_size_m* × *cell_size_m* metres.
    """

    def __init__(self, pois: List[POI], cell_size_m: float = 500.0) -> None:
        self.cell_size_m = cell_size_m
        self.pois = {p.poi_id: p for p in pois}

        # Determine bounding box
        lats = [p.lat for p in pois]
        lons = [p.lon for p in pois]
        self.lat_min = min(lats)
        self.lat_max = max(lats)
        self.lon_min = min(lons)
        self.lon_max = max(lons)

        # Degrees per metre (approximate at city centre)
        mid_lat = (self.lat_min + self.lat_max) / 2.0
        self._deg_per_m_lat = 1.0 / 111_320.0
        self._deg_per_m_lon = 1.0 / (111_320.0 * math.cos(math.radians(mid_lat)))

        self._dlat = cell_size_m * self._deg_per_m_lat
        self._dlon = cell_size_m * self._deg_per_m_lon

        # Build grid
        self._grid: Dict[Tuple[int, int], List[str]] = {}
        for poi in pois:
            key = self._cell_key(poi.lat, poi.lon)
            self._grid.setdefault(key, []).append(poi.poi_id)

    def _cell_key(self, lat: float, lon: float) -> Tuple[int, int]:
        row = int((lat - self.lat_min) / self._dlat)
        col = int((lon - self.lon_min) / self._dlon)
        return (row, col)

    def query_radius(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        category: Optional[POICategory] = None,
    ) -> List[POI]:
        """Return all POIs within *radius_m* metres of (lat, lon)."""
        cell_radius = int(math.ceil(radius_m / self.cell_size_m)) + 1
        cr, cc = self._cell_key(lat, lon)
        candidates: List[POI] = []
        for dr in range(-cell_radius, cell_radius + 1):
            for dc in range(-cell_radius, cell_radius + 1):
                for pid in self._grid.get((cr + dr, cc + dc), []):
                    p = self.pois[pid]
                    if category is None or p.category == category:
                        candidates.append(p)
        # Filter by exact distance
        result = [
            p for p in candidates
            if geodesic((lat, lon), (p.lat, p.lon)).meters <= radius_m
        ]
        return result

    def query_category(self, category: POICategory) -> List[POI]:
        """Return all POIs of a given category."""
        return [p for p in self.pois.values() if p.category == category]

    def nearest(
        self,
        lat: float,
        lon: float,
        category: Optional[POICategory] = None,
        k: int = 10,
    ) -> List[Tuple[POI, float]]:
        """Return the *k* nearest POIs (optionally filtered by category)."""
        pool = (
            list(self.pois.values())
            if category is None
            else self.query_category(category)
        )
        dists = [
            (p, geodesic((lat, lon), (p.lat, p.lon)).meters)
            for p in pool
        ]
        dists.sort(key=lambda x: x[1])
        return dists[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Distance matrix (pre-computed for simulation efficiency)
# ─────────────────────────────────────────────────────────────────────────────

class DistanceMatrix:
    """
    Symmetric POI-to-POI distance matrix (Haversine, in metres).
    For large POI sets consider a sparse approximation.
    """

    def __init__(self, pois: List[POI]) -> None:
        self._ids = [p.poi_id for p in pois]
        self._idx = {pid: i for i, pid in enumerate(self._ids)}
        n = len(pois)
        self._mat = np.zeros((n, n), dtype=np.float32)
        for i, pi in enumerate(pois):
            for j, pj in enumerate(pois):
                if i < j:
                    d = geodesic((pi.lat, pi.lon), (pj.lat, pj.lon)).meters
                    self._mat[i, j] = d
                    self._mat[j, i] = d

    def get(self, id_a: str, id_b: str) -> float:
        return float(self._mat[self._idx[id_a], self._idx[id_b]])

    def row(self, poi_id: str) -> np.ndarray:
        return self._mat[self._idx[poi_id]]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_pois_from_csv(path: str | Path) -> List[POI]:
    """
    Load POIs from a CSV with columns:
        poi_id, name, lat, lon, raw_type
    """
    df = pd.read_csv(path)
    required = {"poi_id", "name", "lat", "lon", "raw_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"POI CSV missing columns: {missing}")
    return [
        POI(
            poi_id=str(row.poi_id),
            name=str(row.name),
            lat=float(row.lat),
            lon=float(row.lon),
            raw_type=str(row.raw_type),
        )
        for row in df.itertuples(index=False)
    ]


def save_pois_to_json(pois: List[POI], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in pois], f, ensure_ascii=False, indent=2)


def load_pois_from_json(path: str | Path) -> List[POI]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [
        POI(
            poi_id=r["poi_id"],
            name=r["name"],
            lat=r["lat"],
            lon=r["lon"],
            raw_type=r["raw_type"],
        )
        for r in records
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build POI index from CSV")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    pois = load_pois_from_csv(args.input)
    save_pois_to_json(pois, args.output)
    print(f"Saved {len(pois)} POIs to {args.output}")


if __name__ == "__main__":
    _main()
