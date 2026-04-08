"""
data/generate_sample_data.py — Generate synthetic sample data for development and testing.

Creates:
  data/raw/pois.csv          — 60 sample Points of Interest across 10 semantic categories
  data/raw/trajectories.csv  — synthetic daily trajectories for 10 simulated users over 10 days

Usage:
    python -m data.generate_sample_data
    python -m data.generate_sample_data --output-dir data/raw --num-users 20 --num-days 14

The generated data follows realistic urban mobility patterns:
  - Users have role-based home/work anchors
  - Activity varies by hour (commute, lunch, evening leisure)
  - Trajectories use 10-minute time slots as specified in the paper
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# City layout configuration (centred on a generic urban area)
# ─────────────────────────────────────────────────────────────────────────────

CITY_CENTER_LAT = 35.6895   # Tokyo-like coordinates (generic demo)
CITY_CENTER_LON = 139.6917
CITY_RADIUS_DEG = 0.05      # ~5 km bounding radius

# Seed for reproducibility
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Sample POI definitions
# ─────────────────────────────────────────────────────────────────────────────

# (name, poi_type, lat_offset, lon_offset)  — offsets from city centre
_POI_TEMPLATES = [
    # Dining
    ("Sakura Ramen",       "ramen restaurant",  0.012,  0.005),
    ("Cafe Verde",         "cafe",              0.008,  0.018),
    ("Sunrise Soba",       "soba restaurant",  -0.005,  0.025),
    ("Pizza Corner",       "restaurant",        0.020, -0.010),
    ("Noodle House",       "restaurant",       -0.015,  0.012),
    ("Tea Garden Cafe",    "cafe",              0.025,  0.003),
    # Shopping
    ("FamilyMart East",    "convenience store", 0.003,  0.010),
    ("7-Eleven North",     "convenience store",-0.008,  0.008),
    ("Lawson West",        "convenience store", 0.010, -0.015),
    ("Big Market",         "supermarket",       0.018,  0.020),
    ("City Mall",          "mall",             -0.020, -0.005),
    ("Farmers Market",     "farmers market",    0.030,  0.010),
    # Transport
    ("Central Station",    "train station",     0.000,  0.000),
    ("North Subway",       "subway",            0.022,  0.000),
    ("South Bus Stop",     "bus stop",         -0.018, -0.012),
    ("East Subway",        "subway",            0.005,  0.030),
    # Residential
    ("Green Heights",      "apartment",         0.040,  0.015),
    ("River View Apts",    "apartment",        -0.030,  0.025),
    ("Sunny Residence",    "residential",       0.035, -0.020),
    ("Park Side Homes",    "residential",      -0.040, -0.015),
    ("Campus Dorms",       "apartment",         0.028,  0.038),
    # Park
    ("Central Park",       "park",              0.010,  0.010),
    ("Riverbank Garden",   "park",             -0.012,  0.020),
    ("Hilltop Open Space", "open space",        0.025,  0.025),
    # Office
    ("Business Tower A",   "office",            0.002,  0.005),
    ("Tech Hub",           "office",           -0.005,  0.008),
    ("Government Office",  "office",            0.015,  0.002),
    ("Co-working Space",   "workplace",         0.008,  0.012),
    # Entertainment
    ("Cineplex City",      "cinema",            0.012, -0.005),
    ("Sports Arena",       "sport",            -0.025,  0.030),
    ("Art Museum",         "museum",            0.000,  0.015),
    ("Karaoke Box",        "entertainment",     0.018,  0.008),
    # Education
    ("City University",    "university",        0.030,  0.035),
    ("High School North",  "school",            0.020,  0.028),
    ("Public Library",     "library",           0.005,  0.018),
    # Healthcare
    ("General Hospital",   "hospital",         -0.010, -0.020),
    ("Community Clinic",   "clinic",            0.015, -0.008),
    ("Downtown Pharmacy",  "pharmacy",          0.003, -0.003),
    # Convenience / mixed
    ("Mini Stop South",    "convenience store",-0.022,  0.005),
    ("Daily Mart",         "convenience store", 0.040, -0.010),
    ("Vending Corner",     "convenience store",-0.035, -0.025),
    ("Express Shop",       "convenience store", 0.045,  0.020),
    ("West Supermarket",   "supermarket",      -0.028,  0.015),
    ("East Retail Strip",  "retail",            0.032, -0.005),
    ("Night Market",       "farmers market",   -0.015, -0.030),
    # Extra dining and residential for variety
    ("Burger Palace",      "restaurant",       -0.032,  0.008),
    ("Sushi Express",      "restaurant",        0.042,  0.000),
    ("Mountain View Apts", "apartment",        -0.045,  0.035),
    ("Office Park B",      "office",           -0.015,  0.002),
    ("Lakeside Office",    "workplace",         0.038, -0.018),
    ("Youth Hostel",       "residential",      -0.038,  0.040),
    ("Tech University",    "university",        0.048, -0.022),
    ("East Clinic",        "clinic",            0.028, -0.025),
    ("West Pharmacy",      "pharmacy",         -0.042,  0.018),
    ("Opera House",        "theater",          -0.008, -0.015),
    ("Gym Plus",           "sport",             0.012, -0.022),
    ("Green Garden",       "garden",           -0.020,  0.035),
    ("Library Annex",      "library",           0.018,  0.042),
    ("Outdoor Stadium",    "sport",            -0.048, -0.008),
    ("West Park",          "park",             -0.030, -0.020),
    ("Dental Clinic",      "clinic",            0.022,  0.015),
]


# ─────────────────────────────────────────────────────────────────────────────
# User role definitions — determine home/work/activity patterns
# ─────────────────────────────────────────────────────────────────────────────

ROLES = {
    "office_worker": {
        "home_types":    ["apartment", "residential"],
        "work_types":    ["office", "workplace"],
        "schedule": [
            # (start_hour, end_hour, poi_types)
            (7, 9,   ["train station", "subway", "convenience store"]),
            (9, 12,  ["office", "workplace"]),
            (12, 13, ["restaurant", "ramen restaurant", "cafe", "convenience store"]),
            (13, 18, ["office", "workplace"]),
            (18, 20, ["train station", "subway", "supermarket"]),
            (20, 23, ["apartment", "residential", "convenience store"]),
        ],
    },
    "student": {
        "home_types":    ["apartment", "residential"],
        "work_types":    ["university", "school", "library"],
        "schedule": [
            (8, 9,   ["train station", "subway"]),
            (9, 12,  ["university", "school", "library"]),
            (12, 13, ["cafe", "convenience store", "restaurant"]),
            (13, 17, ["university", "school", "library"]),
            (17, 19, ["park", "sport", "convenience store"]),
            (19, 23, ["apartment", "residential", "cafe"]),
        ],
    },
    "delivery_rider": {
        "home_types":    ["apartment", "residential"],
        "work_types":    ["restaurant", "ramen restaurant", "convenience store"],
        "schedule": [
            (10, 11, ["apartment", "residential"]),
            (11, 14, ["restaurant", "ramen restaurant", "convenience store"]),
            (14, 17, ["apartment", "residential"]),
            (17, 21, ["restaurant", "ramen restaurant", "soba restaurant"]),
            (21, 23, ["apartment", "residential"]),
        ],
    },
    "retail_employee": {
        "home_types":    ["apartment", "residential"],
        "work_types":    ["mall", "retail", "convenience store"],
        "schedule": [
            (8, 9,   ["bus stop", "train station"]),
            (9, 18,  ["mall", "retail", "convenience store", "shop"]),
            (18, 20, ["restaurant", "supermarket"]),
            (20, 23, ["apartment", "residential"]),
        ],
    },
    "night_shift_worker": {
        "home_types":    ["apartment", "residential"],
        "work_types":    ["office", "convenience store", "workplace"],
        "schedule": [
            (0,  8,  ["office", "workplace", "convenience store"]),
            (8, 10,  ["convenience store", "apartment"]),
            (10, 16, ["apartment", "residential"]),
            (16, 18, ["restaurant", "supermarket"]),
            (18, 22, ["park", "entertainment", "cinema"]),
            (22, 24, ["office", "workplace"]),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────────

def _jitter(value: float, scale: float = 0.001, rng: random.Random = None) -> float:
    """Add small random noise to a coordinate."""
    if rng is None:
        rng = random
    return value + rng.gauss(0, scale)


def generate_pois(output_path: Path, rng: random.Random) -> list:
    """Generate sample POI CSV and return list of poi dicts."""
    pois = []
    for idx, (name, poi_type, dlat, dlon) in enumerate(_POI_TEMPLATES):
        poi = {
            "poi_id": f"poi_{idx:04d}",
            "name": name,
            "lat": round(CITY_CENTER_LAT + dlat + rng.gauss(0, 0.001), 6),
            "lon": round(CITY_CENTER_LON + dlon + rng.gauss(0, 0.001), 6),
            "raw_type": poi_type,
        }
        pois.append(poi)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["poi_id", "name", "lat", "lon", "raw_type"])
        writer.writeheader()
        writer.writerows(pois)

    print(f"  Generated {len(pois)} POIs → {output_path}")
    return pois


def _pois_of_types(pois: list, type_list: list) -> list:
    """Filter POI list to those matching any of the given raw types."""
    return [p for p in pois if p["raw_type"] in type_list]


def _nearest_poi(pois: list, lat: float, lon: float) -> dict:
    """Return the closest POI to (lat, lon)."""
    return min(
        pois,
        key=lambda p: math.hypot(p["lat"] - lat, p["lon"] - lon),
    )


def generate_trajectories(
    output_path: Path,
    pois: list,
    num_users: int,
    num_days: int,
    rng: random.Random,
) -> None:
    """Generate synthetic trajectory CSV."""
    role_list = list(ROLES.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    base_date = datetime(2023, 1, 1, 0, 0, 0)

    for user_idx in range(num_users):
        role_name = role_list[user_idx % len(role_list)]
        role = ROLES[role_name]

        # Assign stable home and work POIs for this user
        home_candidates = _pois_of_types(pois, role["home_types"])
        work_candidates = _pois_of_types(pois, role["work_types"])

        home_poi = rng.choice(home_candidates) if home_candidates else pois[0]
        work_poi = rng.choice(work_candidates) if work_candidates else pois[1]

        for day_idx in range(num_days):
            current_poi = home_poi
            day_base = base_date + timedelta(days=day_idx)

            for schedule_entry in role["schedule"]:
                start_h, end_h, poi_types = schedule_entry

                # Sample 1-3 visits within this time block
                num_visits = rng.randint(1, 3)
                block_minutes = (end_h - start_h) * 60
                if block_minutes <= 0:
                    continue

                visit_times = sorted(
                    rng.randint(0, max(0, block_minutes - 1))
                    for _ in range(num_visits)
                )

                for visit_offset_min in visit_times:
                    visit_time = day_base + timedelta(
                        hours=start_h, minutes=visit_offset_min
                    )
                    # Round to nearest 10 minutes
                    floored = visit_time.replace(
                        minute=(visit_time.minute // 10) * 10, second=0, microsecond=0
                    )

                    # Select a POI of the appropriate type
                    candidates = _pois_of_types(pois, poi_types)
                    if not candidates:
                        candidates = [current_poi]

                    # Prefer nearby POIs (70% chance) vs any matching POI (30%)
                    if rng.random() < 0.7:
                        visit_poi = _nearest_poi(candidates, current_poi["lat"], current_poi["lon"])
                    else:
                        visit_poi = rng.choice(candidates)

                    rows.append({
                        "UserID": f"user_{user_idx:04d}",
                        "lat":    round(_jitter(visit_poi["lat"], 0.0005, rng), 6),
                        "lon":    round(_jitter(visit_poi["lon"], 0.0005, rng), 6),
                        "POI_type": visit_poi["raw_type"],
                        "timestamp": floored.strftime("%Y-%m-%d %H:%M:%S"),
                        "poi_id": visit_poi["poi_id"],
                    })
                    current_poi = visit_poi

    # Sort by user, then timestamp
    rows.sort(key=lambda r: (r["UserID"], r["timestamp"]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["UserID", "lat", "lon", "POI_type", "timestamp", "poi_id"]
        )
        writer.writeheader()
        writer.writerows(rows)

    users = len({r["UserID"] for r in rows})
    print(f"  Generated {len(rows)} trajectory records for {users} users → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(output_dir: str = "data/raw", num_users: int = 10, num_days: int = 10) -> None:
    rng = random.Random(RANDOM_SEED)
    out = Path(output_dir)

    print(f"Generating sample data in '{out}/' ...")
    pois = generate_pois(out / "pois.csv", rng)
    generate_trajectories(out / "trajectories.csv", pois, num_users, num_days, rng)
    print("Done.  You can now run the preprocessing pipeline:")
    print(f"  python -m data.trajectory --input {out}/trajectories.csv --output data/processed/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sample data")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory for CSV files")
    parser.add_argument("--num-users", type=int, default=10, help="Number of simulated users")
    parser.add_argument("--num-days",  type=int, default=10, help="Number of simulation days per user")
    args = parser.parse_args()
    main(args.output_dir, args.num_users, args.num_days)
