"""
data/trajectory.py — Historical trajectory data preprocessing pipeline.

Expected raw CSV format:
    UserID, lat, lon, POI_type, timestamp

Output:
    - Normalised trajectory records (10-min discrete time slots)
    - Per-user feature summaries (high-freq POI types, time-of-day distribution,
      spatial centroid)
    - Train / test split saved as JSON
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.poi import POICategory, raw_type_to_category


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisitEvent:
    """A single visit event in a user's trajectory."""
    user_id: str
    poi_id: str
    lat: float
    lon: float
    poi_type: str
    category: POICategory
    timestamp: pd.Timestamp
    time_slot: int  # 0-143 (10-min slots in a day)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "poi_id": self.poi_id,
            "lat": self.lat,
            "lon": self.lon,
            "poi_type": self.poi_type,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "time_slot": self.time_slot,
        }


@dataclass
class DayTrajectory:
    """One day of visits for one user."""
    user_id: str
    date: str  # YYYY-MM-DD
    events: List[VisitEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "date": self.date,
            "events": [e.to_dict() for e in self.events],
        }


@dataclass
class UserFeatures:
    """Aggregate features extracted from a user's full trajectory history."""
    user_id: str
    # Top-5 most visited POI categories
    top_categories: List[Tuple[str, float]]   # [(category, fraction), ...]
    # Visit frequency by hour-of-day (24 values, normalised to sum=1)
    hourly_distribution: List[float]
    # Spatial centroid of all visited locations
    centroid_lat: float
    centroid_lon: float
    # Total number of visit events
    total_visits: int
    # Number of distinct days in trajectory
    num_days: int

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "top_categories": self.top_categories,
            "hourly_distribution": self.hourly_distribution,
            "centroid_lat": self.centroid_lat,
            "centroid_lon": self.centroid_lon,
            "total_visits": self.total_visits,
            "num_days": self.num_days,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _timestamp_to_slot(ts: pd.Timestamp) -> int:
    """Convert a timestamp to a 10-minute time slot index (0-143)."""
    minutes = ts.hour * 60 + ts.minute
    return min(minutes // 10, 143)


def load_raw_trajectories(path: str | Path) -> pd.DataFrame:
    """
    Load raw trajectory CSV.

    Required columns: UserID, lat, lon, POI_type, timestamp
    Optional columns: poi_id (generated if absent)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names to lower-case
    col_map: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("userid", "user_id"):
            col_map[c] = "user_id"
        elif lc == "lat":
            col_map[c] = "lat"
        elif lc == "lon":
            col_map[c] = "lon"
        elif lc in ("poi_type", "poi type", "poitype", "type"):
            col_map[c] = "poi_type"
        elif lc in ("timestamp", "time", "datetime"):
            col_map[c] = "timestamp"
        elif lc in ("poi_id", "poiid", "poi"):
            col_map[c] = "poi_id"
    df = df.rename(columns=col_map)

    required = {"user_id", "lat", "lon", "poi_type", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory CSV missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["poi_type"] = df["poi_type"].astype(str).str.strip()

    if "poi_id" not in df.columns:
        # Synthesise a stable poi_id from (lat, lon, poi_type)
        df["poi_id"] = (
            df["lat"].round(5).astype(str)
            + "_"
            + df["lon"].round(5).astype(str)
            + "_"
            + df["poi_type"].str.replace(" ", "_", regex=False)
        )

    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def filter_users(
    df: pd.DataFrame,
    min_days: int = 5,
    min_visits: int = 20,
) -> pd.DataFrame:
    """Keep only users with sufficient trajectory data."""
    df["date"] = df["timestamp"].dt.date.astype(str)
    stats = df.groupby("user_id").agg(
        num_days=("date", "nunique"),
        num_visits=("timestamp", "count"),
    )
    valid_users = stats[
        (stats["num_days"] >= min_days) & (stats["num_visits"] >= min_visits)
    ].index
    return df[df["user_id"].isin(valid_users)].copy()


def build_day_trajectories(df: pd.DataFrame) -> List[DayTrajectory]:
    """Convert flat dataframe rows into per-day DayTrajectory objects."""
    trajectories: List[DayTrajectory] = []
    df["date"] = df["timestamp"].dt.date.astype(str)
    for (user_id, date), group in df.groupby(["user_id", "date"]):
        events: List[VisitEvent] = []
        for row in group.itertuples(index=False):
            cat = raw_type_to_category(row.poi_type)
            events.append(VisitEvent(
                user_id=str(user_id),
                poi_id=str(row.poi_id),
                lat=float(row.lat),
                lon=float(row.lon),
                poi_type=str(row.poi_type),
                category=cat,
                timestamp=row.timestamp,
                time_slot=_timestamp_to_slot(row.timestamp),
            ))
        events.sort(key=lambda e: e.timestamp)
        trajectories.append(DayTrajectory(
            user_id=str(user_id),
            date=str(date),
            events=events,
        ))
    return trajectories


def extract_user_features(
    df: pd.DataFrame,
    user_id: str,
) -> UserFeatures:
    """Extract aggregate features for a single user."""
    udf = df[df["user_id"] == user_id]

    cat_counts: Counter = Counter()
    hour_counts: np.ndarray = np.zeros(24, dtype=float)

    for row in udf.itertuples(index=False):
        cat = raw_type_to_category(row.poi_type)
        cat_counts[cat.value] += 1
        hour_counts[row.timestamp.hour] += 1

    total = max(len(udf), 1)
    top_cats = [
        (cat, count / total)
        for cat, count in cat_counts.most_common(5)
    ]

    hourly = (hour_counts / hour_counts.sum()).tolist() if hour_counts.sum() > 0 else [0.0] * 24

    num_days = int(udf["timestamp"].dt.date.nunique())

    return UserFeatures(
        user_id=user_id,
        top_categories=top_cats,
        hourly_distribution=hourly,
        centroid_lat=float(udf["lat"].mean()),
        centroid_lon=float(udf["lon"].mean()),
        total_visits=total,
        num_days=num_days,
    )


def train_test_split(
    trajectories: List[DayTrajectory],
    train_ratio: float = 0.8,
) -> Tuple[List[DayTrajectory], List[DayTrajectory]]:
    """
    Split trajectories chronologically per user (not randomly).
    The last (1 - train_ratio) fraction of each user's days go to test.
    """
    user_days: Dict[str, List[DayTrajectory]] = defaultdict(list)
    for t in trajectories:
        user_days[t.user_id].append(t)
    for uid in user_days:
        user_days[uid].sort(key=lambda t: t.date)

    train, test = [], []
    for uid, days in user_days.items():
        split = max(1, int(len(days) * train_ratio))
        train.extend(days[:split])
        test.extend(days[split:])
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_trajectories(trajectories: List[DayTrajectory], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in trajectories], f, ensure_ascii=False, indent=2)


def load_trajectories(path: str | Path) -> List[DayTrajectory]:
    """Deserialise trajectories from JSON produced by save_trajectories."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    result: List[DayTrajectory] = []
    for r in records:
        events = []
        for e in r["events"]:
            events.append(VisitEvent(
                user_id=e["user_id"],
                poi_id=e["poi_id"],
                lat=e["lat"],
                lon=e["lon"],
                poi_type=e["poi_type"],
                category=POICategory(e["category"]),
                timestamp=pd.Timestamp(e["timestamp"]),
                time_slot=e["time_slot"],
            ))
        result.append(DayTrajectory(
            user_id=r["user_id"],
            date=r["date"],
            events=events,
        ))
    return result


def save_user_features(features: List[UserFeatures], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([feat.to_dict() for feat in features], f, ensure_ascii=False, indent=2)


def load_user_features(path: str | Path) -> List[UserFeatures]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [
        UserFeatures(
            user_id=r["user_id"],
            top_categories=[(c, v) for c, v in r["top_categories"]],
            hourly_distribution=r["hourly_distribution"],
            centroid_lat=r["centroid_lat"],
            centroid_lon=r["centroid_lon"],
            total_visits=r["total_visits"],
            num_days=r["num_days"],
        )
        for r in records
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    raw_path: str | Path,
    output_dir: str | Path,
    min_days: int = 5,
    min_visits: int = 20,
    train_ratio: float = 0.8,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw trajectories from {raw_path} ...")
    df = load_raw_trajectories(raw_path)
    print(f"  Loaded {len(df)} rows, {df['user_id'].nunique()} users.")

    print("Filtering users ...")
    df = filter_users(df, min_days=min_days, min_visits=min_visits)
    print(f"  Retained {df['user_id'].nunique()} users after filtering.")

    print("Building day trajectories ...")
    trajectories = build_day_trajectories(df)
    print(f"  Built {len(trajectories)} day-trajectories.")

    print("Extracting user features ...")
    user_ids = df["user_id"].unique().tolist()
    features = [extract_user_features(df, uid) for uid in user_ids]

    print("Train/test split ...")
    train, test = train_test_split(trajectories, train_ratio=train_ratio)
    print(f"  Train: {len(train)} days, Test: {len(test)} days.")

    save_trajectories(train, output_dir / "trajectories_train.json")
    save_trajectories(test, output_dir / "trajectories_test.json")
    save_user_features(features, output_dir / "user_features.json")
    print("Done. Saved to:", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess trajectory data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-days", type=int, default=5)
    parser.add_argument("--min-visits", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    run_pipeline(
        raw_path=args.input,
        output_dir=args.output,
        min_days=args.min_days,
        min_visits=args.min_visits,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    _main()
