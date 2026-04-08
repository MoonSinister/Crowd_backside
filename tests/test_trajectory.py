"""
tests/test_trajectory.py — Unit tests for the trajectory preprocessing pipeline.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data.trajectory import (
    DayTrajectory,
    VisitEvent,
    UserFeatures,
    load_raw_trajectories,
    filter_users,
    build_day_trajectories,
    extract_user_features,
    train_test_split,
    save_trajectories,
    load_trajectories,
    _timestamp_to_slot,
)
from data.poi import POICategory


def _make_raw_df():
    rows = []
    for user_id in ["u1", "u2"]:
        for day in range(6):
            for hour in [8, 12, 18]:
                rows.append({
                    "user_id": user_id,
                    "lat": 35.001 + day * 0.001,
                    "lon": 139.001 + day * 0.001,
                    "poi_type": "restaurant" if hour == 12 else "office",
                    "timestamp": f"2019-01-{day+1:02d} {hour:02d}:00:00",
                    "poi_id": f"poi_{day}_{hour}",
                })
    return pd.DataFrame(rows)


def test_timestamp_to_slot():
    ts = pd.Timestamp("2019-01-01 08:00:00")
    assert _timestamp_to_slot(ts) == 48  # 8h * 6 slots/h

    ts2 = pd.Timestamp("2019-01-01 23:50:00")
    assert _timestamp_to_slot(ts2) == 143


def test_filter_users_keeps_sufficient():
    df = _make_raw_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    filtered = filter_users(df, min_days=5, min_visits=10)
    assert set(filtered["user_id"].unique()) == {"u1", "u2"}


def test_filter_users_removes_insufficient():
    df = _make_raw_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Very high thresholds
    filtered = filter_users(df, min_days=100, min_visits=1000)
    assert len(filtered) == 0


def test_build_day_trajectories():
    df = _make_raw_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = filter_users(df, min_days=1, min_visits=1)
    trajs = build_day_trajectories(df)
    # 2 users × 6 days = 12 DayTrajectory objects
    assert len(trajs) == 12
    for t in trajs:
        assert isinstance(t, DayTrajectory)
        assert len(t.events) > 0


def test_extract_user_features():
    df = _make_raw_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feat = extract_user_features(df, "u1")
    assert feat.user_id == "u1"
    assert feat.total_visits > 0
    assert len(feat.hourly_distribution) == 24
    assert abs(sum(feat.hourly_distribution) - 1.0) < 1e-6
    assert len(feat.top_categories) > 0


def test_train_test_split():
    df = _make_raw_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = filter_users(df, min_days=1, min_visits=1)
    trajs = build_day_trajectories(df)
    train, test = train_test_split(trajs, train_ratio=0.8)
    total = len(train) + len(test)
    assert total == len(trajs)
    assert len(train) > len(test)


def test_save_and_load_trajectories():
    df = _make_raw_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = filter_users(df, min_days=1, min_visits=1)
    trajs = build_day_trajectories(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trajs.json"
        save_trajectories(trajs, path)
        loaded = load_trajectories(path)
        assert len(loaded) == len(trajs)
        assert loaded[0].user_id == trajs[0].user_id
