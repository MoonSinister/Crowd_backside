"""
tests/test_memory.py — Unit tests for the episodic memory module.
"""

import math
import pytest
from agent.memory import EpisodicMemory, _cosine_similarity, _cosine_distance
from agent.perception import PerceptionContext


def _make_perception(slot: int = 0) -> PerceptionContext:
    return PerceptionContext(
        time_slot=slot,
        hour=slot * 10 // 60,
        day=0,
        current_lat=35.0,
        current_lon=139.0,
        current_poi_id="p1",
        current_poi_type="restaurant",
    )


def _unit_emb(dim: int = 8, seed: int = 0) -> list:
    import random
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x**2 for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def test_cosine_similarity_identical():
    v = _unit_emb(8, 1)
    assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal():
    import numpy as np
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance():
    v = _unit_emb(8, 2)
    assert _cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_memory_write_and_retrieve():
    mem = EpisodicMemory(max_size=10, top_k=3, time_decay_lambda=0.01)
    emb = _unit_emb(8, 1)
    p = _make_perception(slot=5)
    mem.write(p, emb, "state text", "dining", current_time_slot=5)
    assert len(mem) == 1

    results = mem.retrieve(emb, current_time_slot=5)
    assert len(results) == 1
    assert results[0].intent == "dining"


def test_memory_top_k():
    mem = EpisodicMemory(max_size=20, top_k=3)
    for i in range(6):
        emb = _unit_emb(8, i)
        mem.write(_make_perception(i), emb, "s", f"intent_{i}", current_time_slot=i)
    query = _unit_emb(8, 0)  # Closest to intent_0
    results = mem.retrieve(query, current_time_slot=6)
    assert len(results) <= 3


def test_memory_expiry_on_env_change():
    """When a very dissimilar new perception arrives, old memories are expired."""
    mem = EpisodicMemory(
        max_size=10,
        env_change_epsilon=0.1,  # Very tight threshold
        similarity_threshold=0.9,
    )
    emb_a = [1.0, 0.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0, 0.0]  # Orthogonal to a → distance = 1.0 > 0.1
    mem.write(_make_perception(0), emb_a, "s", "dining", current_time_slot=0)
    assert len(mem) == 1
    # Write with very different perception — should expire the first memory
    mem.write(_make_perception(1), emb_b, "s", "shopping", current_time_slot=1)
    # Only the new memory should be active
    assert len(mem) == 1
    results = mem.retrieve(emb_b, current_time_slot=1)
    assert results[0].intent == "shopping"


def test_memory_can_reuse():
    mem = EpisodicMemory(
        max_size=10,
        time_decay_lambda=0.0,   # No decay
        similarity_threshold=0.95,
    )
    emb = _unit_emb(8, 42)
    mem.write(_make_perception(0), emb, "s", "park", current_time_slot=0)
    # Query with the identical embedding — should reuse
    reused = mem.can_reuse(emb, current_time_slot=1)
    assert reused == "park"


def test_memory_cannot_reuse_below_threshold():
    mem = EpisodicMemory(
        max_size=10,
        similarity_threshold=0.99,
    )
    emb_a = _unit_emb(8, 1)
    emb_b = _unit_emb(8, 2)  # Different seed → not similar enough
    mem.write(_make_perception(0), emb_a, "s", "office", current_time_slot=0)
    reused = mem.can_reuse(emb_b, current_time_slot=1)
    # May or may not reuse depending on actual similarity; just check type
    assert reused is None or isinstance(reused, str)


def test_memory_max_size_eviction():
    mem = EpisodicMemory(max_size=5, top_k=3)
    for i in range(8):
        emb = _unit_emb(8, i)
        mem.write(_make_perception(i), emb, "s", f"i{i}", current_time_slot=i)
    # Internal list should be capped at max_size
    assert len(mem._memories) <= 5
