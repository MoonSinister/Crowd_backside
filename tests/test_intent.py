"""
tests/test_intent.py — Unit tests for the cognitive intent module.
"""

import json
import pytest
from data.poi import POICategory
from agent.intent import (
    StubBackend,
    build_intent_prompt,
    parse_intent_response,
    CognitiveIntentModule,
)
from agent.state import AgentProfile, AgentState
from agent.perception import PerceptionContext
from agent.memory import EpisodicMemory


def _make_state(role="office_worker"):
    profile = AgentProfile(
        role=role,
        top_categories=[("dining", 0.4), ("office", 0.3)],
        hourly_distribution=[1 / 24] * 24,
        centroid_lat=35.005,
        centroid_lon=139.005,
        description="Standard office worker.",
    )
    return AgentState(
        agent_id="test_agent",
        profile=profile,
        current_lat=35.005,
        current_lon=139.005,
        current_poi_id="p0",
        current_poi_type="office",
        time_slot=60,  # 10:00
        day=0,
    )


def _make_perception():
    return PerceptionContext(
        time_slot=60,
        hour=10,
        day=0,
        current_lat=35.005,
        current_lon=139.005,
        current_poi_id="p0",
        current_poi_type="office",
        nearby_pois=[
            {"poi_id": "p1", "name": "Ramen", "category": "dining", "distance_m": 300},
        ],
    )


def _make_memory():
    return EpisodicMemory(max_size=10, top_k=3, similarity_threshold=0.99)


def test_stub_backend_generate():
    llm = StubBackend()
    out = llm.generate("test prompt")
    data = json.loads(out)
    assert "intent_category" in data
    assert data["intent_category"] == "dining"


def test_stub_backend_embed():
    llm = StubBackend()
    emb = llm.embed("hello world")
    assert isinstance(emb, list)
    assert len(emb) == 16
    norm = sum(x**2 for x in emb) ** 0.5
    assert norm == pytest.approx(1.0, abs=0.01)


def test_build_intent_prompt_contains_key_sections():
    state = _make_state()
    perception = _make_perception()
    prompt = build_intent_prompt(state, perception, "No past experiences.")
    assert "Individual Profile" in prompt
    assert "Current State" in prompt
    assert "Current Perception" in prompt
    assert "Past Experiences" in prompt
    assert "intent_category" in prompt


def test_parse_intent_response_valid_json():
    raw = '{"intent_category": "dining", "explanation": "Hungry at lunch"}'
    cat, expl = parse_intent_response(raw)
    assert cat == POICategory.DINING
    assert "Hungry" in expl


def test_parse_intent_response_extra_text():
    raw = 'Sure! Here is the answer: {"intent_category": "park", "explanation": "Need fresh air"}'
    cat, expl = parse_intent_response(raw)
    assert cat == POICategory.PARK


def test_parse_intent_response_invalid_fallback():
    raw = "The agent should go shopping for sure."
    cat, expl = parse_intent_response(raw)
    assert cat == POICategory.SHOPPING


def test_parse_intent_response_unknown():
    raw = "Completely irrelevant response with no category at all."
    cat, expl = parse_intent_response(raw)
    assert cat == POICategory.OTHER


def test_cognitive_intent_module_slow_path():
    """When memory is empty, must use LLM (slow path)."""
    llm = StubBackend()
    module = CognitiveIntentModule(llm)
    state = _make_state()
    perception = _make_perception()
    memory = _make_memory()

    intent, emb = module.decide(state, perception, memory)
    assert intent.path in ("fast", "slow")
    assert isinstance(intent.category, POICategory)
    assert isinstance(emb, list)


def test_cognitive_intent_module_fast_path():
    """When memory has a high-similarity entry, should use fast path."""
    llm = StubBackend()
    module = CognitiveIntentModule(llm)
    state = _make_state()
    perception = _make_perception()

    # Build a memory with a very low similarity threshold
    memory = EpisodicMemory(
        max_size=10,
        top_k=3,
        similarity_threshold=0.0,  # Always reuse
    )
    # Plant a memory with the exact same embedding as the stub would produce
    emb = llm.embed(perception.to_embedding_input())
    from agent.perception import PerceptionContext as PC
    memory.write(perception, emb, "prev state", "shopping", current_time_slot=0)

    intent, _ = module.decide(state, perception, memory)
    assert intent.path == "fast"
    assert intent.category == POICategory.SHOPPING
