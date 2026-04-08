"""
agent/memory.py — M: Episodic memory module.

Implements the paper's episodic memory mechanism:

  score(m | p_t) = cosine_sim(p_t, p_m) · exp(-lambda · (t - t_m))

Memory entries are stored as triples  m = (p, s, I):
  - p : PerceptionContext at the time the memory was formed
  - s : AgentState snapshot
  - I : Intent (string) generated at that step

A new memory is written after every decision step.  When the perceived
environment changes significantly (Δ > ε), old memories whose perception
differs are marked as expired and excluded from future retrieval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.perception import PerceptionContext


# ─────────────────────────────────────────────────────────────────────────────
# Memory entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """
    A single episodic memory entry: m = (p, s_summary, I).

    Attributes
    ----------
    perception_text : str
        Compact text snapshot of pₜ used for embedding-based retrieval.
    perception_embedding : list[float]
        Embedding vector of the perception text (set externally).
    state_summary : str
        Brief text snapshot of the agent's state when memory was formed.
    intent : str
        The intent text Iₜ generated at this step.
    time_slot : int
        Absolute simulation time slot when memory was created.
    active : bool
        False once the memory has been expired by an environment change.
    """

    perception_text: str
    perception_embedding: List[float]
    state_summary: str
    intent: str
    time_slot: int
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perception_text": self.perception_text,
            "state_summary": self.state_summary,
            "intent": self.intent,
            "time_slot": self.time_slot,
            "active": self.active,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Memory module
# ─────────────────────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Per-agent episodic memory store with time-decay retrieval.

    Parameters
    ----------
    max_size : int
        Maximum number of memories to retain (oldest are evicted when full).
    top_k : int
        Number of memories returned by :meth:`retrieve`.
    time_decay_lambda : float
        λ in the score formula.  Larger values give stronger recency bias.
    env_change_epsilon : float
        If cosine distance between current and historical perception exceeds
        this threshold, the historical memory is marked as expired.
    similarity_threshold : float
        If the best memory score exceeds this value, the agent can reuse the
        historical intent directly without calling the LLM.
    """

    def __init__(
        self,
        max_size: int = 200,
        top_k: int = 5,
        time_decay_lambda: float = 0.05,
        env_change_epsilon: float = 0.3,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._max_size = max_size
        self._top_k = top_k
        self._lambda = time_decay_lambda
        self._epsilon = env_change_epsilon
        self._sim_threshold = similarity_threshold
        self._memories: List[MemoryEntry] = []

    # ── Core operations ────────────────────────────────────────────────────

    def write(
        self,
        perception: PerceptionContext,
        embedding: List[float],
        state_summary: str,
        intent: str,
        current_time_slot: int,
    ) -> None:
        """
        Store a new memory entry and optionally expire stale memories.

        Steps:
        1. Compute distance between new perception and existing active memories.
        2. Expire those whose distance > ε (environment changed significantly).
        3. Append new memory.
        4. Evict oldest memories if capacity exceeded.
        """
        new_emb = np.array(embedding, dtype=np.float32)

        for mem in self._memories:
            if not mem.active:
                continue
            mem_emb = np.array(mem.perception_embedding, dtype=np.float32)
            dist = _cosine_distance(new_emb, mem_emb)
            if dist > self._epsilon:
                mem.active = False  # Expire — environment has changed

        entry = MemoryEntry(
            perception_text=perception.to_embedding_input(),
            perception_embedding=embedding,
            state_summary=state_summary,
            intent=intent,
            time_slot=current_time_slot,
        )
        self._memories.append(entry)

        # Evict oldest memories beyond capacity
        if len(self._memories) > self._max_size:
            self._memories = self._memories[-self._max_size:]

    def retrieve(
        self,
        query_embedding: List[float],
        current_time_slot: int,
    ) -> List[MemoryEntry]:
        """
        Return the Top-K active memories ranked by the time-decay score:

            score(m | p_t) = cosine_sim(p_t, p_m) · exp(-λ · (t - t_m))
        """
        q = np.array(query_embedding, dtype=np.float32)
        scored: List[Tuple[float, MemoryEntry]] = []
        for mem in self._memories:
            if not mem.active:
                continue
            mem_emb = np.array(mem.perception_embedding, dtype=np.float32)
            sim = _cosine_similarity(q, mem_emb)
            age = max(0, current_time_slot - mem.time_slot)
            score = sim * math.exp(-self._lambda * age)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[: self._top_k]]

    def can_reuse(
        self,
        query_embedding: List[float],
        current_time_slot: int,
    ) -> Optional[str]:
        """
        If the best-scoring memory exceeds the similarity threshold,
        return its intent directly (experience-reuse / fast path).
        Otherwise return None to trigger LLM reasoning.
        """
        q = np.array(query_embedding, dtype=np.float32)
        best_score = 0.0
        best_intent: Optional[str] = None
        for mem in self._memories:
            if not mem.active:
                continue
            mem_emb = np.array(mem.perception_embedding, dtype=np.float32)
            sim = _cosine_similarity(q, mem_emb)
            age = max(0, current_time_slot - mem.time_slot)
            score = sim * math.exp(-self._lambda * age)
            if score > best_score:
                best_score = score
                best_intent = mem.intent
        if best_score >= self._sim_threshold and best_intent is not None:
            return best_intent
        return None

    def to_prompt_text(self, memories: List[MemoryEntry]) -> str:
        """Format retrieved memories as structured text for LLM prompt."""
        if not memories:
            return "No relevant past experiences."
        lines = ["Past experiences (most relevant first):"]
        for i, mem in enumerate(memories, 1):
            lines.append(
                f"  [{i}] Context: {mem.perception_text}\n"
                f"       State: {mem.state_summary}\n"
                f"       Intent taken: {mem.intent}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return sum(1 for m in self._memories if m.active)


# ─────────────────────────────────────────────────────────────────────────────
# Vector utilities
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - _cosine_similarity(a, b)
