"""
agent/intent.py — C: Cognitive intent module (LLM-driven).

Implements the paper's two-path cognitive decision process:

  Fast path (experience reuse):
    If memory similarity score ≥ threshold → reuse historical intent.

  Slow path (LLM reasoning):
    Build prompt from {profile, state, perception, retrieved_memories} →
    call LLM → parse structured intent Iₜ.

The intent is expressed as a POICategory label (e.g. "dining", "shopping")
plus an optional natural-language explanation.  This label is then passed to
the behavior module for gravity-model location sampling.

The LLM is accessed through a pluggable LLMBackend interface so that the same
code works with:
  - A local HuggingFace model loaded via Transformers (LlamaCppBackend /
    TransformersBackend)
  - An OpenAI-compatible API endpoint (ApiBackend)
  - A stub for unit tests (StubBackend)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from data.poi import POICategory
from agent.state import AgentState
from agent.perception import PerceptionContext
from agent.memory import EpisodicMemory, MemoryEntry


# ─────────────────────────────────────────────────────────────────────────────
# Intent data class
# ─────────────────────────────────────────────────────────────────────────────

class Intent:
    """
    A structured cognitive intent Iₜ.

    Attributes
    ----------
    category : POICategory
        The semantic activity category (maps to a set of candidate POIs).
    explanation : str
        Brief natural-language explanation of why this intent was formed.
    path : str
        "fast" if derived from experience reuse, "slow" if LLM reasoning was used.
    """

    __slots__ = ("category", "explanation", "path")

    def __init__(
        self,
        category: POICategory,
        explanation: str = "",
        path: str = "slow",
    ) -> None:
        self.category = category
        self.explanation = explanation
        self.path = path

    def to_dict(self) -> Dict[str, str]:
        return {
            "category": self.category.value,
            "explanation": self.explanation,
            "path": self.path,
        }

    def __str__(self) -> str:
        return f"Intent(category={self.category.value}, path={self.path})"


# ─────────────────────────────────────────────────────────────────────────────
# LLM backend interface
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """Abstract interface for LLM inference backends."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text given a prompt string and return the completion."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return a fixed-length embedding vector for the given text."""


class StubBackend(LLMBackend):
    """
    Deterministic stub backend for unit tests — does not require a GPU.

    Always returns the most common category ("dining") with a canned reason.
    """

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        return json.dumps({
            "intent_category": "dining",
            "explanation": "Stub: default dining intent for testing.",
        })

    def embed(self, text: str) -> List[float]:
        # Hash-based pseudo-embedding (reproducible, dimension=16 floats from sha512)
        import hashlib
        import struct
        h = hashlib.sha512(text.encode()).digest()  # 64 bytes → 16 float32s
        floats = [struct.unpack("f", h[i: i + 4])[0] for i in range(0, 64, 4)]
        norm = sum(x ** 2 for x in floats) ** 0.5 or 1.0
        return [x / norm for x in floats]


class TransformersBackend(LLMBackend):
    """
    HuggingFace Transformers inference backend.

    Lazy-loads the model on first use.  Uses the DPO-aligned intent model
    stored at *model_path*.
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        load_in_4bit: bool = True,
    ) -> None:
        self._model_path = model_path
        self._device_map = device_map
        self._load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._embed_model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        quantization_config = None
        if self._load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map=self._device_map,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        self._load()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def embed(self, text: str) -> List[float]:
        """Mean-pool last-hidden-state embeddings from the causal LM."""
        self._load()
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._model.device)
        with torch.no_grad():
            outputs = self._model(
                **inputs,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-1]  # (1, seq_len, dim)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled[0].cpu().float().tolist()


class ApiBackend(LLMBackend):
    """
    OpenAI-compatible API backend (works with vLLM server, local Ollama, etc.).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
        embed_model: Optional[str] = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._embed_model = embed_model or model

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        import httpx

        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def embed(self, text: str) -> List[float]:
        import httpx

        resp = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._embed_model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_VALID_CATEGORIES = [c.value for c in POICategory]
_CATEGORY_LIST = ", ".join(_VALID_CATEGORIES)


def build_intent_prompt(
    state: AgentState,
    perception: PerceptionContext,
    memory_text: str,
) -> str:
    """
    Construct the structured LLM prompt for intent generation.

    The prompt instructs the model to output a JSON object with two keys:
      - intent_category : one of the valid POI categories
      - explanation     : brief reasoning chain (chain-of-thought)
    """
    profile_text = state.profile.to_prompt_text()
    state_text = state.to_prompt_text()
    perception_text = perception.to_prompt_text()

    return f"""You are simulating an urban resident making a movement decision.

=== Individual Profile ===
{profile_text}

=== Current State ===
{state_text}

=== Current Perception ===
{perception_text}

=== Relevant Past Experiences ===
{memory_text}

=== Task ===
Based on the profile, current state, perception, and past experiences, decide
what type of activity/destination the resident should head to next.

Rules:
1. Choose ONE category from: {_CATEGORY_LIST}
2. The choice must be consistent with the profile's role and preferences.
3. The choice must be feasible given the current time and reachable POIs listed above.
4. Respect social norms (no unusual activity at odd hours unless role demands it).

Respond ONLY with valid JSON in this exact format:
{{"intent_category": "<category>", "explanation": "<brief reasoning>"}}"""


def parse_intent_response(response: str) -> Tuple[POICategory, str]:
    """
    Parse the LLM's JSON response into (POICategory, explanation).
    Falls back gracefully if the response is malformed.
    """
    # Try strict JSON parse first
    try:
        # Extract JSON from the response (model may add extra text)
        match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            cat_str = data.get("intent_category", "").strip().lower()
            explanation = data.get("explanation", "")
            try:
                return POICategory(cat_str), explanation
            except ValueError:
                pass
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: scan for the first recognised category keyword
    text_lower = response.lower()
    for cat in POICategory:
        if cat.value in text_lower:
            return cat, response.strip()

    return POICategory.OTHER, response.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Cognitive intent module
# ─────────────────────────────────────────────────────────────────────────────

class CognitiveIntentModule:
    """
    C: Cognitive intent module.

    Decision flow per time step:
      1. Embed current perception → query embedding.
      2. Check experience-reuse threshold (fast path).
         If threshold met → return cached intent (no LLM call).
      3. Retrieve Top-K memories.
      4. Build prompt and call LLM (slow path).
      5. Validate/filter intent against individual constraints.
      6. Return final Intent Iₜ.
    """

    def __init__(self, llm_backend: LLMBackend) -> None:
        self._llm = llm_backend

    def decide(
        self,
        state: AgentState,
        perception: PerceptionContext,
        memory: EpisodicMemory,
    ) -> Tuple[Intent, List[float]]:
        """
        Generate intent Iₜ for the current step.

        Returns
        -------
        intent : Intent
        query_embedding : list[float]
            The embedding of the current perception (used by behavior module
            to write the new memory entry).
        """
        query_text = perception.to_embedding_input()
        query_emb = self._llm.embed(query_text)
        current_slot = perception.time_slot + perception.day * 144

        # ── Fast path: experience reuse ────────────────────────────────────
        reused_intent = memory.can_reuse(query_emb, current_slot)
        if reused_intent is not None:
            try:
                cat = POICategory(reused_intent.strip().lower())
            except ValueError:
                cat = POICategory.OTHER
            return Intent(category=cat, explanation="Reused from memory.", path="fast"), query_emb

        # ── Slow path: LLM reasoning ───────────────────────────────────────
        retrieved = memory.retrieve(query_emb, current_slot)
        memory_text = memory.to_prompt_text(retrieved)
        prompt = build_intent_prompt(state, perception, memory_text)

        raw_response = self._llm.generate(prompt, max_new_tokens=300)
        category, explanation = parse_intent_response(raw_response)

        # Apply individual consistency filter: prefer profile-aligned categories
        category = self._apply_profile_filter(category, state)

        intent = Intent(category=category, explanation=explanation, path="slow")
        return intent, query_emb

    @staticmethod
    def _apply_profile_filter(
        category: POICategory,
        state: AgentState,
    ) -> POICategory:
        """
        Lightweight profile consistency check.
        If the chosen category is completely absent from the profile's top
        preferences, nudge toward the most preferred category.
        This is a soft constraint; it rarely fires unless the LLM hallucinates.
        """
        if not state.profile.top_categories:
            return category
        top_cat_values = {c for c, _ in state.profile.top_categories}
        if category.value in top_cat_values or len(top_cat_values) == 0:
            return category
        # If category is OTHER, replace with the user's top preference
        if category == POICategory.OTHER:
            try:
                return POICategory(state.profile.top_categories[0][0])
            except ValueError:
                pass
        return category
