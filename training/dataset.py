"""
training/dataset.py — Algorithm 1: DPO/SFT dataset construction.

For each user u and each decision time step t in the training trajectories:
  1. Construct input context xₜ = (sₜ, pₜ, M̃ₜ)
  2. Call LLM to generate K candidate intents {I¹, ..., Iᴷ}
  3. Compare each candidate against the real trajectory label (POI category)
       - Iʷ (preferred): category matches the actual next POI category
       - Iˡ (rejected): does not match
  4. Add (xₜ, Iʷ, Iˡ) to DPO dataset
  5. Add (xₜ, Iʷ) to SFT dataset

Output JSON schemas:

  SFT dataset:
    [{"prompt": "<system + user prompt>", "completion": "<intent JSON>"}, ...]

  DPO dataset:
    [{"prompt": "<system + user prompt>",
      "chosen": "<preferred intent JSON>",
      "rejected": "<rejected intent JSON>"}, ...]
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.poi import POICategory, raw_type_to_category
from data.trajectory import DayTrajectory, VisitEvent
from agent.state import AgentProfile, AgentState
from agent.perception import PerceptionContext
from agent.memory import EpisodicMemory
from agent.intent import (
    LLMBackend,
    build_intent_prompt,
    parse_intent_response,
    Intent,
)
from training.profile import load_profiles


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_perception_from_event(
    event: VisitEvent,
    nearby_pois: Optional[List[Dict[str, Any]]] = None,
) -> PerceptionContext:
    """Build a minimal PerceptionContext from a trajectory visit event."""
    return PerceptionContext(
        time_slot=event.time_slot,
        hour=event.timestamp.hour,
        day=0,
        current_lat=event.lat,
        current_lon=event.lon,
        current_poi_id=event.poi_id,
        current_poi_type=event.poi_type,
        nearby_pois=nearby_pois or [],
        nearby_agent_count=0,
    )


def _intent_matches_label(intent_category: POICategory, label_category: POICategory) -> bool:
    """
    Consistency function φ from the paper:
    Returns True if the intent category maps to the same functional space as
    the real trajectory label.
    """
    return intent_category == label_category


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Constructs SFT and DPO training datasets from historical trajectories.

    Parameters
    ----------
    llm_backend : LLMBackend
        LLM used to generate candidate intents.
    profiles : dict[str, AgentProfile]
        Per-user profiles produced by ProfileBuilder.
    num_candidates : int
        K candidate intents generated per decision step.
    memory_cfg : dict
        Memory module configuration (max_size, top_k, lambda, epsilon, threshold).
    """

    def __init__(
        self,
        llm_backend: LLMBackend,
        profiles: Dict[str, AgentProfile],
        num_candidates: int = 5,
        memory_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._llm = llm_backend
        self._profiles = profiles
        self._k = num_candidates
        self._mem_cfg = memory_cfg or {}

    def build(
        self,
        trajectories: List[DayTrajectory],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process all trajectories and return (sft_records, dpo_records).

        Each record is a dict matching the HuggingFace TRL dataset format.
        """
        sft_records: List[Dict] = []
        dpo_records: List[Dict] = []

        # Group trajectories by user
        by_user: Dict[str, List[DayTrajectory]] = defaultdict(list)
        for t in trajectories:
            by_user[t.user_id].append(t)

        for user_id, user_trajs in by_user.items():
            profile = self._profiles.get(user_id)
            if profile is None:
                continue  # Skip users without a profile

            user_trajs.sort(key=lambda t: t.date)
            user_sft, user_dpo = self._process_user(user_id, profile, user_trajs)
            sft_records.extend(user_sft)
            dpo_records.extend(user_dpo)

        return sft_records, dpo_records

    # ── Per-user processing ────────────────────────────────────────────────

    def _process_user(
        self,
        user_id: str,
        profile: AgentProfile,
        trajectories: List[DayTrajectory],
    ) -> Tuple[List[Dict], List[Dict]]:
        from agent.memory import EpisodicMemory

        sft_records: List[Dict] = []
        dpo_records: List[Dict] = []

        memory = EpisodicMemory(
            max_size=self._mem_cfg.get("max_size", 200),
            top_k=self._mem_cfg.get("top_k", 5),
            time_decay_lambda=self._mem_cfg.get("time_decay_lambda", 0.05),
            env_change_epsilon=self._mem_cfg.get("env_change_epsilon", 0.3),
            similarity_threshold=self._mem_cfg.get("similarity_threshold", 0.85),
        )

        global_slot = 0  # Running absolute time slot across days

        for day_idx, day_traj in enumerate(trajectories):
            events = day_traj.events
            if len(events) < 2:
                global_slot += 144
                continue

            for step_idx in range(len(events) - 1):
                current_event = events[step_idx]
                next_event = events[step_idx + 1]

                # Ground-truth label: category of next POI
                true_label = raw_type_to_category(next_event.poi_type)

                # Build agent state
                state = AgentState(
                    agent_id=user_id,
                    profile=profile,
                    current_lat=current_event.lat,
                    current_lon=current_event.lon,
                    current_poi_id=current_event.poi_id,
                    current_poi_type=current_event.poi_type,
                    time_slot=current_event.time_slot,
                    day=day_idx,
                )

                # Build perception context
                perception = _make_perception_from_event(current_event)
                current_slot = global_slot + current_event.time_slot

                # Retrieve memories
                query_text = perception.to_embedding_input()
                query_emb = self._llm.embed(query_text)
                retrieved = memory.retrieve(query_emb, current_slot)
                memory_text = memory.to_prompt_text(retrieved)

                # Build prompt
                prompt = build_intent_prompt(state, perception, memory_text)

                # Generate K candidate intents
                candidates = self._generate_candidates(prompt)

                # Partition into preferred / rejected
                preferred = [
                    c for c in candidates
                    if _intent_matches_label(c[0], true_label)
                ]
                rejected = [
                    c for c in candidates
                    if not _intent_matches_label(c[0], true_label)
                ]

                if preferred and rejected:
                    chosen_cat, chosen_expl = preferred[0]
                    rejected_cat, rejected_expl = rejected[0]

                    chosen_json = json.dumps(
                        {"intent_category": chosen_cat.value, "explanation": chosen_expl}
                    )
                    rejected_json = json.dumps(
                        {"intent_category": rejected_cat.value, "explanation": rejected_expl}
                    )

                    dpo_records.append({
                        "prompt": prompt,
                        "chosen": chosen_json,
                        "rejected": rejected_json,
                    })

                if preferred:
                    chosen_cat, chosen_expl = preferred[0]
                    completion_json = json.dumps(
                        {"intent_category": chosen_cat.value, "explanation": chosen_expl}
                    )
                    sft_records.append({
                        "prompt": prompt,
                        "completion": completion_json,
                    })

                # Write memory (use preferred intent if available)
                intent_str = preferred[0][0].value if preferred else (
                    candidates[0][0].value if candidates else true_label.value
                )
                memory.write(
                    perception=perception,
                    embedding=query_emb,
                    state_summary=state.to_prompt_text(),
                    intent=intent_str,
                    current_time_slot=current_slot,
                )

            global_slot += 144  # Advance by one day's worth of slots

        return sft_records, dpo_records

    def _generate_candidates(
        self, prompt: str
    ) -> List[Tuple[POICategory, str]]:
        """Generate K candidate intents by calling the LLM K times."""
        candidates: List[Tuple[POICategory, str]] = []
        for _ in range(self._k):
            raw = self._llm.generate(prompt, max_new_tokens=200)
            cat, expl = parse_intent_response(raw)
            candidates.append((cat, expl))
        return candidates


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset(records: List[Dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_dataset(path: str | Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    import yaml
    from data.trajectory import load_trajectories

    parser = argparse.ArgumentParser(description="Build SFT and DPO datasets")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed = Path(cfg["data"]["processed_dir"])
    train_trajs = load_trajectories(processed / "trajectories_train.json")
    profiles = load_profiles(processed / "profiles.json")

    if args.stub:
        from agent.intent import StubBackend
        llm = StubBackend()
    else:
        from agent.intent import TransformersBackend
        llm = TransformersBackend(
            model_path=cfg["llm"]["base_model"],
            device_map=cfg["llm"]["device_map"],
            load_in_4bit=cfg["llm"]["load_in_4bit"],
        )

    builder = DatasetBuilder(
        llm_backend=llm,
        profiles=profiles,
        num_candidates=cfg["agent"]["intent"]["num_candidates"],
        memory_cfg=cfg["agent"]["memory"],
    )
    print(f"Building dataset from {len(train_trajs)} day-trajectories ...")
    sft_records, dpo_records = builder.build(train_trajs)
    print(f"  SFT records: {len(sft_records)}, DPO records: {len(dpo_records)}")

    save_dataset(sft_records, cfg["data"]["sft_dataset_path"])
    save_dataset(dpo_records, cfg["data"]["dpo_dataset_path"])
    print("Done.")


if __name__ == "__main__":
    _main()
