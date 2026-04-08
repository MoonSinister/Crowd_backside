"""
training/profile.py — Individual profile builder (offline phase, Stage 1).

Implements the paper's activity profile assignment:
1. Pre-define social role set R = {student, office_worker, ...}
2. For each user, extract trajectory statistics (top POI types, hour distribution)
3. Call LLM to generate a candidate activity pattern for each role
4. Score each candidate against the user's actual trajectory history
5. Assign the role whose pattern best aligns with the observed behaviour

The resulting AgentProfile is used to initialise agents in simulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.poi import POICategory, raw_type_to_category
from data.trajectory import DayTrajectory, UserFeatures
from agent.state import AgentProfile, SOCIAL_ROLES


# ─────────────────────────────────────────────────────────────────────────────
# Role templates (used as LLM prompt priors)
# ─────────────────────────────────────────────────────────────────────────────

ROLE_DESCRIPTIONS: Dict[str, str] = {
    "student": (
        "A university or school student. Frequently visits educational facilities, "
        "cafes, convenience stores, and parks. Active mostly during daytime. "
        "Tends to dine out for lunch and evening meals."
    ),
    "office_worker": (
        "A full-time office employee working standard business hours (9–18). "
        "Commutes to an office district. Frequently visits dining and convenience "
        "locations during lunch breaks and after work."
    ),
    "teacher": (
        "A school or university teacher following an academic schedule. "
        "Regularly visits educational POIs, dining spots, and public transport. "
        "May share shopping or dining habits with office workers."
    ),
    "night_shift_worker": (
        "Works a non-standard schedule, often at night or in early morning hours. "
        "Visits convenience stores at unusual hours. Resting and residential "
        "activities dominate daytime. Transport and retail interactions at night."
    ),
    "delivery_rider": (
        "A food or package delivery rider. Highly mobile — frequently visits "
        "restaurants (pickup) and residential/office addresses (dropoff). "
        "Activity peaks at meal times. Income and order count drive decisions."
    ),
    "retail_employee": (
        "Works in a retail shop or market. Regular daytime schedule. "
        "Frequently visits shopping, convenience, and dining POIs. "
        "Shares some transport patterns with office workers."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_role_prompt(role: str, user_stats: Dict[str, Any]) -> str:
    """
    Prompt asking the LLM to describe the expected activity pattern for a given
    role, conditioned on the user's observed statistics.
    """
    role_desc = ROLE_DESCRIPTIONS.get(role, "An urban resident.")
    top_cats = user_stats.get("top_categories", [])
    cat_str = ", ".join(f"{c} ({v:.1%})" for c, v in top_cats)
    hour_dist = user_stats.get("hourly_distribution", [])
    peak_hours = sorted(range(24), key=lambda h: -hour_dist[h])[:4] if hour_dist else []
    peak_str = ", ".join(f"{h:02d}:00" for h in sorted(peak_hours))

    return (
        f"Role: {role}\n"
        f"Role description: {role_desc}\n\n"
        f"Observed user statistics:\n"
        f"  Top activity categories: {cat_str}\n"
        f"  Peak activity hours: {peak_str}\n"
        f"  Total days observed: {user_stats.get('num_days', 0)}\n\n"
        f"Task: Write a concise activity pattern summary (2-3 sentences) that "
        f"describes this user's typical daily behaviour if they are a {role}. "
        f"Focus on what POI types they visit and when."
    )


def _build_alignment_prompt(
    candidate_pattern: str,
    trajectory_sample: str,
) -> str:
    """
    Prompt asking the LLM to score how well a candidate pattern matches a
    real trajectory sample (returns a score 0-10).
    """
    return (
        f"Candidate activity pattern:\n{candidate_pattern}\n\n"
        f"Observed trajectory sample:\n{trajectory_sample}\n\n"
        f"Rate how well the candidate pattern matches the observed trajectory "
        f"on a scale from 0 (no match) to 10 (perfect match). "
        f'Respond with JSON: {{"score": <integer 0-10>, "reason": "<brief reason>"}}'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Profile builder
# ─────────────────────────────────────────────────────────────────────────────

class ProfileBuilder:
    """
    Assigns an AgentProfile to each user by selecting the social role whose
    LLM-generated activity pattern best aligns with the user's trajectory.

    Parameters
    ----------
    llm_backend : LLMBackend
        Any LLMBackend implementation (see agent.intent).
    roles : list[str]
        Social roles to consider. Defaults to SOCIAL_ROLES.
    """

    def __init__(
        self,
        llm_backend: Any,
        roles: Optional[List[str]] = None,
    ) -> None:
        self._llm = llm_backend
        self._roles = roles or SOCIAL_ROLES

    def build_profile(
        self,
        user_id: str,
        features: UserFeatures,
        trajectories: List[DayTrajectory],
    ) -> AgentProfile:
        """
        Build an AgentProfile for a single user.

        Steps:
        1. Generate a candidate activity pattern for each role via LLM.
        2. Score each pattern against the user's trajectory sample.
        3. Select the highest-scoring role.
        """
        user_stats = features.to_dict()
        trajectory_sample = self._format_trajectory_sample(trajectories)

        best_role = self._roles[0]
        best_score = -1.0
        best_description = ""
        role_patterns: Dict[str, str] = {}

        for role in self._roles:
            # Step 1: generate candidate pattern
            role_prompt = _build_role_prompt(role, user_stats)
            pattern = self._llm.generate(role_prompt, max_new_tokens=200).strip()
            role_patterns[role] = pattern

            # Step 2: score alignment with real trajectory
            score, _ = self._score_alignment(pattern, trajectory_sample)
            if score > best_score:
                best_score = score
                best_role = role
                best_description = pattern

        return AgentProfile(
            role=best_role,
            top_categories=features.top_categories,
            hourly_distribution=features.hourly_distribution,
            centroid_lat=features.centroid_lat,
            centroid_lon=features.centroid_lon,
            description=best_description,
        )

    def build_profiles_batch(
        self,
        user_features: List[UserFeatures],
        trajectories_by_user: Dict[str, List[DayTrajectory]],
    ) -> Dict[str, AgentProfile]:
        """Build profiles for all users and return a mapping user_id → profile."""
        profiles: Dict[str, AgentProfile] = {}
        for feat in user_features:
            uid = feat.user_id
            user_trajs = trajectories_by_user.get(uid, [])
            profiles[uid] = self.build_profile(uid, feat, user_trajs)
            print(f"  User {uid}: role={profiles[uid].role}")
        return profiles

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_trajectory_sample(
        trajectories: List[DayTrajectory],
        max_days: int = 5,
    ) -> str:
        """Format a few days of trajectory as readable text for the LLM."""
        lines = []
        for traj in trajectories[:max_days]:
            day_line = f"Date: {traj.date}"
            events = ", ".join(
                f"{e.time_str()} {e.poi_type}"
                for e in traj.events
            )
            lines.append(f"{day_line}  |  {events}")
        return "\n".join(lines) if lines else "No trajectory data available."

    def _score_alignment(
        self,
        candidate_pattern: str,
        trajectory_sample: str,
    ) -> Tuple[float, str]:
        """Ask LLM to score the alignment; fall back to 5.0 on parse error."""
        prompt = _build_alignment_prompt(candidate_pattern, trajectory_sample)
        response = self._llm.generate(prompt, max_new_tokens=100)

        import re
        import json as _json

        try:
            match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
            if match:
                data = _json.loads(match.group())
                score = float(data.get("score", 5))
                reason = data.get("reason", "")
                return score, reason
        except Exception:
            pass
        return 5.0, ""


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_profiles(profiles: Dict[str, AgentProfile], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        uid: {
            "role": p.role,
            "top_categories": p.top_categories,
            "hourly_distribution": p.hourly_distribution,
            "centroid_lat": p.centroid_lat,
            "centroid_lon": p.centroid_lon,
            "description": p.description,
        }
        for uid, p in profiles.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_profiles(path: str | Path) -> Dict[str, AgentProfile]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        uid: AgentProfile(
            role=v["role"],
            top_categories=[(c, s) for c, s in v["top_categories"]],
            hourly_distribution=v["hourly_distribution"],
            centroid_lat=v["centroid_lat"],
            centroid_lon=v["centroid_lon"],
            description=v["description"],
        )
        for uid, v in data.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    import yaml
    from data.trajectory import load_trajectories, load_user_features
    from agent.intent import StubBackend

    parser = argparse.ArgumentParser(description="Build agent profiles")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stub", action="store_true", help="Use stub LLM for testing")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed = Path(cfg["data"]["processed_dir"])
    features = load_user_features(processed / "user_features.json")
    train_trajs = load_trajectories(processed / "trajectories_train.json")

    trajs_by_user: Dict[str, List[DayTrajectory]] = {}
    for t in train_trajs:
        trajs_by_user.setdefault(t.user_id, []).append(t)

    if args.stub:
        llm = StubBackend()
    else:
        from agent.intent import TransformersBackend
        llm = TransformersBackend(
            model_path=cfg["llm"]["base_model"],
            device_map=cfg["llm"]["device_map"],
            load_in_4bit=cfg["llm"]["load_in_4bit"],
        )

    builder = ProfileBuilder(llm_backend=llm, roles=cfg["agent"]["roles"])
    print(f"Building profiles for {len(features)} users ...")
    profiles = builder.build_profiles_batch(features, trajs_by_user)
    out_path = processed / "profiles.json"
    save_profiles(profiles, out_path)
    print(f"Saved {len(profiles)} profiles to {out_path}")


if __name__ == "__main__":
    _main()
