"""
simulation/runner.py — Multi-agent simulation orchestrator.

Implements Algorithm 2 from the paper:

  Offline phase  (handled by training/*)
  Online phase:
    Initialise agents from profiles and city environment.
    for t = 1 to T:
      for each agent a_i:
        Perceive environment → retrieve memories
        Call intent model → generate high-level intent I_t
        Execute movement → update state and memory
      Advance environment clock

Repast4Py is an optional dependency.  When it is available the simulation
runs inside Repast's scheduler.  When it is not installed (e.g. in unit-test
environments without MPI), the runner falls back to a pure-Python loop.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.poi import SpatialGrid, load_pois_from_json
from agent.state import AgentProfile, AgentState
from agent.perception import PerceptionModule
from agent.memory import EpisodicMemory
from agent.intent import CognitiveIntentModule, LLMBackend
from agent.behavior import BehaviorModule
from simulation.environment import CityEnvironment, TrajectoryRecord
from training.profile import load_profiles


# ─────────────────────────────────────────────────────────────────────────────
# Single-agent wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CrowdAgent:
    """
    Wraps all five TIMB modules for one simulated agent.

    Attributes map to the paper's five-tuple:
      S → state
      P → perception_module
      M → memory
      C → intent_module
      A → behavior_module
    """

    def __init__(
        self,
        state: AgentState,
        perception_module: PerceptionModule,
        memory: EpisodicMemory,
        intent_module: CognitiveIntentModule,
        behavior_module: BehaviorModule,
    ) -> None:
        self.state = state
        self._perception = perception_module
        self.memory = memory
        self._intent = intent_module
        self._behavior = behavior_module

    def step(self, env: CityEnvironment) -> Optional[TrajectoryRecord]:
        """
        Execute one simulation time step for this agent.

        Returns a TrajectoryRecord if the agent moved, None otherwise.
        """
        # P: Perceive
        nearby_count = env.count_agents_near(
            self.state.current_lat,
            self.state.current_lon,
        )
        perception = self._perception.perceive(self.state, nearby_count)

        # C: Generate intent (fast or slow path)
        intent, query_emb = self._intent.decide(
            self.state, perception, self.memory
        )

        # A: Execute movement and write memory
        destination = self._behavior.execute_and_update(
            state=self.state,
            perception=perception,
            intent=intent,
            query_embedding=query_emb,
            memory=self.memory,
        )

        if destination is None:
            return None

        # Update environment registry
        env.update_agent_position(
            self.state.agent_id,
            self.state.current_lat,
            self.state.current_lon,
        )
        self.state.advance_time()

        return TrajectoryRecord(
            agent_id=self.state.agent_id,
            day=env.current_day,
            time_slot=env.current_slot,
            lat=self.state.current_lat,
            lon=self.state.current_lon,
            poi_id=self.state.current_poi_id,
            poi_type=self.state.current_poi_type,
            intent_category=intent.category.value,
            intent_explanation=intent.explanation,
            intent_path=intent.path,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

class SimulationRunner:
    """
    Orchestrates the full multi-agent simulation (Algorithm 2, online phase).

    Parameters
    ----------
    cfg : dict
        Full configuration dict (loaded from config.yaml).
    llm_backend : LLMBackend
        The DPO-aligned intent reasoning model.
    """

    def __init__(self, cfg: dict, llm_backend: LLMBackend) -> None:
        self._cfg = cfg
        self._llm = llm_backend
        self._sim_cfg = cfg["simulation"]
        self._agent_cfg = cfg["agent"]

    def run(self) -> None:
        """Execute the full simulation and save output trajectories."""
        processed = Path(self._cfg["data"]["processed_dir"])
        output_dir = Path(self._sim_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Load city environment ──────────────────────────────────────────
        print("Loading POI data ...")
        pois = load_pois_from_json(processed / "pois.json")
        grid = SpatialGrid(pois)
        env = CityEnvironment(
            spatial_grid=grid,
            step_minutes=self._sim_cfg["step_minutes"],
        )

        # ── Build shared modules ───────────────────────────────────────────
        mem_cfg = self._agent_cfg["memory"]
        beh_cfg = self._agent_cfg["behavior"]

        perception_module = PerceptionModule(
            spatial_grid=grid,
            perception_radius_m=beh_cfg.get("search_radius_m", 3000.0),
            max_nearby_pois=beh_cfg["max_candidate_pois"],
        )
        intent_module = CognitiveIntentModule(llm_backend=self._llm)
        behavior_module = BehaviorModule(
            spatial_grid=grid,
            distance_decay_beta=beh_cfg["distance_decay_beta"],
            max_candidate_pois=beh_cfg["max_candidate_pois"],
        )

        # ── Initialise agents ─────────────────────────────────────────────
        print("Initialising agents ...")
        profiles = load_profiles(processed / "profiles.json")
        agents = self._create_agents(
            profiles=profiles,
            pois=pois,
            perception_module=perception_module,
            intent_module=intent_module,
            behavior_module=behavior_module,
            mem_cfg=mem_cfg,
        )
        for agent in agents:
            env.register_agent(
                agent.state.agent_id,
                agent.state.current_lat,
                agent.state.current_lon,
            )
        print(f"  Spawned {len(agents)} agents.")

        # ── Main simulation loop ───────────────────────────────────────────
        num_steps = self._sim_cfg["num_steps"]
        print(f"Running {num_steps} simulation steps ...")

        try:
            self._run_loop(agents, env, num_steps)
        except KeyboardInterrupt:
            print("Simulation interrupted by user.")

        # ── Save results ───────────────────────────────────────────────────
        out_path = output_dir / "trajectories.json"
        env.save_trajectory_log(out_path)
        print(f"Simulation complete.  Output: {out_path}")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _create_agents(
        self,
        profiles: Dict[str, AgentProfile],
        pois,
        perception_module: PerceptionModule,
        intent_module: CognitiveIntentModule,
        behavior_module: BehaviorModule,
        mem_cfg: Dict,
    ) -> List[CrowdAgent]:
        rng = random.Random(self._sim_cfg.get("random_seed", 42))
        num_agents = self._sim_cfg["num_agents"]

        profile_list = list(profiles.values())
        if not profile_list:
            raise ValueError(
                "No profiles found.  Run 'python -m training.profile' first."
            )

        agents: List[CrowdAgent] = []
        for idx in range(num_agents):
            # Cycle through available profiles
            profile = profile_list[idx % len(profile_list)]

            # Spawn at a random POI near the user's centroid
            start_poi = self._find_spawn_poi(
                pois=pois,
                lat=profile.centroid_lat,
                lon=profile.centroid_lon,
                rng=rng,
            )

            state = AgentState(
                agent_id=f"agent_{idx:04d}",
                profile=profile,
                current_lat=start_poi.lat if start_poi else profile.centroid_lat,
                current_lon=start_poi.lon if start_poi else profile.centroid_lon,
                current_poi_id=start_poi.poi_id if start_poi else "",
                current_poi_type=start_poi.raw_type if start_poi else "",
                time_slot=0,
                day=0,
            )

            memory = EpisodicMemory(
                max_size=mem_cfg.get("max_size", 200),
                top_k=mem_cfg.get("top_k", 5),
                time_decay_lambda=mem_cfg.get("time_decay_lambda", 0.05),
                env_change_epsilon=mem_cfg.get("env_change_epsilon", 0.3),
                similarity_threshold=mem_cfg.get("similarity_threshold", 0.85),
            )

            agents.append(CrowdAgent(
                state=state,
                perception_module=perception_module,
                memory=memory,
                intent_module=intent_module,
                behavior_module=behavior_module,
            ))
        return agents

    @staticmethod
    def _find_spawn_poi(pois, lat: float, lon: float, rng: random.Random):
        """Select a random POI within 2 km of (lat, lon) as the spawn location."""
        from geopy.distance import geodesic
        nearby = [
            p for p in pois
            if geodesic((lat, lon), (p.lat, p.lon)).meters <= 2000
        ]
        if nearby:
            return rng.choice(nearby)
        return rng.choice(pois) if pois else None

    def _run_loop(
        self,
        agents: List[CrowdAgent],
        env: CityEnvironment,
        num_steps: int,
    ) -> None:
        """Main simulation loop — pure Python (no Repast4Py dependency)."""
        for step in range(num_steps):
            if step % 10 == 0:
                print(f"  Step {step}/{num_steps}  {env.clock.time_str()}")

            for agent in agents:
                record = agent.step(env)
                if record is not None:
                    env.log_move(record)

            env.step()

    def _run_with_repast(
        self,
        agents: List[CrowdAgent],
        env: CityEnvironment,
        num_steps: int,
    ) -> None:
        """
        Simulation loop using Repast4Py for distributed / MPI execution.

        Falls back to _run_loop if repast4py is not installed.
        """
        try:
            from repast4py import core, schedule, logging as rlog

            runner = schedule.Runner(comm=None)

            def sim_step():
                for agent in agents:
                    record = agent.step(env)
                    if record is not None:
                        env.log_move(record)
                env.step()

            runner.schedule_repeating_event(1, 1, sim_step)
            runner.run(num_steps)
        except ImportError:
            print("repast4py not available — falling back to pure-Python loop.")
            self._run_loop(agents, env, num_steps)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run multi-agent simulation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stub", action="store_true", help="Use stub LLM (for testing)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.stub:
        from agent.intent import StubBackend
        llm = StubBackend()
    else:
        from agent.intent import TransformersBackend
        llm = TransformersBackend(
            model_path=cfg["llm"]["dpo_output_dir"],
            device_map=cfg["llm"]["device_map"],
            load_in_4bit=cfg["llm"]["load_in_4bit"],
        )

    runner = SimulationRunner(cfg=cfg, llm_backend=llm)
    runner.run()


if __name__ == "__main__":
    _main()
