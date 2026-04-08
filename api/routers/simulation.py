"""
api/routers/simulation.py — Simulation management API endpoints.

POST   /simulations/            Create and start a new simulation run
GET    /simulations/            List all simulation runs
GET    /simulations/{run_id}    Get details of a simulation run
GET    /simulations/{run_id}/trajectories
                                Stream trajectory records for a run
GET    /simulations/{run_id}/intents
                                Aggregate intent statistics for a run
DELETE /simulations/{run_id}    Delete a simulation run and its data
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db import (
    AgentStateDB,
    SimulationRun,
    TrajectoryRecordDB,
    get_session,
)

router = APIRouter(prefix="/simulations", tags=["simulation"])


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class SimulationCreate(BaseModel):
    name: str
    num_agents: int = 100
    num_steps: int = 144
    config_override: Optional[Dict[str, Any]] = None


class SimulationOut(BaseModel):
    id: int
    name: str
    num_agents: Optional[int]
    num_steps: Optional[int]
    status: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


class TrajectoryOut(BaseModel):
    agent_id: str
    day: int
    time_slot: int
    lat: float
    lon: float
    poi_id: str
    poi_type: str
    intent_category: str
    intent_path: str

    class Config:
        from_attributes = True


class IntentStat(BaseModel):
    intent_category: str
    count: int
    fast_path_count: int
    slow_path_count: int


# ─────────────────────────────────────────────────────────────────────────────
# Dependency
# ─────────────────────────────────────────────────────────────────────────────

def _get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Background task: run simulation
# ─────────────────────────────────────────────────────────────────────────────

def _run_simulation_task(run_id: int, sim_cfg: dict) -> None:
    """
    Background task that executes the simulation and persists results to DB.
    Heavy imports are deferred here so the API server starts quickly.
    """
    import yaml
    from pathlib import Path
    from agent.intent import StubBackend, TransformersBackend
    from simulation.runner import SimulationRunner
    from simulation.environment import TrajectoryRecord

    db = get_session()
    try:
        run = db.get(SimulationRun, run_id)
        if run is None:
            return
        run.status = "running"
        db.commit()

        # Load global config and merge overrides
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)
        if sim_cfg.get("config_override"):
            _deep_merge(cfg, sim_cfg["config_override"])
        cfg["simulation"]["num_agents"] = sim_cfg["num_agents"]
        cfg["simulation"]["num_steps"] = sim_cfg["num_steps"]

        # Choose LLM backend
        dpo_path = Path(cfg["llm"]["dpo_output_dir"])
        if dpo_path.exists():
            llm = TransformersBackend(
                model_path=str(dpo_path),
                device_map=cfg["llm"].get("device_map", "auto"),
                load_in_4bit=cfg["llm"].get("load_in_4bit", True),
            )
        else:
            llm = StubBackend()

        runner = SimulationRunner(cfg=cfg, llm_backend=llm)

        # Run and capture trajectory records (patch save to also write to DB)
        from simulation.environment import CityEnvironment
        from data.poi import load_pois_from_json, SpatialGrid
        from training.profile import load_profiles

        processed = Path(cfg["data"]["processed_dir"])
        pois = load_pois_from_json(processed / "pois.json")
        grid = SpatialGrid(pois)
        env = CityEnvironment(spatial_grid=grid, step_minutes=cfg["simulation"]["step_minutes"])

        runner.run()  # Saves JSON output; also collect from log
        records = env.get_trajectory_log()

        # Persist to DB
        for rec in records:
            db.add(TrajectoryRecordDB(
                run_id=run_id,
                agent_id=rec.agent_id,
                day=rec.day,
                time_slot=rec.time_slot,
                lat=rec.lat,
                lon=rec.lon,
                poi_id=rec.poi_id,
                poi_type=rec.poi_type,
                intent_category=rec.intent_category,
                intent_explanation=rec.intent_explanation,
                intent_path=rec.intent_path,
            ))
        run.status = "done"
        run.finished_at = datetime.utcnow()
        db.commit()

    except Exception as exc:
        run = db.get(SimulationRun, run_id)
        if run:
            run.status = "failed"
            db.commit()
        raise
    finally:
        db.close()


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/", response_model=SimulationOut, status_code=201)
def create_simulation(
    payload: SimulationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(_get_db),
):
    run = SimulationRun(
        name=payload.name,
        num_agents=payload.num_agents,
        num_steps=payload.num_steps,
        config_snapshot=json.dumps(payload.config_override or {}),
        status="pending",
        started_at=datetime.utcnow(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    sim_cfg = {
        "num_agents": payload.num_agents,
        "num_steps": payload.num_steps,
        "config_override": payload.config_override,
    }
    background_tasks.add_task(_run_simulation_task, run.id, sim_cfg)
    return run


@router.get("/", response_model=List[SimulationOut])
def list_simulations(db: Session = Depends(_get_db)):
    return db.query(SimulationRun).order_by(SimulationRun.started_at.desc()).all()


@router.get("/{run_id}", response_model=SimulationOut)
def get_simulation(run_id: int, db: Session = Depends(_get_db)):
    run = db.get(SimulationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return run


@router.get("/{run_id}/trajectories", response_model=List[TrajectoryOut])
def get_trajectories(
    run_id: int,
    agent_id: Optional[str] = None,
    day: Optional[int] = None,
    limit: int = 1000,
    offset: int = 0,
    db: Session = Depends(_get_db),
):
    query = db.query(TrajectoryRecordDB).filter(TrajectoryRecordDB.run_id == run_id)
    if agent_id:
        query = query.filter(TrajectoryRecordDB.agent_id == agent_id)
    if day is not None:
        query = query.filter(TrajectoryRecordDB.day == day)
    return query.order_by(
        TrajectoryRecordDB.agent_id,
        TrajectoryRecordDB.day,
        TrajectoryRecordDB.time_slot,
    ).offset(offset).limit(limit).all()


@router.get("/{run_id}/intents", response_model=List[IntentStat])
def get_intent_stats(run_id: int, db: Session = Depends(_get_db)):
    from sqlalchemy import func

    rows = (
        db.query(
            TrajectoryRecordDB.intent_category,
            TrajectoryRecordDB.intent_path,
            func.count().label("cnt"),
        )
        .filter(TrajectoryRecordDB.run_id == run_id)
        .group_by(
            TrajectoryRecordDB.intent_category,
            TrajectoryRecordDB.intent_path,
        )
        .all()
    )

    stats: Dict[str, Dict] = {}
    for cat, path, cnt in rows:
        if cat not in stats:
            stats[cat] = {"intent_category": cat, "count": 0, "fast_path_count": 0, "slow_path_count": 0}
        stats[cat]["count"] += cnt
        if path == "fast":
            stats[cat]["fast_path_count"] += cnt
        else:
            stats[cat]["slow_path_count"] += cnt

    return list(stats.values())


@router.delete("/{run_id}", status_code=204)
def delete_simulation(run_id: int, db: Session = Depends(_get_db)):
    run = db.get(SimulationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    db.query(TrajectoryRecordDB).filter(TrajectoryRecordDB.run_id == run_id).delete()
    db.query(AgentStateDB).filter(AgentStateDB.run_id == run_id).delete()
    db.delete(run)
    db.commit()
