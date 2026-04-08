"""
api/db.py — SQLAlchemy models and database session management.

Tables:
  - simulation_runs      : Metadata for each simulation experiment
  - trajectory_records   : Per-step movement events (agent_id, time, location, intent)
  - agent_states         : Snapshot of agent state at each step (optional)
  - intent_sequences     : Intent sequences per agent per simulation
"""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


# ─────────────────────────────────────────────────────────────────────────────
# ORM base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    config_snapshot = Column(Text)                    # JSON-encoded config
    num_agents = Column(Integer)
    num_steps = Column(Integer)
    status = Column(String(50), default="pending")    # pending / running / done / failed
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)

    trajectory_records = relationship("TrajectoryRecordDB", back_populates="run")


class TrajectoryRecordDB(Base):
    __tablename__ = "trajectory_records"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("simulation_runs.id"), index=True)
    agent_id = Column(String(50), index=True)
    day = Column(Integer)
    time_slot = Column(Integer)
    lat = Column(Float)
    lon = Column(Float)
    poi_id = Column(String(100))
    poi_type = Column(String(100))
    intent_category = Column(String(50))
    intent_explanation = Column(Text)
    intent_path = Column(String(10))   # "fast" | "slow"

    run = relationship("SimulationRun", back_populates="trajectory_records")


class AgentStateDB(Base):
    __tablename__ = "agent_states"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("simulation_runs.id"), index=True)
    agent_id = Column(String(50), index=True)
    day = Column(Integer)
    time_slot = Column(Integer)
    role = Column(String(50))
    fatigue = Column(Float)
    workload = Column(Float)


# ─────────────────────────────────────────────────────────────────────────────
# Database engine and session factory
# ─────────────────────────────────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def init_db(database_url: str) -> None:
    """Initialise the database engine and create all tables."""
    global _engine, _SessionLocal
    _engine = create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"charset": "utf8mb4"} if "mysql" in database_url else {},
    )
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    Base.metadata.create_all(bind=_engine)


def get_session() -> Session:
    """Return a new SQLAlchemy session (caller must close it)."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _SessionLocal()
