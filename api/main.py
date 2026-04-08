"""
api/main.py — FastAPI application entry point.

Endpoints:
  /api/simulations/*   Simulation management (see routers/simulation.py)
  /api/health          Health check
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import init_db
from api.routers.simulation import router as sim_router


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(config_path: str = "config/config.yaml") -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Crowd Backside API",
        description="LLM-driven urban crowd movement simulation backend",
        version="0.1.0",
    )

    # ── Load config ────────────────────────────────────────────────────────
    cfg: dict = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    api_cfg = cfg.get("api", {})
    db_cfg = cfg.get("database", {})

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins = api_cfg.get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Database ───────────────────────────────────────────────────────────
    db_url = db_cfg.get(
        "url",
        os.environ.get(
            "DATABASE_URL",
            "sqlite:///./crowd_backside.db",   # SQLite fallback for development
        ),
    )
    init_db(db_url)

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(sim_router, prefix="/api")

    # ── Health check ───────────────────────────────────────────────────────
    @app.get("/api/health")
    async def health_check():
        return {"status": "ok"}

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Module-level app instance for uvicorn
# ─────────────────────────────────────────────────────────────────────────────

app = create_app()
