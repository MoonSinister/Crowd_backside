# Crowd Backside — LLM-Driven Urban Crowd Movement Simulation

基于多智能体意图建模的人群出行行为涌现机制分析系统

## Architecture

```
crowd_backside/
├── config/             # YAML configuration files
├── data/               # Data pipeline
│   ├── poi.py          # POI data structures and spatial grid
│   ├── trajectory.py   # Trajectory preprocessing
│   └── raw/ processed/ # Data directories
├── agent/              # Agent five-tuple (S, P, M, C, A)
│   ├── state.py        # S: Individual state layer
│   ├── perception.py   # P: Perception module
│   ├── memory.py       # M: Episodic memory module
│   ├── intent.py       # C: Cognitive intent module (LLM-driven)
│   └── behavior.py     # A: Behavior module (gravity-model location sampling)
├── training/           # LLM training pipeline
│   ├── profile.py      # Individual profile builder
│   ├── dataset.py      # Algorithm 1: DPO dataset construction
│   ├── sft.py          # Stage 1: Supervised Fine-Tuning (LoRA)
│   └── dpo.py          # Stage 2: Direct Preference Optimization
├── simulation/         # Repast4Py multi-agent simulation
│   ├── environment.py  # City simulation environment
│   └── runner.py       # Simulation orchestrator
├── api/                # FastAPI backend
│   ├── main.py         # Application entry point
│   ├── routers/        # API route handlers
│   └── db.py           # MySQL connection and models
└── frontend/           # Vue3 + Three.js frontend
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your settings
```

### 3. Prepare data

**Option A — Use the included synthetic sample data (quickest way to get started):**

```bash
# Generate synthetic POIs and trajectory CSV files for development/testing
python -m data.generate_sample_data
# Optional flags:
#   --num-users  20   (default: 10)
#   --num-days   14   (default: 10)
#   --output-dir data/raw
```

This creates `data/raw/pois.csv` and `data/raw/trajectories.csv` with realistic urban
mobility patterns across 5 social roles (office worker, student, delivery rider, etc.).

**Option B — Bring your own data:**

Place your own CSVs in `data/raw/`:
- `trajectories.csv`: columns `UserID, lat, lon, POI_type, timestamp[, poi_id]`
- `pois.csv`: columns `poi_id, name, lat, lon, raw_type`

**Preprocess (both options):**

```bash
# Preprocess trajectory data
python -m data.trajectory --input data/raw/trajectories.csv --output data/processed/

# Build city spatial grid from POI data
python -m data.poi --input data/raw/pois.csv --output data/processed/pois.json
```

### 4. Build individual profiles and training datasets

```bash
# Step 1: Build per-user activity profiles (role assignment)
python -m training.profile --config config/config.yaml

# Step 2: Construct SFT + DPO datasets (Algorithm 1)
python -m training.dataset --config config/config.yaml
```

### 5. Fine-tune the LLM

```bash
# Stage 1: SFT with LoRA
python -m training.sft --config config/config.yaml

# Stage 2: DPO preference alignment
python -m training.dpo --config config/config.yaml
```

### 6. Run simulation

```bash
python -m simulation.runner --config config/config.yaml
```

### 7. Start API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 8. Start frontend (development)

```bash
cd frontend && npm install && npm run dev
```

## Key Concepts

### Agent Decision Flow (per time step)

```
Perception (P) --> Retrieve Top-K memories (M)
                         |
                         v
               Cognitive Intent Module (C) -- LLM -->  Intent It
                         |
                         v
               Behavior Module (A): gravity-model sampling --> next location l_{t+1}
                         |
                         v
               Update state (S) and memory (M)
```

### Intent-to-Location Mapping (Gravity Model)

```
P(l_{t+1} | l_t, I_t) = rho(l, I_t) * d(l_t, l)^{-beta}
                         / sum_{l in L_t(I_t)} rho(l, I_t) * d(l_t, l)^{-beta}
```

Where:
- `rho(l, I_t)` = semantic matching strength between POI l and intent I_t
- `d(l_t, l)` = spatial distance (Haversine)
- `beta` = distance decay coefficient
- `L_t(I_t)` = set of candidate POIs semantically compatible with intent I_t

### Memory Retrieval Score

```
score(m | p_t) = cosine_sim(p_t, p_m) * exp(-lambda * (t - t_m))
```

Where `lambda` controls the time-decay rate (recency bias).
