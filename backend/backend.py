from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import json
import math
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.neural_ode import NeuralODE

app = FastAPI(title="NeuroODE", version="3.0")
BASE_DIR = Path(__file__).resolve().parent.parent

app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

@app.get("/")
def serve():
    return FileResponse(BASE_DIR / "index.html")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WEIGHTS = Path(__file__).parent / "weights.pt"
STATS   = Path(__file__).parent / "norm_stats.json"
DEVICE  = "cpu"

_model: NeuralODE | None = None
_norm:  dict | None = None


def get_model() -> NeuralODE:
    global _model, _norm
    if _model is None:
        if not WEIGHTS.exists():
            raise HTTPException(503, detail="Model not trained. Run: python scripts/train.py")
        _model = NeuralODE.load(str(WEIGHTS), device=DEVICE)
        with open(STATS) as f:
            _norm = json.load(f)
    return _model


class SimulateRequest(BaseModel):
    attention: float = Field(0.65, ge=0.0, le=1.0)
    fatigue:   float = Field(0.25, ge=0.0, le=1.0)
    stress:    float = Field(0.30, ge=0.0, le=1.0)
    alpha_power: float = Field(0.40, ge=0.0, le=1.0)
    theta_power: float = Field(0.30, ge=0.0, le=1.0)
    beta_power:  float = Field(0.25, ge=0.0, le=1.0)
    duration:    float = Field(10.0, ge=1.0, le=60.0)
    n_steps:     int   = Field(40,   ge=4,   le=200)


class TrajectoryPoint(BaseModel):
    t: float
    attention: float
    fatigue: float
    stress: float


class SimulateResponse(BaseModel):
    trajectory: list[TrajectoryPoint]
    mse: float
    n_steps: int


@app.get("/api/health")
def health():
    trained = WEIGHTS.exists()
    return {
        "status": "ok",
        "model": "NeuralODE-v3",
        "trained": trained,
        "state_dim": 3,
        "states": ["attention", "fatigue", "stress"],
        "dataset": "DEAP (EEG alpha/theta/beta power)",
    }


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    model = get_model()
    mean  = np.array(_norm["mean"], dtype=np.float32)
    std   = np.array(_norm["std"],  dtype=np.float32)

    S0_raw  = np.array([req.attention, req.fatigue, req.stress], dtype=np.float32)
    S0_norm = (S0_raw - mean) / std
    S0      = torch.tensor(S0_norm, device=DEVICE)

    total = req.alpha_power + req.theta_power + req.beta_power + 1e-10
    U = torch.tensor([
        req.alpha_power / total,
        req.theta_power / total,
        req.beta_power  / total,
    ], device=DEVICE)

    t_span = torch.linspace(0.0, float(req.n_steps - 1), req.n_steps, device=DEVICE)

    with torch.no_grad():
        pred_norm = model(S0, t_span, U=U)

    pred_np = pred_norm.cpu().numpy() * std + mean
    pred_np = np.clip(pred_np, 0.0, 1.0)

    trajectory = [
        TrajectoryPoint(
            t=float(t_span[i].item() / (req.n_steps - 1) * req.duration),
            attention=float(pred_np[i, 0]),
            fatigue=float(pred_np[i, 1]),
            stress=float(pred_np[i, 2]),
        )
        for i in range(req.n_steps)
    ]

    mse = float(np.mean((pred_np - pred_np[0]) ** 2))

    return SimulateResponse(trajectory=trajectory, mse=round(mse, 6), n_steps=req.n_steps)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.backend:app", host="0.0.0.0", port=8000, reload=False)
