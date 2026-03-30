# NeuroODE — Cognitive State Simulator

A focused Neural ODE that models how cognitive states (attention, fatigue, stress) evolve through time using EEG features from the DEAP dataset.

## What It Does

Learns a differential equation `dS/dt = f(S, U, θ)` from EEG data. Given an initial state and EEG band powers, it predicts how attention, fatigue, and stress evolve continuously through time.

## Architecture

```
DEAP EEG Dataset  (32-channel, 32 subjects, 40 emotion trials each)
       ↓
EEG Band Features  →  [alpha_power, theta_power, beta_power]
       ↓
State Mapping      →  S = [attention, fatigue, stress]  ∈ ℝ³
       ↓
Neural ODE         →  dS/dt = f(S, U, θ)   learned via torchdiffeq
       ↓
Trajectory S(t)    →  dopri5 adaptive solver
       ↓
FastAPI  →  React frontend
```

## Project Structure

```
neuro_ode/
├── data/
│   └── preprocess.py       ← EEG feature extraction + state mapping
├── model/
│   └── neural_ode.py       ← ODEFunc + NeuralODE (PyTorch)
├── scripts/
│   └── train.py            ← training loop, evaluation, saves weights
├── api/
│   ├── backend.py          ← FastAPI: /api/health + /api/simulate
│   ├── weights.pt          ← saved after training
│   └── norm_stats.json     ← normalization stats
├── frontend/
│   └── index.html          ← React + Recharts visualization
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Train (uses synthetic DEAP-like data if real DEAP not present)
python scripts/train.py

# Serve API
python api/backend.py
# → http://localhost:8000

# Open frontend/index.html in browser
```

## Real DEAP Data

1. Request access at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
2. Place `.mat` files in `data/deap/`
3. Re-run `python scripts/train.py`

## API

### `GET /api/health`
```json
{ "status": "ok", "trained": true, "state_dim": 3 }
```

### `POST /api/simulate`
```json
{
  "attention": 0.65, "fatigue": 0.25, "stress": 0.30,
  "alpha_power": 0.40, "theta_power": 0.30, "beta_power": 0.25,
  "duration": 10.0, "n_steps": 40
}
```
Returns: `{ trajectory: [{t, attention, fatigue, stress}], mse, n_steps }`

## State Space

| State | EEG Proxy | Interpretation |
|-------|-----------|----------------|
| Attention | ↑ alpha relative power | Focused, alert |
| Fatigue | ↑ theta relative power | Drowsy, slowing |
| Stress | ↑ beta relative power | Anxious, activated |

## What Was Removed

| Before | After |
|--------|-------|
| SyntheticDataLayer (4 fake datasets) | Single real DEAP pipeline |
| CORAL multi-domain alignment | Simple z-score normalization |
| Hand-crafted coupling matrices | Learned ODEFunc (nn.Sequential) |
| RK4 duplicated in frontend JS | Backend-only inference |
| HNNBridge, domain_gains, 5-D state | Clean 3-D state space |
| 8 API endpoints | 2 endpoints |
