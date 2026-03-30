import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


from data.preprocess import build_synthetic_deap_trajectories, normalize_trajectories, build_trajectories_from_deap
from model.neural_ode import NeuralODE

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS   = 120
LR       = 3e-3
BATCH    = 8
WEIGHTS  = Path(__file__).parent.parent / "api" / "weights.pt"
STATS    = Path(__file__).parent.parent / "api" / "norm_stats.json"
DATA_DIR = Path(__file__).parent.parent / "data" / "deap"


def load_data() -> list[dict]:
    if DATA_DIR.exists() and any(DATA_DIR.glob("s*.dat")):
        print(f"[data] Loading real DEAP from {DATA_DIR}")
        return build_trajectories_from_deap(str(DATA_DIR), n_subjects=10)
    print("[data] DEAP not found → using synthetic DEAP-like data")
    return build_synthetic_deap_trajectories(n_subjects=12, n_trials=12)


def make_batch(trajectories: list[dict], batch_size: int) -> list[dict]:
    idxs = np.random.choice(len(trajectories), size=batch_size, replace=False)
    return [trajectories[i] for i in idxs]


def train():
    raw = load_data()
    trajectories, norm = normalize_trajectories(raw)
    print(f"[data] {len(trajectories)} trajectories loaded")

    model = NeuralODE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = max(1, len(trajectories) // BATCH)

        for _ in range(n_batches):
            batch = make_batch(trajectories, min(BATCH, len(trajectories)))
            batch_loss = torch.tensor(0.0, device=DEVICE)

            for traj in batch:
                states = torch.tensor(traj["states_norm"], device=DEVICE)
                n_t = states.shape[0]
                if n_t < 2:
                    continue

                t_span = torch.linspace(0.0, float(n_t - 1), n_t, device=DEVICE)
                S0     = states[0]

                feats  = torch.tensor(traj["features"][0], device=DEVICE)
                pred   = model(S0, t_span, U=feats)

                loss = F.mse_loss(pred, states)
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()

        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(round(avg, 6))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{EPOCHS}  loss={avg:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    WEIGHTS.parent.mkdir(exist_ok=True)
    model.save(str(WEIGHTS))
    print(f"\n[train] Saved weights → {WEIGHTS}")

    with open(STATS, "w") as f:
        json.dump(norm, f)
    print(f"[train] Saved norm stats → {STATS}")

    loss_path = WEIGHTS.parent / "train_losses.json"
    with open(loss_path, "w") as f:
        json.dump(losses, f)
    print(f"[train] Saved loss curve → {loss_path}")

    print("\n[eval] Running quick evaluation...")
    eval_trajectories(model, trajectories[:10], norm)


def eval_trajectories(model: NeuralODE, trajectories: list[dict], norm: dict):
    model.eval()
    mean = np.array(norm["mean"])
    std  = np.array(norm["std"])
    total_mse = 0.0

    with torch.no_grad():
        for traj in trajectories:
            states = torch.tensor(traj["states_norm"], device=DEVICE)
            n_t    = states.shape[0]
            if n_t < 2:
                continue

            t_span = torch.linspace(0.0, float(n_t - 1), n_t, device=DEVICE)
            S0     = states[0]
            feats  = torch.tensor(traj["features"][0], device=DEVICE)

            pred   = model(S0, t_span, U=feats)
            mse    = F.mse_loss(pred, states).item()
            total_mse += mse

    avg_mse = total_mse / max(len(trajectories), 1)
    pred_np = pred.cpu().numpy() * std + mean
    true_np = states.cpu().numpy() * std + mean

    print(f"  avg MSE (normalized) : {avg_mse:.6f}")
    print(f"  final pred  : att={pred_np[-1,0]:.3f}  fat={pred_np[-1,1]:.3f}  str={pred_np[-1,2]:.3f}")
    print(f"  final true  : att={true_np[-1,0]:.3f}  fat={true_np[-1,1]:.3f}  str={true_np[-1,2]:.3f}")


if __name__ == "__main__":
    train()
