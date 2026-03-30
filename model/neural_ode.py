import torch
import torch.nn as nn
from torchdiffeq import odeint


STATE_DIM = 3
INPUT_DIM = 3
HIDDEN_DIM = 32


class ODEFunc(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

        self._U = torch.zeros(input_dim)

    def set_input(self, U: torch.Tensor):
        self._U = U.to(next(self.parameters()).device)

    def forward(self, t: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        if S.dim() == 1:
            U = self._U.expand(1, -1).squeeze(0)
            x = torch.cat([S, U], dim=-1)
        else:
            U = self._U.unsqueeze(0).expand(S.shape[0], -1)
            x = torch.cat([S, U], dim=-1)
        return self.net(x)


class NeuralODE(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.func = ODEFunc(state_dim, input_dim, hidden_dim)

    def forward(self, S0: torch.Tensor, t_span: torch.Tensor, U: torch.Tensor | None = None) -> torch.Tensor:
        if U is not None:
            self.func.set_input(U)
        trajectory = odeint(self.func, S0, t_span, method="dopri5", rtol=1e-4, atol=1e-5)
        return trajectory

    def save(self, path: str):
        torch.save({"func_state": self.func.state_dict(), "state_dim": self.func.state_dim}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NeuralODE":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(state_dim=ckpt.get("state_dim", STATE_DIM))
        model.func.load_state_dict(ckpt["func_state"])
        model.eval()
        return model
