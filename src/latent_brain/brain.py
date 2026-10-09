"""Trainable recurrent latent transition model, intentionally CPU-friendly."""
from __future__ import annotations

import torch
from torch import nn


class LatentBrain(nn.Module):
    """Build a learned hidden state and refine it before predicting a move.

    This is a modest recurrent world-model prototype, NOT a general-purpose
    reasoner. Actual future trajectories are explored by the search algorithm.
    """

    def __init__(self, hidden_size: int = 64, thinking_steps: int = 2):
        super().__init__()
        if hidden_size < 8 or thinking_steps < 1:
            raise ValueError("hidden_size >= 8 and thinking_steps >= 1 required")
        self.hidden_size = hidden_size
        self.thinking_steps = thinking_steps
        self.encoder = nn.Sequential(nn.Linear(15, hidden_size), nn.Tanh())
        self.refine = nn.GRUCell(hidden_size, hidden_size)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(observations)
        hidden = torch.zeros_like(encoded)
        for _ in range(self.thinking_steps):
            hidden = self.refine(encoded, hidden)
        return self.head(hidden).squeeze(-1)

    def probability(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self(observations).sigmoid()

    def metadata(self) -> dict:
        return {
            "format_version": 1,
            "hidden_size": self.hidden_size,
            "thinking_steps": self.thinking_steps,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


def save_checkpoint(brain: LatentBrain, path: str, training_meta: dict | None = None):
    """Only store tensors and our own primitive metadata. No user pickle data."""
    torch.save({"model": brain.state_dict(), "architecture": brain.metadata(), "training": training_meta or {}}, path)


def load_checkpoint(path: str) -> tuple[LatentBrain, dict]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    cfg = data["architecture"]
    if cfg["format_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    brain = LatentBrain(hidden_size=int(cfg["hidden_size"]), thinking_steps=int(cfg["thinking_steps"]))
    brain.load_state_dict(data["model"])
    brain.eval()
    return brain, data.get("training", {})
