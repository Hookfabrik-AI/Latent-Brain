from pathlib import Path

import torch

from latent_brain.brain import LatentBrain, load_checkpoint, save_checkpoint
from latent_brain.experiment import TrainConfig, make_datasets


def test_brain_shape_and_latent_steps():
    brain = LatentBrain(hidden_size=16, thinking_steps=3)
    out = brain(torch.zeros(5, 15))
    assert out.shape == (5,)
    assert brain.metadata()["parameters"] > 0


def test_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(111)
    original = LatentBrain(hidden_size=16)
    path = tmp_path / "weights.pt"
    save_checkpoint(original, str(path), {"source": "test"})
    restored, meta = load_checkpoint(str(path))
    sample = torch.rand(4, 15)
    assert torch.allclose(original(sample), restored(sample))
    assert meta["source"] == "test"


def test_disjoint_training_test_maps():
    train, test = make_datasets(TrainConfig(train_maps=8, test_maps=8))
    assert not ({m.rows for m in train} & {m.rows for m in test})
