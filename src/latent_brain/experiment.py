"""Train on disjoint maps, evaluate world-model and real navigation on new maps."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import torch
from torch import nn

from .brain import LatentBrain
from .planning import brain_predictor, greedy_baseline, oracle_plan, plan, replay
from .world import dataset, make_mazes

TRAIN_MAP_SEED = 1107
TEST_MAP_SEED = 99007


@dataclass
class TrainConfig:
    seed: int = 42
    train_maps: int = 120
    test_maps: int = 45
    size: int = 7
    epochs: int = 8
    batch_size: int = 512
    learning_rate: float = 0.006
    hidden_size: int = 64
    thinking_steps: int = 2


def make_datasets(config: TrainConfig):
    training = make_mazes(TRAIN_MAP_SEED + config.seed, config.train_maps, config.size)
    heldout = make_mazes(TEST_MAP_SEED + config.seed, config.test_maps, config.size)
    # Collision-resistant split check: protect accidentally identical maps.
    train_layouts = {m.rows for m in training}
    eval_layouts = {m.rows for m in heldout}
    if train_layouts & eval_layouts:
        raise RuntimeError("train and evaluation maps overlap")
    return training, heldout


def train(config: TrainConfig) -> tuple[LatentBrain, dict]:
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    training_maps, _ = make_datasets(config)
    observations, labels = dataset(training_maps)
    x = torch.tensor(observations, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    brain = LatentBrain(config.hidden_size, config.thinking_steps)
    optimizer = torch.optim.Adam(brain.parameters(), lr=config.learning_rate)
    objective = nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(config.epochs):
        order = torch.randperm(len(y))
        brain.train()
        total = 0.0
        for batch in order.split(config.batch_size):
            loss = objective(brain(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        losses.append(round(total / len(y), 6))
    brain.eval()
    return brain, {"config": asdict(config), "training_samples": len(y), "training_loss_by_epoch": losses,
                   "training_map_seed": TRAIN_MAP_SEED + config.seed, "evaluation_map_seed": TEST_MAP_SEED + config.seed}


def evaluate(brain: LatentBrain, config: TrainConfig) -> dict:
    _, eval_maps = make_datasets(config)
    x, truth = dataset(eval_maps)
    observations = torch.tensor(x, dtype=torch.float32)
    # Evaluation on held-out inputs, not on training maps.
    predictions = []
    with torch.inference_mode():
        for batch in observations.split(2048):
            predictions.extend(brain(batch).sigmoid().ge(0.5).tolist())
    correct = sum(int(a == bool(b)) for a, b in zip(predictions, truth))
    naive = max(sum(truth), len(truth) - sum(truth)) / len(truth)
    predictor = brain_predictor(brain)
    planned = [plan(m, predictor) for m in eval_maps]
    outcomes = [replay(m, p) for m, p in zip(eval_maps, planned)]
    greedy = [greedy_baseline(m) for m in eval_maps]
    oracle = [replay(m, oracle_plan(m)) for m in eval_maps]
    # Matched-planner ablation: SAME A* search, untrained neural transitions.
    # This separates learning from the effect of adding an A* planner, though
    # a deterministic non-neural learned model is still a future comparison.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1987 + config.seed)
        untrained = LatentBrain(config.hidden_size, config.thinking_steps)
    raw_predictor = brain_predictor(untrained)
    random_init_results = [replay(m, plan(m, raw_predictor)) for m in eval_maps]
    return {
        "disjoint_evaluation_maps": len(eval_maps),
        "evaluation_transitions": len(truth),
        "transition_accuracy": round(correct / len(truth), 4),
        "majority_class_baseline_accuracy": round(naive, 4),
        "model_based_planning_successes": sum(outcomes),
        "untrained_same_planner_successes": sum(random_init_results),
        "greedy_baseline_successes": sum(greedy),
        "oracle_successes": sum(oracle),
        "planning_trials": len(eval_maps),
        "note": "Model-based plans use learned predicted transitions; successes verified by real environment replay.",
    }
