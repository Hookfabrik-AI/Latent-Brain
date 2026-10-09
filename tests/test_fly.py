"""Learning and action-truth constraints for the experimental FlyBrain."""
from pathlib import Path

import pytest
import torch

from latent_brain.brain import LatentBrain
from latent_brain.fly import (BUDGETS, FlyBrain, FlySession, FlyTrainConfig,
                               decide, load_fly, observe_features, save_fly,
                               teacher_label, train_fly, training_episodes,
                               oracle_distances, roll_out_episode)
from latent_brain.world import Maze, make_mazes


def test_observation_state_dimensions_and_actions():
    m = Maze(("...", ".#.", "..."))
    f = observe_features(m, (0, 0), previous_action=1, previous_success=True)
    assert len(f) == 18
    assert f[-1] == 1
    assert f[13] == 0 and f[14] == 1
    with pytest.raises(ValueError):
        observe_features(m, (3, 0))


def test_trainable_heads_and_memory_have_expected_shapes():
    model = FlyBrain(16)
    observations = torch.rand(3, 18)
    policy, budget, hidden = model.step(observations, model.initial_state(3))
    assert policy.shape == (3, 4) and budget.shape == (3, 3)
    assert hidden.shape == (3, 16)
    imagined = model.imagine(hidden, torch.tensor([0, 1, 2]))
    assert imagined.shape == hidden.shape
    assert not torch.allclose(imagined, hidden)
    with pytest.raises(ValueError):
        model.imagine(hidden, torch.tensor([0, 1, 4]))


def test_hidden_persists_until_explicit_reset():
    model = FlyBrain(16)
    maze = Maze(("...", "...", "..."))
    s = FlySession(model)
    _, _, h1 = s.observe(maze, (0, 0))
    s.accept_outcome(0, True)
    _, _, h2 = s.observe(maze, (1, 0))
    assert not torch.allclose(h1, h2)
    s.reset()
    _, _, h3 = s.observe(maze, (0, 0))
    assert torch.allclose(h1, h3)
    with pytest.raises(ValueError):
        s.accept_outcome(6, True)


def test_learning_uses_disjoint_worlds_and_loss_improves():
    brain, info = train_fly(FlyTrainConfig(seed=123, train_maps=14, test_maps=6,
                                          epochs=3, hidden_size=16))
    assert brain.metadata()["parameters"] > 0
    assert info["train_eval_maps_disjoint"] is True
    assert info["train_losses"][-1] < info["train_losses"][0]


def test_checkpoint_roundtrip(tmp_path: Path):
    model = FlyBrain(16)
    target = tmp_path / "fly.pt"
    save_fly(model, str(target), {"phase": "test"})
    restored, metadata = load_fly(str(target))
    assert metadata["phase"] == "test"
    x = torch.rand(1, 18)
    p1, b1, _ = model.step(x, model.initial_state())
    p2, b2, _ = restored.step(x, restored.initial_state())
    assert torch.allclose(p1, p2) and torch.allclose(b1, b2)


def test_decision_simulates_without_real_move(monkeypatch):
    # A plan may use only the learned transition model, not ground-truth moves.
    m = Maze(("...", "...", "..."))
    brain = FlyBrain(16)
    def fake_predictor(_maze, queries):
        return [True] * len(queries)
    def fail_if_move_called(*args, **kwargs):
        raise AssertionError("planner called real move")
    monkeypatch.setattr(Maze, "move", fail_if_move_called)
    response = decide(brain, m, (0, 0), fake_predictor, forced_budget=4)
    assert response["thinking_budget"] == 4
    assert response["verification"] == "not_executed"
    assert response["imagined_expansions"] > 0
    assert response["action_id"] in range(4)
    assert abs(sum(response["policy_probabilities"]) - 1) < 1e-5


def test_all_blocked_predictions_do_not_invent_success():
    m = Maze(("..", ".."))
    response = decide(FlyBrain(16), m, m.start,
                      lambda _m, q: [False] * len(q))
    assert response["action_id"] is None
    assert response["verification"] == "not_executed"


def test_unobserved_rollouts_are_not_declared_real_execution():
    m = Maze(("...", "...", "..."))
    model = FlyBrain(16)
    r = roll_out_episode(model, m, lambda _m, questions: [False] * len(questions), max_steps=3)
    assert r["success"] is False and r["steps"] == 0


def test_teacher_labels_only_from_known_training_truth():
    maze = Maze(("...", ".#.", "..."))
    d = oracle_distances(maze)
    action, budget = teacher_label(maze, maze.start, d)
    assert action in range(4) and budget in range(len(BUDGETS))
    assert training_episodes([maze], seed=9)


def test_forced_budget_and_mismatched_session_rejected():
    m = Maze(("..", ".."))
    model = FlyBrain(16)
    pred = lambda _m, q: [True] * len(q)
    with pytest.raises(ValueError):
        decide(model, m, m.start, pred, forced_budget=3)
    with pytest.raises(ValueError):
        decide(model, m, m.start, pred, session=FlySession(FlyBrain(16)))
