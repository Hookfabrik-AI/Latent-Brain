"""Experimental, CPU-first FlyBrain: persistent state, learned action/budget
preferences, learned latent imagination and bounded model-based rollouts.

This model does *not* contain an LLM, an executor or any execution authority.
The existing independent maze evaluator remains the source of truth.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from random import Random
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from .planning import Predictor, make_transition_table
from .world import ACTIONS, ACTION_NAMES, Maze

INPUT_SIZE = 18
BUDGETS = (1, 2, 4)


def observe_features(maze: Maze, state: tuple[int, int], previous_action: int | None = None,
                     previous_success: bool = False) -> list[float]:
    """Public position/goal/local-wall observations, not future outcomes."""
    x, y = state
    if not 0 <= x < maze.width or not 0 <= y < maze.height:
        raise ValueError("state outside map")
    if previous_action is not None and previous_action not in range(4):
        raise ValueError("invalid previous_action")
    patch = [float(maze.blocked(x + dx, y + dy))
             for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    gx, gy = maze.goal
    return ([x / (maze.width - 1), y / (maze.height - 1)] + patch
            + [(gx - x) / (maze.width - 1), (gy - y) / (maze.height - 1)]
            + [float(previous_action == a) for a in range(4)]
            + [float(previous_success)])


class FlyBrain(nn.Module):
    """GRU memory + trainable policy, compute budget and latent dynamics."""

    def __init__(self, hidden_size: int = 48):
        super().__init__()
        if not 8 <= hidden_size <= 256:
            raise ValueError("hidden_size must be 8..256")
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(INPUT_SIZE, hidden_size), nn.Tanh())
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, 4)
        self.budget = nn.Linear(hidden_size, len(BUDGETS))
        self.imagination = nn.Sequential(nn.Linear(hidden_size + 4, hidden_size), nn.Tanh())

    def initial_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, dtype=torch.float32)

    def step(self, observation: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[1] != INPUT_SIZE:
            raise ValueError("observation must be [batch,18]")
        if hidden.shape != (observation.shape[0], self.hidden_size):
            raise ValueError("hidden has the wrong shape")
        nxt = self.memory(self.encoder(observation), hidden)
        return self.policy(nxt), self.budget(nxt), nxt

    def imagine(self, hidden: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next hidden state from action and current hidden only."""
        if hidden.ndim != 2 or action.ndim != 1 or action.shape[0] != hidden.shape[0]:
            raise ValueError("invalid imagined action or hidden")
        if not bool(((action >= 0) & (action < 4)).all()):
            raise ValueError("action is outside the known action space")
        return self.imagination(torch.cat((hidden, F.one_hot(action.long(), 4).float()), dim=-1))

    def metadata(self) -> dict:
        return {"format_version": 1, "input_size": INPUT_SIZE,
                "hidden_size": self.hidden_size, "budgets": list(BUDGETS),
                "parameters": sum(p.numel() for p in self.parameters())}


def save_fly(brain: FlyBrain, path: str, metadata: dict | None = None) -> None:
    torch.save({"architecture": brain.metadata(), "model": brain.state_dict(),
                "training": metadata or {}}, path)


def load_fly(path: str) -> tuple[FlyBrain, dict]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    cfg = data["architecture"]
    if cfg["format_version"] != 1 or cfg["input_size"] != INPUT_SIZE or cfg["budgets"] != list(BUDGETS):
        raise ValueError("unsupported FlyBrain checkpoint format")
    model = FlyBrain(hidden_size=int(cfg["hidden_size"]))
    model.load_state_dict(data["model"], strict=True)
    return model.eval(), data.get("training", {})


def oracle_distances(maze: Maze) -> dict[tuple[int, int], int]:
    """Ground-truth teacher for training/evaluation only; NEVER for decisions."""
    distance = {maze.goal: 0}
    queue = deque([maze.goal])
    while queue:
        state = queue.popleft()
        for action in range(4):
            nxt = maze.move(state, action)
            if nxt != state and nxt not in distance:
                distance[nxt] = distance[state] + 1
                queue.append(nxt)
    return distance


def teacher_label(maze: Maze, state: tuple[int, int], distance: dict) -> tuple[int, int]:
    """Return expert action and a *proxy* for useful depth, not optimal compute."""
    candidates = [(distance.get(maze.move(state, a), 10**6), a)
                  for a in range(4) if maze.move(state, a) != state]
    if not candidates or state not in distance or state == maze.goal:
        raise ValueError("teacher requires a reachable, nonterminal state")
    _, action = min(candidates)
    gx, gy = maze.goal
    def manhattan(s):
        return abs(gx - s[0]) + abs(gy - s[1])
    naive = min(candidates, key=lambda pair: (manhattan(maze.move(state, pair[1])), pair[1]))[1]
    detour = naive != action and distance[maze.move(state, naive)] > distance[maze.move(state, action)]
    gap = distance[state] - manhattan(state)
    budget_index = 2 if detour else (1 if gap >= 2 else 0)
    return action, budget_index


def training_episodes(mazes: list[Maze], seed: int = 314, episodes_per_map: int = 3,
                      max_steps: int = 28) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Expert labels from verified transitions; NO test maps supplied here."""
    rng = Random(seed)
    episodes = []
    for maze in mazes:
        distance = oracle_distances(maze)
        starts = [maze.start] + [rng.choice(list(maze.free_states())) for _ in range(episodes_per_map - 1)]
        for start in starts:
            state, previous, previous_ok = start, None, False
            inputs, actions, budgets, actual = [], [], [], []
            for _ in range(max_steps):
                if state == maze.goal or state not in distance:
                    break
                teacher, budget = teacher_label(maze, state, distance)
                inputs.append(observe_features(maze, state, previous, previous_ok))
                actions.append(teacher)
                budgets.append(budget)
                # Real transitions for training are explicitly allowed; not used in prediction.
                chosen = rng.randrange(4) if rng.random() < 0.17 else teacher
                actual.append(chosen)
                nxt = maze.move(state, chosen)
                previous, previous_ok = chosen, nxt != state
                state = nxt
            if inputs:
                episodes.append((torch.tensor(inputs, dtype=torch.float32),
                                 torch.tensor(actions, dtype=torch.long),
                                 torch.tensor(budgets, dtype=torch.long),
                                 torch.tensor(actual, dtype=torch.long)))
    return episodes


@dataclass(frozen=True)
class FlyTrainConfig:
    seed: int = 42
    train_maps: int = 90
    test_maps: int = 30
    size: int = 7
    epochs: int = 8
    hidden_size: int = 48
    learning_rate: float = 0.003


def train_fly(config: FlyTrainConfig) -> tuple[FlyBrain, dict]:
    """Imitation + latent one-step consistency on verified training trajectories."""
    from .experiment import make_datasets, TrainConfig
    if config.epochs < 1 or config.train_maps < 1 or config.test_maps < 1:
        raise ValueError("training config values must be positive")
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    training, evaluation = make_datasets(TrainConfig(seed=config.seed, train_maps=config.train_maps,
                                                    test_maps=config.test_maps, size=config.size))
    episodes = training_episodes(training, seed=config.seed)
    brain = FlyBrain(config.hidden_size)
    optimizer = torch.optim.Adam(brain.parameters(), lr=config.learning_rate)
    history = []
    rng = Random(config.seed)
    for _ in range(config.epochs):
        rng.shuffle(episodes)
        total_loss = 0.0
        brain.train()
        for x, actions, budgets, actual in episodes:
            hidden = brain.initial_state()
            action_loss, budget_loss, future_loss = [], [], []
            last_hidden = None
            for idx in range(len(x)):
                logits, effort, hidden = brain.step(x[idx:idx+1], hidden)
                action_loss.append(F.cross_entropy(logits, actions[idx:idx+1]))
                budget_loss.append(F.cross_entropy(effort, budgets[idx:idx+1]))
                if last_hidden is not None:
                    predicted = brain.imagine(last_hidden, actual[idx-1:idx])
                    future_loss.append(F.mse_loss(predicted, hidden.detach()))
                last_hidden = hidden
            loss = (torch.stack(action_loss).mean() + 0.30 * torch.stack(budget_loss).mean()
                    + (0.15 * torch.stack(future_loss).mean() if future_loss else 0.0))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(brain.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
        history.append(round(total_loss / len(episodes), 6))
    brain.eval()
    return brain, {"config": vars(config), "episodes": len(episodes), "train_losses": history,
                   "objective": "supervised verified shortest-path demonstrations + proxy compute labels + latent consistency",
                   "train_eval_maps_disjoint": not ({m.rows for m in training} & {m.rows for m in evaluation})}


@dataclass
class FlySession:
    """Explicit per-episode memory. Never shared implicitly between users/tasks."""
    brain: FlyBrain
    hidden: torch.Tensor | None = None
    last_action: int | None = None
    last_success: bool = False

    def reset(self) -> None:
        self.hidden, self.last_action, self.last_success = None, None, False

    def accept_outcome(self, action: int, success: bool) -> None:
        if type(action) is not int or not 0 <= action < 4 or type(success) is not bool:
            raise ValueError("invalid outcome")
        self.last_action = action
        self.last_success = success

    def observe(self, maze: Maze, state: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tensor([observe_features(maze, state, self.last_action, self.last_success)])
        hidden = self.brain.initial_state() if self.hidden is None else self.hidden
        with torch.inference_mode():
            actions, budgets, self.hidden = self.brain.step(x, hidden)
        return actions[0], budgets[0], self.hidden


def decide(brain: FlyBrain, maze: Maze, state: tuple[int, int], transition_predictor: Predictor,
           session: FlySession | None = None, forced_budget: int | None = None,
           max_beam: int = 8) -> dict:
    """Learned weighting and bounded hypothetical rollouts, no real actions.

    Rollouts use the separate learned transition predictor and the FlyBrain's
    imagined hidden states. Neither can claim that an action was executed.
    """
    if forced_budget is not None and forced_budget not in BUDGETS:
        raise ValueError("unsupported forced budget")
    if not 1 <= max_beam <= 32:
        raise ValueError("max_beam must be 1..32")
    brain.eval()
    memory = session if session is not None else FlySession(brain)
    if memory.brain is not brain:
        raise ValueError("session belongs to another model")
    with torch.inference_mode():
        logits, budget_logits, hidden = memory.observe(maze, state)
        budget_probs = budget_logits.softmax(-1)
        depth = forced_budget if forced_budget is not None else BUDGETS[int(budget_probs.argmax())]
        probs = logits.softmax(-1)
        table = make_transition_table(maze, transition_predictor)
        gx, gy = maze.goal
        def dist(s):
            return abs(s[0] - gx) + abs(s[1] - gy)
        # beam item: (score, state, imagined hidden, first action, visited)
        beam = [(0.0, state, hidden, None, frozenset([state]))]
        expanded = 0
        reached_goal = False
        for _ in range(depth):
            next_beam = []
            for score, where, latent, first, visited in beam:
                if where == maze.goal:
                    next_beam.append((score + 8.0, where, latent, first, visited))
                    reached_goal = True
                    continue
                preferences = brain.policy(latent).log_softmax(-1).tolist()[0]
                for a in range(4):
                    nxt = table[where, a]
                    if nxt == where or nxt in visited:
                        continue
                    imagined = brain.imagine(latent, torch.tensor([a]))
                    gain = (dist(where) - dist(nxt))
                    next_score = score + 0.90 * gain + 0.25 * preferences[a] - 0.08
                    if nxt == maze.goal:
                        next_score += 8.0
                    next_beam.append((next_score, nxt, imagined, a if first is None else first,
                                      visited | {nxt}))
                    expanded += 1
            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[:max_beam]
        if not expanded:
            action = None
        else:
            valid = [b for b in beam if b[3] is not None]
            action = max(valid, key=lambda b: b[0])[3] if valid else None
        return {"action_id": action, "action": ACTION_NAMES[action] if action is not None else None,
                "policy_probabilities": [round(float(x), 6) for x in probs.tolist()],
                "budget_probabilities": [round(float(x), 6) for x in budget_probs.tolist()],
                "thinking_budget": depth, "imagined_expansions": expanded,
                "hypothetical_goal_found": reached_goal,
                "verification": "not_executed", "transition_source": "learned_model",
                "note": "No real action was executed; imagined states can be incorrect."}


def roll_out_episode(brain: FlyBrain, maze: Maze, predictor: Predictor,
                     forced_budget: int | None = None, max_steps: int = 64) -> dict:
    """Evaluate decisions through independent real replay outside the planner."""
    session = FlySession(brain)
    state = maze.start
    budgets, steps = [], 0
    for _ in range(max_steps):
        if state == maze.goal:
            break
        decision = decide(brain, maze, state, predictor, session, forced_budget)
        a = decision["action_id"]
        if a is None:
            break
        nxt = maze.move(state, a)  # only here: independently verified real action
        session.accept_outcome(a, nxt != state)
        budgets.append(decision["thinking_budget"])
        steps += 1
        state = nxt
    return {"success": state == maze.goal, "steps": steps, "budgets": budgets}


def evaluate_fly(brain: FlyBrain, world_model, config: FlyTrainConfig) -> dict:
    """Disjoint maps, independent replay, matched-budget no-imagination ablation."""
    from .experiment import make_datasets, TrainConfig
    from .planning import brain_predictor, greedy_baseline, plan, replay
    _, maps = make_datasets(TrainConfig(seed=config.seed, train_maps=config.train_maps,
                                        test_maps=config.test_maps, size=config.size))
    world = brain_predictor(world_model)
    outcomes = [roll_out_episode(brain, m, world) for m in maps]
    no_imagination = [roll_out_episode(brain, m, world, forced_budget=1) for m in maps]
    baseline = [greedy_baseline(m) for m in maps]
    a_star = [replay(m, plan(m, world)) for m in maps]
    total_budget = sum(sum(r["budgets"]) for r in outcomes)
    steps = sum(r["steps"] for r in outcomes)
    return {"held_out_maps": len(maps), "fly_successes": sum(r["success"] for r in outcomes),
            "same_policy_depth_1_successes": sum(r["success"] for r in no_imagination),
            "greedy_baseline_successes": sum(baseline), "a_star_world_model_successes": sum(a_star),
            "mean_selected_budget": round(total_budget / steps, 3) if steps else None,
            "actual_executed_steps": steps,
            "note": "All success claims come from real replay. This is a toy maze benchmark, not LLM reasoning proof."}
