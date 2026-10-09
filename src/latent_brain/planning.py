"""Model-based lookahead with learned transitions (no ground-truth in planning)."""
from __future__ import annotations

import heapq
from typing import Callable

import torch

from .world import ACTIONS, Maze, features

Predictor = Callable[[Maze, list[tuple[tuple[int, int], int]]], list[bool]]


def brain_predictor(brain) -> Predictor:
    brain.eval()

    def predict(maze: Maze, queries: list[tuple[tuple[int, int], int]]) -> list[bool]:
        if not queries:
            return []
        xs = torch.tensor([features(maze, state, action) for state, action in queries], dtype=torch.float32)
        probs = brain.probability(xs).tolist()
        return [prob >= 0.5 for prob in probs]

    return predict


def make_transition_table(maze: Maze, predictor: Predictor) -> dict[tuple[tuple[int, int], int], tuple[int, int]]:
    """Construct hypothetical successors ONLY from the brain predictions."""
    # Include all in-bounds hypothetical states, including locations the true
    # environment considers walls. Never silently use wall truth to prune plans.
    questions = [((x, y), a) for y in range(maze.height) for x in range(maze.width) for a in range(4)]
    answers = predictor(maze, questions)
    if len(answers) != len(questions):
        raise ValueError("predictor returned wrong number of answers")
    table = {}
    for (state, action), can_move in zip(questions, answers):
        dx, dy = ACTIONS[action]
        # No call to maze.move() and no blocked() checks on candidate next state.
        nx, ny = state[0] + dx, state[1] + dy
        in_bounds = 0 <= nx < maze.width and 0 <= ny < maze.height
        table[state, action] = (nx, ny) if can_move and in_bounds else state
    return table


def plan(maze: Maze, predictor: Predictor, max_nodes: int = 5000) -> list[int] | None:
    """A* over simulated transitions. A complete plan is verified by real replay."""
    table = make_transition_table(maze, predictor)
    start, goal = maze.start, maze.goal
    def h(s):
        return abs(s[0] - goal[0]) + abs(s[1] - goal[1])
    fringe = [(h(start), 0, 0, start)]
    best = {start: 0}
    parent = {}
    tiebreak = 0
    visited = 0
    while fringe and visited < max_nodes:
        _, g, _, state = heapq.heappop(fringe)
        if g != best.get(state):
            continue
        visited += 1
        if state == goal:
            result = []
            while state != start:
                old, action = parent[state]
                result.append(action)
                state = old
            return result[::-1]
        for action in range(4):
            nxt = table.get((state, action), state)
            if nxt == state:
                continue
            new_g = g + 1
            if new_g < best.get(nxt, 10**9):
                best[nxt] = new_g
                parent[nxt] = (state, action)
                tiebreak += 1
                heapq.heappush(fringe, (new_g + h(nxt), new_g, tiebreak, nxt))
    return None


def replay(maze: Maze, actions: list[int] | None) -> bool:
    if actions is None:
        return False
    state = maze.start
    for action in actions:
        state = maze.move(state, action)  # independent execution/evaluation truth
    return state == maze.goal


def greedy_baseline(maze: Maze, max_steps: int = 100) -> bool:
    """One-step greedy heuristic, with the SAME public map information.

    It knows valid neighbors, but never searches through temporarily worse states.
    """
    state = maze.start
    goal = maze.goal
    seen = {state}
    for _ in range(max_steps):
        if state == goal:
            return True
        distance = abs(state[0] - goal[0]) + abs(state[1] - goal[1])
        choices = []
        for action in range(4):
            nxt = maze.move(state, action)
            if nxt == state or nxt in seen:
                continue
            d = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
            if d < distance:
                choices.append((d, action, nxt))
        if not choices:
            return False
        _, _, state = min(choices)
        seen.add(state)
    return state == goal


def oracle_plan(maze: Maze) -> list[int] | None:
    def oracle(m, queries):
        return [m.move(s, a) != s for s, a in queries]
    return plan(maze, oracle)
