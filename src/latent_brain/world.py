"""Small, public toy environments with objectively checkable transitions."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random

# x, y offsets. Keep this order stable for checkpoints.
ACTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTION_NAMES = ("right", "down", "left", "up")


@dataclass(frozen=True)
class Maze:
    """A static known map; transition physics must be learned by the brain."""

    rows: tuple[str, ...]

    def __post_init__(self) -> None:
        assert self.rows and all(len(row) == len(self.rows[0]) for row in self.rows)
        assert all(set(row) <= {".", "#"} for row in self.rows)
        assert self.rows[0][0] == "." and self.rows[-1][-1] == "."

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def start(self) -> tuple[int, int]:
        return (0, 0)

    @property
    def goal(self) -> tuple[int, int]:
        return (self.width - 1, self.height - 1)

    def blocked(self, x: int, y: int) -> bool:
        return not (0 <= x < self.width and 0 <= y < self.height) or self.rows[y][x] == "#"

    def move(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        if not 0 <= action < len(ACTIONS):
            raise ValueError(f"invalid action: {action}")
        dx, dy = ACTIONS[action]
        nx, ny = state[0] + dx, state[1] + dy
        return state if self.blocked(nx, ny) else (nx, ny)

    def reachable(self) -> bool:
        seen = {self.start}
        queue = deque([self.start])
        while queue:
            state = queue.popleft()
            if state == self.goal:
                return True
            for action in range(len(ACTIONS)):
                nxt = self.move(state, action)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    def free_states(self):
        for y in range(self.height):
            for x in range(self.width):
                if not self.blocked(x, y):
                    yield (x, y)


def make_mazes(seed: int, count: int, size: int = 7, wall_probability: float = 0.24) -> list[Maze]:
    """Deterministic generation, with solvability verified by the real engine."""
    if count < 1 or size < 3 or not 0 <= wall_probability < 1:
        raise ValueError("invalid maze generation parameters")
    rng = Random(seed)
    mazes = []
    for _ in range(count):
        for _attempt in range(10000):
            rows = []
            for y in range(size):
                row = []
                for x in range(size):
                    endpoint = (x, y) in ((0, 0), (size - 1, size - 1))
                    row.append("#" if not endpoint and rng.random() < wall_probability else ".")
                rows.append("".join(row))
            maze = Maze(tuple(rows))
            if maze.reachable():
                mazes.append(maze)
                break
        else:
            raise RuntimeError("could not generate connected maze")
    return mazes


def features(maze: Maze, state: tuple[int, int], action: int) -> list[float]:
    """Local 3x3 geometry, normalized position and action, but NO transition result.

    The brain must infer whether the move will actually succeed.
    """
    x, y = state
    # During model-based lookahead, a predicted (possibly incorrect) transition
    # can reach an occupied cell. Keep those states representable so real replay
    # can detect the hallucination rather than hiding it inside the planner.
    patch = [float(maze.blocked(x + dx, y + dy)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    return [x / (maze.width - 1), y / (maze.height - 1)] + patch + [float(i == action) for i in range(4)]


def dataset(mazes: list[Maze]):
    """Every legal origin and each action; labels from independent real move()."""
    xs, ys = [], []
    for maze in mazes:
        for state in maze.free_states():
            for action in range(4):
                xs.append(features(maze, state, action))
                ys.append(int(maze.move(state, action) != state))
    return xs, ys
