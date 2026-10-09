import pytest

from latent_brain.world import Maze, dataset, features, make_mazes


def test_physics_collision_and_boundaries():
    maze = Maze((".#.", "...", "..."))
    assert maze.move((0, 0), 0) == (0, 0)  # wall
    assert maze.move((0, 0), 2) == (0, 0)  # left boundary
    assert maze.move((0, 0), 1) == (0, 1)
    with pytest.raises(ValueError):
        maze.move((0, 0), 9)


def test_observation_has_no_explicit_transition_label():
    maze = Maze((".#.", "...", "..."))
    assert len(features(maze, (0, 0), 0)) == 15
    assert len(features(maze, (0, 0), 1)) == 15


def test_maps_are_reproducible_and_reachable():
    a = make_mazes(99, 7)
    b = make_mazes(99, 7)
    assert a == b
    assert all(m.reachable() for m in a)


def test_ground_truth_transitions_are_supervised():
    xs, ys = dataset([Maze((".#.", "...", "..."))])
    assert len(xs) == len(ys) == 8 * 4
    assert set(ys) == {0, 1}
