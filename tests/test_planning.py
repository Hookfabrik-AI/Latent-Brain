from latent_brain.planning import greedy_baseline, oracle_plan, plan, replay
from latent_brain.world import Maze


def test_oracle_finds_route_around_obstacle():
    maze = Maze((".#.", "...", "..."))
    route = oracle_plan(maze)
    assert route is not None
    assert replay(maze, route)
    assert len(route) == 4


def test_planner_does_not_cheat_with_real_transition():
    maze = Maze(("...", "...", "..."))
    def all_blocked(_maze, queries):
        return [False] * len(queries)
    assert plan(maze, all_blocked) is None
    assert not replay(maze, None)


def test_wrong_simulation_is_caught_by_real_execution():
    maze = Maze((".#.", "...", "..."))
    def hallucinate_all_open(_maze, queries):
        return [True] * len(queries)
    route = plan(maze, hallucinate_all_open)
    assert route is not None
    assert not replay(maze, route)


def test_greedy_gets_stuck_on_detour():
    maze = Maze((".....", "##.##", ".....", ".....", "....."))
    assert maze.reachable()
    assert not greedy_baseline(maze)  # greedily moves right to dead end on top row
