import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_route_database_v1 import (centre_cost_map, constrained_astar,
                                     fast_turn_limited_astar)


def test_centre_cost_prefers_the_middle_of_a_free_corridor():
    safe = np.ones((9, 13), dtype=bool)
    cost = np.ones_like(safe, dtype=float)
    cost[4, :] = 0.0

    path = constrained_astar(
        (1, 2), (11, 6), safe, terminal_direction=0,
        centre_cost=cost, centre_weight=2.0)

    assert any(y == 4 for _, y in path)


def _open_crossing():
    """A 61-cell-wide hall with a 15-cell-tall open crossing in the middle."""
    blocked = np.zeros((41, 61), dtype=bool)
    blocked[0, :] = blocked[-1, :] = blocked[:, 0] = blocked[:, -1] = True
    blocked[5:13, 10:51] = True    # rack row above the crossing
    blocked[28:36, 10:51] = True   # rack row below it
    return blocked


def test_centre_line_runs_through_the_middle_of_free_space():
    cost = centre_cost_map(_open_crossing(), 1.0)
    # The crossing spans rows 13-27; its centre line is row 20.
    assert cost[20, 30] == 0.0
    assert cost[14, 30] > 0.7 and cost[26, 30] > 0.7
    assert cost[17, 30] > cost[19, 30] > cost[20, 30]


def test_open_crossing_is_crossed_in_the_middle_not_along_a_rack():
    """The fault seen in the warehouse: a route hugging a rack row because the
    middle of the crossing costs a short detour."""
    blocked = _open_crossing()
    safe = ~blocked
    for y in range(41):
        for x in range(61):
            if blocked[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any():
                safe[y, x] = False
    cost = centre_cost_map(blocked, 1.0)
    # Straight along the lower rack row is shortest; the middle costs a detour.
    path = fast_turn_limited_astar((5, 26), (55, 26), safe, 0, 3, axis=0,
                                   centre_cost=cost, centre_weight=2.0, factor=2)
    middle = [y for x, y in path if 15 <= x <= 45]
    assert middle and all(abs(y - 20) <= 1 for y in middle)


def test_departure_gate_follows_the_mission_that_arrived():
    from build_route_database_v1 import required_departure_entrance

    chains = [['HOME', 'S1', 'S2', 'HOME']]
    routes = {'HOME__TO__S1': {'target_entrance': 'east'},
              'S1__TO__S2': {'target_entrance': 'south'},
              'HOME__TO__S2': {'target_entrance': 'east'}}
    fallback = {'S1': 'east', 'S2': 'east'}
    assert required_departure_entrance('S2', 'HOME', routes, chains, fallback) == 'south'
    assert required_departure_entrance('S1', 'S2', routes, chains, fallback) == 'east'
    # Not a mission leg: keep the gate used when arriving from HOME.
    assert required_departure_entrance('S2', 'S1', routes, chains, fallback) == 'east'


def test_pass_through_stop_may_be_left_by_any_gate():
    from build_route_database_v1 import required_departure_entrance

    chains = [['S4', 'HOME', 'S1']]
    routes = {'S4__TO__HOME': {'target_entrance': 'east'}}
    assert required_departure_entrance(
        'HOME', 'S1', routes, chains, {}, pass_through=('HOME',)) is None
    assert required_departure_entrance('HOME', 'S1', routes, chains, {}) == 'east'


def test_corner_cost_keeps_a_straight_run_instead_of_a_side_step():
    """A side-step of a few cells toward a slightly cheaper lane is not worth
    two corners."""
    safe = np.ones((21, 41), dtype=bool)
    cost = np.full(safe.shape, 0.2)
    cost[:, 18] = 0.0      # a marginally more central lane one step aside
    path = constrained_astar((20, 20), (20, 0), safe, terminal_direction=3,
                             centre_cost=cost, centre_weight=2.0, corner_penalty=60.0)
    assert {x for x, _ in path} == {20}
