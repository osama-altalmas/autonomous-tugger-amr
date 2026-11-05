import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_route_database_v1 import constrained_astar


def test_clearance_bias_prefers_the_middle_of_a_free_corridor():
    safe = np.ones((9, 13), dtype=bool)
    clearance = np.ones_like(safe, dtype=float)
    clearance[4, :] = 4.0

    path = constrained_astar(
        (1, 2), (11, 6), safe, terminal_direction=0,
        clearance_map=clearance, clearance_weight=2.0)

    assert any(y == 4 for _, y in path)


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
