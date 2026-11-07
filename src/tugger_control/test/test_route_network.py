"""Chains of saved paths through pass-through stops (pure graph search)."""

from tugger_control.route_network import plan_route_chain


def _edge(source, gate_out, target, gate_in, length=20.0, approved=True):
    return {'key': f'{source}-{gate_out}-{target}', 'source': source, 'target': target,
            'source_entrance': gate_out, 'target_entrance': gate_in,
            'length_m': length, 'approved': approved}


EDGES = [
    _edge('S3', 'south', 'S4', 'north', 14.0),
    _edge('S3', 'west', 'HOME', 'east', 23.6),
    _edge('HOME', 'east', 'S4', 'north', 24.1),
    _edge('S3', 'west', 'S2', 'east', 35.0),
    _edge('S2', 'east', 'S4', 'west', 40.0),
]


def test_direct_path_wins_when_it_leaves_the_entry_gate():
    assert plan_route_chain(EDGES, 'S3', 'south', 'S4', ['HOME']) == ['S4']


def test_blocked_gate_is_reached_through_home():
    # Entered Station 3 by its west gate; no west-gate path goes to Station 4.
    assert plan_route_chain(EDGES, 'S3', 'west', 'S4', ['HOME']) == ['HOME', 'S4']


def test_any_station_gate_can_be_a_waypoint_left_by_the_same_gate():
    # Without HOME, Station 2's east gate is the waypoint: the robot arrives
    # there and the next path leaves from that same gate. No docking at S2.
    edges = [e for e in EDGES if e['source'] != 'HOME']
    assert plan_route_chain(edges, 'S3', 'west', 'S4', ['HOME']) == ['S2', 'S4']
    other_gate = [dict(e, source_entrance='south') if e['source'] == 'S2' else e
                  for e in edges]
    assert plan_route_chain(other_gate, 'S3', 'west', 'S4', ['HOME']) is None


def test_unapproved_paths_are_not_used():
    edges = [dict(e, approved=False) if e['source'] in ('HOME', 'S2') else e for e in EDGES]
    assert plan_route_chain(edges, 'S3', 'west', 'S4', ['HOME']) is None
    edges = [dict(e, approved=False) if e['source'] == 'HOME' else e for e in EDGES]
    assert plan_route_chain(edges, 'S3', 'west', 'S4', ['HOME']) == ['S2', 'S4']


def test_shortest_connection_is_chosen_not_the_first_found():
    edges = [_edge('S3', 'west', 'HOME', 'east', 23.6),
             _edge('S3', 'west', 'HOME', 'east', 60.0),          # a longer way to HOME
             _edge('HOME', 'east', 'S4', 'north', 24.1),
             _edge('HOME', 'east', 'S4', 'west', 90.0)]
    stops = plan_route_chain(edges, 'S3', 'west', 'S4', ['HOME'])
    assert stops == ['HOME', 'S4']


def test_a_direct_path_beats_a_slightly_shorter_chain_with_a_stop():
    edges = [_edge('S3', 'west', 'S4', 'west', 45.0),
             _edge('S3', 'west', 'HOME', 'east', 20.0),
             _edge('HOME', 'east', 'S4', 'north', 20.0)]
    # 40 m of driving plus a 10 m stop cost (50 m) loses to 45 m direct.
    assert plan_route_chain(edges, 'S3', 'west', 'S4', ['HOME']) == ['S4']
