"""Station cargo state and mission sense checks (pure, no ROS)."""

from tugger_control.station_occupancy import (
    assess_destination, load_occupancy, measure_occupancy, save_occupancy)

CENTERS = {'HOME': [0.0, 0.0], 'cargo_dropoff_northwest': [-17.5, 7.0],
           'cargo_dropoff_southeast': [17.5, -7.0], 'cargo_station_0': [-17.5, -7.0],
           'cargo_station_3': [17.5, 7.0]}
START = {'cargo_station_0': 0, 'cargo_station_3': 3,
         'cargo_dropoff_northwest': None, 'cargo_dropoff_southeast': None}


def test_simulation_start_has_stations_1_and_3_loaded():
    poses = {0: (-17.5, -7.0, 0.28, 0.0), 3: (17.49, 7.01, 0.28, 0.0)}
    assert measure_occupancy(poses, None, CENTERS) == START


def test_carried_cargo_occupies_no_station_and_unknown_pose_gives_no_state():
    poses = {0: (-10.0, 0.0, 0.34, 0.0), 3: (17.5, 7.0, 0.28, 0.0)}
    state = measure_occupancy(poses, 0, CENTERS)
    assert state['cargo_station_0'] is None and state['cargo_station_3'] == 3
    assert measure_occupancy({0: None, 3: (17.5, 7.0, 0, 0)}, None, CENTERS) is None


def test_loaded_robot_to_another_pickup_must_drop_off_first():
    blocks, warnings, _ = assess_destination('cargo_station_3', 0, START)
    assert blocks == ['Robot is carrying Station 1 cargo: drop it off first (at Station 2)']


def test_normal_cycle_has_no_warnings():
    assert assess_destination('cargo_station_0', None, START) == ([], [], False)
    after_pickup = dict(START, cargo_station_0=None)
    assert assess_destination('cargo_dropoff_northwest', 0, after_pickup) == ([], [], False)
    assert assess_destination('HOME', 3, START) == ([], [], False)


def test_unusual_but_safe_missions_are_warnings():
    empty = dict(START, cargo_station_0=None)
    _, warnings, dock_only = assess_destination('cargo_station_0', None, empty)
    assert 'no cargo to pick up' in warnings[0] and dock_only
    _, warnings, dock_only = assess_destination('cargo_dropoff_northwest', None, START)
    assert 'no cargo to drop off' in warnings[0] and dock_only
    after = dict(START, cargo_station_3=None)
    blocks, warnings, dock_only = assess_destination('cargo_dropoff_northwest', 3, after)
    assert not blocks and 'belongs at Station 4' in warnings[0] and not dock_only


def test_cargo_onto_cargo_is_blocked():
    occupied = dict(START, cargo_dropoff_northwest=0, cargo_station_0=None)
    blocks, _, _ = assess_destination('cargo_dropoff_northwest', 3, occupied)
    assert 'already holds Station 1 cargo' in blocks[0]


def test_unknown_state_is_a_warning_not_a_guess(tmp_path):
    _, warnings, _ = assess_destination('cargo_station_0', None, None)
    assert 'unknown' in warnings[0]
    save_occupancy(START, tmp_path / 'state.json')
    assert load_occupancy(tmp_path / 'state.json') == START
    assert load_occupancy(tmp_path / 'missing.json') is None
