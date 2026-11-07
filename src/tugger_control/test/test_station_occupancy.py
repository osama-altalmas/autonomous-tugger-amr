"""Station cargo state and the operator's ATTACH/DETACH choice (pure, no ROS)."""

from tugger_control.station_occupancy import (
    ATTACH, DETACH, assess_destination, load_occupancy, measure_occupancy, save_occupancy)

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


def test_any_station_can_attach_or_detach():
    # Pick up at Station 3 and set down at Station 4, then the other way.
    assert assess_destination('cargo_station_3', None, START, ATTACH) == ([], [], False)
    after = dict(START, cargo_station_3=None)
    assert assess_destination('cargo_dropoff_southeast', 3, after, DETACH) == ([], [], False)
    back = dict(after, cargo_dropoff_southeast=3)
    assert assess_destination('cargo_dropoff_southeast', None, back, ATTACH) == ([], [], False)
    assert assess_destination('cargo_station_3', 3, after, DETACH) == ([], [], False)


def test_impossible_actions_are_blocked_and_the_lift_never_moves():
    empty = dict(START, cargo_station_0=None)
    assert 'no cargo to attach' in assess_destination('cargo_station_0', None, empty, ATTACH)[0][0]
    assert 'nothing more to attach' in assess_destination(
        'cargo_dropoff_northwest', 0, empty, ATTACH)[0][0]
    assert 'nothing to detach' in assess_destination(
        'cargo_dropoff_northwest', None, START, DETACH)[0][0]
    assert 'no stand' in assess_destination('HOME', None, START, ATTACH)[0][0]


def test_loaded_robot_cannot_dock_under_a_stand_that_holds_cargo():
    blocks, _, _ = assess_destination('cargo_station_3', 0, START, DETACH)
    assert 'drop the carried cargo off first' in blocks[0]


def test_no_action_selected_is_a_dock_only_warning():
    blocks, warnings, dock_only = assess_destination('cargo_station_0', None, START, None)
    assert not blocks and 'no cargo action selected' in warnings[0] and dock_only
    assert assess_destination('HOME', 3, START, None) == ([], [], True)


def test_unknown_state_is_a_warning_not_a_guess(tmp_path):
    _, warnings, _ = assess_destination('cargo_station_0', None, None, ATTACH)
    assert 'unknown' in warnings[0]
    save_occupancy(START, tmp_path / 'state.json')
    assert load_occupancy(tmp_path / 'state.json') == START
    assert load_occupancy(tmp_path / 'missing.json') is None
