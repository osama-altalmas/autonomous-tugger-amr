"""Safety fields: sizing, shape along the arc, station mode, restart and e-stop."""

from tugger_control.safety_fields import (
    CLEAR, classify_points, EMERGENCY_STOP, MAX_SPEED, protective_length, PROTECTIVE_STOP,
    RestartTimer, speed_limit, STATION_SPEED, stopping_distance, WARNING, WARNING_SPEED)


def test_protective_field_covers_the_stopping_distance_plus_supplement():
    # 1.0 m/s: 0.17 s reaction travel + 0.5 m braking at 1 m/s^2.
    assert abs(stopping_distance(1.0) - 0.67) < 1e-9
    assert abs(protective_length(1.0) - (0.67 + 0.065 + 0.10)) < 1e-9
    assert protective_length(0.3) < protective_length(1.0)


def test_object_ahead_stops_beside_slows_behind_ignored():
    ahead = [(0.44 + 0.5, 0.0)]
    assert classify_points(ahead, 1.0, 0.0)[0] == PROTECTIVE_STOP
    further = [(0.44 + 1.8, 0.0)]
    assert classify_points(further, 1.0, 0.0)[0] == WARNING
    behind = [(-0.44 - 0.5, 0.0)]
    assert classify_points(behind, 1.0, 0.0)[0] == CLEAR
    # Reversing: the field is behind.
    assert classify_points(behind, -1.0, 0.0)[0] == PROTECTIVE_STOP


def test_field_follows_the_curve_instead_of_the_straight_line():
    # Turning left on a 0.5 m radius: a rack 0.35 m ahead and 0.36 m to the
    # right is in the straight field but beside the arc actually driven.
    rack_ahead = [(0.44 + 0.35, -0.36)]
    assert classify_points(rack_ahead, 0.6, 0.0)[0] == PROTECTIVE_STOP
    assert classify_points(rack_ahead, 0.6, 2.0)[0] != PROTECTIVE_STOP


def test_own_body_points_are_ignored():
    assert classify_points([(0.30, 0.20), (-0.43, -0.28)], 1.0, 0.0)[0] == CLEAR


def test_station_fields_let_stand_legs_pass_beside_the_robot():
    legs = [(0.56, 0.56), (0.56, -0.56), (-0.56, 0.56), (-0.56, -0.56)]
    assert classify_points(legs, 0.3, 0.0, station=True)[0] == CLEAR
    # Outside stations the wider warning field sees them.
    assert classify_points(legs, 0.3, 0.0)[0] == WARNING
    # Something straight ahead still stops the robot at a station.
    assert classify_points([(0.6, 0.0)], 0.3, 0.0, station=True)[0] == PROTECTIVE_STOP


def test_turning_ring_clears_the_legs_and_stops_for_a_closer_object():
    legs = [(0.56, 0.56), (-0.56, -0.56)]
    assert classify_points(legs, 0.0, 0.0, turning=True, turn_radius=0.6)[0] != PROTECTIVE_STOP
    assert classify_points([(0.5, 0.35)], 0.0, 0.0, turning=True,
                           turn_radius=0.6)[0] == PROTECTIVE_STOP


def test_speed_limits():
    assert speed_limit(CLEAR) == MAX_SPEED == 1.0
    assert speed_limit(WARNING) == WARNING_SPEED
    assert speed_limit(CLEAR, station=True) == STATION_SPEED == 0.3
    assert speed_limit(PROTECTIVE_STOP) == speed_limit(EMERGENCY_STOP) == 0.0


def test_protective_stop_restarts_after_two_clear_seconds():
    timer = RestartTimer()
    assert timer.update(PROTECTIVE_STOP, 10.0) == PROTECTIVE_STOP
    assert timer.update(CLEAR, 10.5) == PROTECTIVE_STOP
    assert timer.update(CLEAR, 12.0) == PROTECTIVE_STOP
    assert timer.update(CLEAR, 12.6) == CLEAR


def test_emergency_stop_latches_until_reset():
    timer = RestartTimer()
    timer.trigger_emergency('emergency stop panel pressed')
    assert timer.update(CLEAR, 1.0) == EMERGENCY_STOP
    assert timer.update(CLEAR, 100.0) == EMERGENCY_STOP
    timer.reset()
    # After a reset the robot still waits for 2 s of clear fields.
    assert timer.update(CLEAR, 101.0) == PROTECTIVE_STOP
    assert timer.update(CLEAR, 103.5) == CLEAR
