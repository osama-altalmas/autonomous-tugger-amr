"""Rules for unexpected objects: size and placement decide what happens."""
import math

from tugger_control.obstacle_monitor import Tracks, cluster
from tugger_control.obstacle_policy import (
    ALERT, AVOID, CONTINUE, GAP_ROUTE_M, GAP_STATION_M, HOLD, ObstacleSupervisor,
    ROBOT_HALF_WIDTH_M, effective_phase, interfering, path_ahead, plan_detour, WAIT_BEFORE_AVOID_S)
import numpy as np

LINE = [(i * 0.1, 0.0) for i in range(200)]     # 20 m along x


def obj(x, y, r=0.1, moving=False):
    return {'x': x, 'y': y, 'r': r, 'moving': moving}


def test_only_objects_close_to_the_path_count():
    half = ROBOT_HALF_WIDTH_M
    assert interfering([obj(5, 0)], LINE, half, GAP_ROUTE_M)
    assert interfering([obj(5, 0.8)], LINE, half, GAP_ROUTE_M)
    assert not interfering([obj(5, 1.2)], LINE, half, GAP_ROUTE_M)


def test_a_small_object_beside_a_station_path_is_ignored():
    assert not interfering([obj(1, 0.7, 0.04)], LINE, ROBOT_HALF_WIDTH_M, GAP_STATION_M)
    assert interfering([obj(1, 0.3, 0.04)], LINE, ROBOT_HALF_WIDTH_M, GAP_STATION_M)


def test_only_the_path_ahead_is_looked_at():
    ahead = path_ahead(LINE, (5.0, 0.0), 2.0)
    assert math.isclose(ahead[0][0], 5.0) and ahead[-1][0] <= 7.1


def test_route_waits_then_avoids_a_standing_object():
    s = ObstacleSupervisor(wait=10.0)
    assert s.update(0.0, 'route', [obj(5, 0)]) == HOLD
    assert s.update(9.0, 'route', [obj(5, 0)]) == HOLD
    assert s.update(10.5, 'route', [obj(5, 0)]) == AVOID


def test_a_moving_object_is_waited_for_and_never_avoided():
    s = ObstacleSupervisor(wait=10.0)
    for t in range(0, 30, 2):
        assert s.update(float(t), 'route', [obj(5, 0, moving=True)]) == HOLD


def test_a_station_pauses_and_alerts_until_the_object_is_gone():
    s = ObstacleSupervisor(wait=10.0, clear=1.0)
    assert s.update(0.0, 'station', [obj(1, 0)]) == ALERT
    assert s.update(60.0, 'station', [obj(1, 0)]) == ALERT
    assert s.update(60.5, 'station', []) == ALERT     # not yet clear for long enough
    assert s.update(61.5, 'station', []) == CONTINUE


def test_detour_keeps_the_gap_and_returns_to_the_route():
    o = obj(10, 0.1, 0.2)
    bent = plan_detour(LINE, (0.0, 0.0), o, ROBOT_HALF_WIDTH_M, lambda p: True)
    assert bent is not None
    assert not interfering([o], bent, ROBOT_HALF_WIDTH_M, GAP_ROUTE_M - 1e-6)
    assert bent[0] == LINE[0] and bent[-1] == LINE[-1]


def test_detour_picks_the_other_side_when_one_is_blocked():
    o = obj(10, 0.0, 0.2)
    bent = plan_detour(LINE, (0.0, 0.0), o, ROBOT_HALF_WIDTH_M,
                       lambda p: all(y <= 0.01 for _, y in p))
    assert bent is not None and min(y for _, y in bent) < -0.5


def test_no_detour_without_room_or_distance():
    o = obj(10, 0, 0.2)
    assert plan_detour(LINE, (0, 0), o, ROBOT_HALF_WIDTH_M, lambda p: False) is None
    assert plan_detour(LINE, (9.0, 0), o, ROBOT_HALF_WIDTH_M, lambda p: True) is None


def test_detour_stays_within_car_like_steering():
    o = obj(10, 0, 0.5)
    b = plan_detour(LINE, (0, 0), o, ROBOT_HALF_WIDTH_M, lambda p: True)
    k = []
    for a, m, c in zip(b[::3], b[3::3], b[6::3]):
        h1, h2 = math.atan2(m[1] - a[1], m[0] - a[0]), math.atan2(c[1] - m[1], c[0] - m[0])
        k.append(abs(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1))) / math.dist(a, m))
    assert max(k) < 1.0 / 0.29


def test_points_group_into_objects_with_a_size():
    pts = np.array([[1.0, 1.0], [1.05, 1.0], [1.1, 1.0], [5.0, 5.0], [5.04, 5.0], [8.0, 8.0]])
    found = cluster(pts)
    assert len(found) == 2          # a lone point is noise
    assert min(abs(f[0] - 1.05) for f in found) < 0.01


def test_tracks_confirm_objects_and_tell_moving_from_standing():
    t = Tracks()
    assert t.update(0.0, [(1, 1, 0.1)]) == []
    t.update(0.2, [(1, 1, 0.1)])
    still = t.update(0.4, [(1, 1, 0.1)])
    assert len(still) == 1 and not still[0]['moving']
    m = Tracks()
    out = []
    for i in range(12):
        out = m.update(i * 0.2, [(i * 0.2 * 0.8, 0, 0.1)])
    assert out and out[0]['moving']


def test_an_object_at_the_destination_is_a_station_matter_even_from_the_route():
    centre = (-17.5, -7.0)
    assert effective_phase('route', [obj(-16.5, -7.0)], centre) == 'station'
    assert effective_phase('route', [obj(-6.5, -3.0)], centre) == 'route'
    assert effective_phase('route', [], centre) == 'route'


def test_a_person_walking_towards_the_path_counts_before_reaching_it():
    walking = {'x': 5.0, 'y': 3.0, 'r': 0.25, 'moving': True, 'vx': 0.0, 'vy': -1.2}
    leaving = {'x': 5.0, 'y': 3.0, 'r': 0.25, 'moving': True, 'vx': 0.0, 'vy': 1.2}
    assert interfering([walking], LINE, ROBOT_HALF_WIDTH_M, GAP_ROUTE_M)
    assert not interfering([leaving], LINE, ROBOT_HALF_WIDTH_M, GAP_ROUTE_M)


def test_a_moving_object_blocking_for_30_seconds_means_another_path():
    s = ObstacleSupervisor()
    assert s.update(0.0, 'route', [obj(5, 0, moving=True)]) == HOLD
    assert s.update(29.0, 'route', [obj(5, 0, moving=True)]) == HOLD
    assert s.update(30.5, 'route', [obj(5, 0, moving=True)]) == AVOID


def test_at_a_station_a_moving_object_is_still_waited_for():
    s = ObstacleSupervisor()
    assert s.update(0.0, 'station', [obj(1, 0, moving=True)]) == ALERT
    assert s.update(120.0, 'station', [obj(1, 0, moving=True)]) == ALERT


def test_an_object_that_stops_gets_the_normal_wait_from_when_it_stopped():
    s = ObstacleSupervisor()
    assert s.update(0.0, 'route', [obj(5, 0, moving=True)]) == HOLD
    assert s.update(19.0, 'route', [obj(5, 0, moving=True)]) == HOLD
    assert s.update(20.0, 'route', [obj(5, 0)]) == HOLD
    assert s.update(19.0 + WAIT_BEFORE_AVOID_S - 0.5, 'route', [obj(5, 0)]) == HOLD
    assert s.update(19.0 + WAIT_BEFORE_AVOID_S + 0.5, 'route', [obj(5, 0)]) == AVOID


def test_a_standing_object_is_avoided_after_five_seconds():
    assert WAIT_BEFORE_AVOID_S == 5.0
    s = ObstacleSupervisor()
    assert s.update(0.0, 'route', [obj(5, 0)]) == HOLD
    assert s.update(4.5, 'route', [obj(5, 0)]) == HOLD
    assert s.update(5.0, 'route', [obj(5, 0)]) == AVOID


def test_a_jittering_standing_object_is_not_called_moving():
    t = Tracks()
    out = []
    for i in range(30):
        out = t.update(i * 0.2, [(1.0 + 0.12 * (-1) ** i, 1.0, 0.2)])
    assert out and not out[0]['moving']
