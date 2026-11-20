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


def test_a_long_object_beside_the_path_is_judged_by_its_shape_not_a_circle():
    # A 1.0 x 0.5 m cart along the path, its near side 1.05 m from the centre
    # line: 0.72 m of free space beside the robot, more than the route gap.
    # Its circle (radius 0.56) would reach 25 cm closer and call it in the way.
    xs, ys = np.meshgrid(np.linspace(4.5, 5.5, 21), np.linspace(1.05, 1.55, 11))
    pts = np.column_stack((xs.ravel(), ys.ravel()))
    (x, y, r, shape), = cluster(pts)
    cart = {'x': x, 'y': y, 'r': r, 'outline': shape, 'moving': False}
    assert not interfering([cart], LINE, ROBOT_HALF_WIDTH_M, GAP_ROUTE_M)
    assert interfering([dict(cart, outline=None)], LINE, ROBOT_HALF_WIDTH_M, GAP_ROUTE_M)
    # The same cart turned across the path is in the way.
    across = np.column_stack((ys.ravel() - 1.3 + 0.0, xs.ravel() - 5.0 + 0.4))
    (x, y, r, shape), = cluster(across + np.array([5.0, 0.0]))
    assert interfering([{'x': x, 'y': y, 'r': r, 'outline': shape, 'moving': False}],
                       LINE, ROBOT_HALF_WIDTH_M, GAP_ROUTE_M)


def test_a_moving_object_outline_moves_with_its_prediction():
    walker = {'x': 5.0, 'y': 2.0, 'r': 0.3, 'moving': True, 'vx': 0.0, 'vy': -0.6,
              'outline': [(4.8, 1.8), (5.2, 1.8), (5.2, 2.2), (4.8, 2.2)]}
    assert interfering([walker], LINE, ROBOT_HALF_WIDTH_M, GAP_ROUTE_M)
    assert not interfering([dict(walker, moving=False)], LINE, ROBOT_HALF_WIDTH_M, GAP_ROUTE_M)


def test_points_on_a_line_still_give_an_outline():
    pts = np.array([[1.0 + 0.02 * i, 1.0] for i in range(10)])
    (_, _, _, shape), = cluster(pts)
    assert len(shape) == 2 and min(p[0] for p in shape) == 1.0


def test_standing_objects_15_to_50_cm_away_are_passed_slowly():
    from tugger_control.obstacle_policy import GAP_ROUTE_MIN_M, route_objects
    half = ROBOT_HALF_WIDTH_M
    edge = lambda gap: half + gap + 0.1          # centre of a 0.1 m object at that gap
    far, slow, close = obj(5, edge(0.6)), obj(6, edge(0.3)), obj(7, edge(0.10))
    in_way, to_slow = route_objects([far, slow, close], LINE, half)
    assert in_way == [close] and to_slow == [slow]
    assert route_objects([obj(5, edge(GAP_ROUTE_MIN_M + 0.01))], LINE, half) == (
        [], [obj(5, edge(GAP_ROUTE_MIN_M + 0.01))])


def test_a_person_still_needs_the_full_gap():
    from tugger_control.obstacle_policy import route_objects
    person = obj(5, ROBOT_HALF_WIDTH_M + 0.3 + 0.1, moving=True)
    in_way, to_slow = route_objects([person], LINE, ROBOT_HALF_WIDTH_M)
    assert in_way == [person] and to_slow == []


def test_while_holding_the_standing_object_must_clear_the_extra_gap():
    from tugger_control.obstacle_policy import HOLD_EXTRA_GAP_M, route_objects
    box = obj(5, ROBOT_HALF_WIDTH_M + 0.20 + 0.1)        # 20 cm away
    assert route_objects([box], LINE, ROBOT_HALF_WIDTH_M)[0] == []
    assert route_objects([box], LINE, ROBOT_HALF_WIDTH_M, HOLD_EXTRA_GAP_M)[0] == [box]


def test_a_standing_person_or_forklift_is_waited_for_like_a_moving_object():
    """Named by the colour cameras (AI): may move at any moment (user decision 2026-11-08)."""
    from tugger_control.obstacle_policy import may_move
    person = dict(obj(5, 0), cls='person')
    s = ObstacleSupervisor()
    assert s.update(0.0, 'route', [person]) == HOLD
    assert s.update(WAIT_BEFORE_AVOID_S + 1.0, 'route', [person]) == HOLD     # a box would be avoided now
    assert s.update(30.5, 'route', [person]) == AVOID                          # patience, then another way
    assert may_move(dict(obj(5, 0), cls='forklift'))
    assert not may_move(dict(obj(5, 0), cls='pallet')) and not may_move(dict(obj(5, 0), cls='cart'))
    assert not may_move(obj(5, 0))                                            # unnamed: as before


def test_a_standing_person_keeps_the_full_gap_a_named_pallet_does_not():
    from tugger_control.obstacle_policy import route_objects
    at = ROBOT_HALF_WIDTH_M + 0.3 + 0.1                                        # 30 cm from the robot's side
    person, pallet = dict(obj(5, at), cls='person'), dict(obj(6, at), cls='pallet')
    in_way, to_slow = route_objects([person, pallet], LINE, ROBOT_HALF_WIDTH_M)
    assert in_way == [person] and to_slow == [pallet]


def test_an_object_the_cameras_have_not_looked_at_is_approached_not_judged():
    from tugger_control.obstacle_policy import LOOK_STOP_M, split_for_look
    unseen = dict(obj(5, 0), id=1, looks=0)
    judged, approach = split_for_look([unseen], LINE)
    assert judged == [] and approach == [unseen]
    # Close to it, it is handled as today even without a look.
    near = dict(obj(LOOK_STOP_M, 0), id=2, looks=0)
    assert split_for_look([near], LINE) == ([near], [])


def test_looked_at_named_moving_or_untracked_by_recognition_objects_are_judged_as_today():
    from tugger_control.obstacle_policy import LOOKS_BEFORE_JUDGING, split_for_look
    looked = dict(obj(5, 0), id=1, looks=LOOKS_BEFORE_JUDGING)
    named = dict(obj(5, 0), id=2, looks=0, cls='pallet')
    moving = dict(obj(5, 0, moving=True), id=3, looks=0)
    no_recognition = dict(obj(5, 0), id=4)            # no 'looks': recognition not running
    objects = [looked, named, moving, no_recognition]
    assert split_for_look(objects, LINE) == (objects, [])


def test_the_5_s_clock_starts_only_once_the_cameras_have_looked():
    from tugger_control.obstacle_policy import split_for_look
    sup = ObstacleSupervisor()
    box = dict(obj(5, 0), id=1, looks=0)
    for t in range(20):                    # 10 s unseen: approached, never avoided
        judged, _ = split_for_look([box], LINE)
        assert sup.update(t * 0.5, 'route', judged) == CONTINUE
    box['looks'] = 2
    assert sup.update(10.0, 'route', split_for_look([box], LINE)[0]) == HOLD
    assert sup.update(10.0 + WAIT_BEFORE_AVOID_S, 'route', split_for_look([box], LINE)[0]) == AVOID


def test_distance_along_follows_the_path_round_a_corner():
    from tugger_control.obstacle_policy import distance_along
    corner = [(0.0, i * 0.1) for i in range(31)] + [(i * 0.1, 3.0) for i in range(1, 41)]
    assert math.isclose(distance_along(corner, (3.0, 3.0)), 6.0, abs_tol=1e-6)


def _block(cx, cy, size=0.145, step=0.03):
    xs = np.arange(cx - size / 2, cx + size / 2 + 1e-9, step)
    ys = np.arange(cy - size / 2, cy + size / 2 + 1e-9, step)
    return np.array([(x, y) for x in xs for y in ys])


def test_the_blocks_of_a_euro_pallet_are_one_object():
    # 1.2 x 0.8 m: 3 x 3 blocks, 0.38 m apart along the length, 0.18 m across.
    blocks = [_block(10 + dx, 7 + dy) for dx in (-0.528, 0.0, 0.528) for dy in (-0.328, 0.0, 0.328)]
    objects = cluster(np.vstack(blocks))
    assert len(objects) == 1
    assert math.isclose(objects[0][0], 10.0, abs_tol=0.03)


def test_a_persons_two_legs_are_one_object_and_objects_a_metre_apart_stay_two():
    legs = np.vstack([_block(5.0, 1.0, 0.12), _block(5.0, 1.3, 0.12)])
    assert len(cluster(legs)) == 1
    apart = np.vstack([_block(5.0, 1.0, 0.3), _block(5.0, 2.3, 0.3)])
    assert len(cluster(apart)) == 2


def test_a_standing_pallet_seen_partly_then_whole_is_not_taken_for_moving():
    near_end = np.vstack([_block(9.472, 7 + dy) for dy in (-0.328, 0.0, 0.328)])
    whole = np.vstack([_block(10 + dx, 7 + dy) for dx in (-0.528, 0.0, 0.528) for dy in (-0.328, 0.0, 0.328)])
    tracks = Tracks()
    for k in range(30):
        confirmed = tracks.update(k * 0.2, cluster(near_end if k % 4 == 0 else whole))
    assert len(confirmed) == 1 and not confirmed[0]['moving']


def _pallet_points(seen_from_x):
    """A Euro pallet's blocks at (10, 7), only those at or before ``seen_from_x``."""
    blocks = [_block(10 + dx, 7 + dy) for dx in (-0.528, 0.0, 0.528) for dy in (-0.328, 0.0, 0.328)
              if 10 + dx <= seen_from_x]
    return np.vstack(blocks)


def test_a_pallet_coming_into_view_corner_first_is_not_taken_for_moving():
    # First only the near corner block (round a corner), then the whole pallet:
    # the centre shifts 0.62 m while the pallet stands still.
    def seen(blocks):
        return np.vstack([_block(10 + dx, 7 + dy) for dx, dy in blocks])
    corner = [(-0.528, -0.328)]
    near_side = corner + [(0.0, -0.328), (0.528, -0.328)]
    two_rows = near_side + [(-0.528, 0.0), (0.0, 0.0), (0.528, 0.0)]
    whole = two_rows + [(-0.528, 0.328), (0.0, 0.328), (0.528, 0.328)]
    tracks = Tracks()
    views = [seen(corner)] * 4 + [seen(near_side)] * 3 + [seen(two_rows)] * 3 + [seen(whole)] * 20
    for k, points in enumerate(views):
        confirmed = tracks.update(k * 0.2, cluster(points))
        assert not any(t['moving'] for t in confirmed), k


def test_a_walking_person_and_a_driving_forklift_are_still_moving():
    for size, speed in ((0.12, 0.4), (1.0, 0.35)):     # legs at 0.4 m/s; a 1 m wide truck
        tracks = Tracks()
        for k in range(20):
            confirmed = tracks.update(k * 0.2, cluster(_block(3.0 + speed * k * 0.2, 1.0, size)))
        assert confirmed and confirmed[0]['moving'], size


def _box_piece(x0, x1, y0, y1):
    shape = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx, cy, math.hypot(x1 - cx, y1 - cy) + 0.03, shape)


def test_an_object_seen_whole_then_in_two_pieces_stays_one_steady_track():
    whole = [_box_piece(9.4, 10.6, 7.4, 8.2)]
    split = [_box_piece(9.4, 9.85, 7.4, 8.2), _box_piece(10.2, 10.6, 7.4, 8.2)]   # 0.35 m gap
    tracks = Tracks()
    for k in range(40):
        confirmed = tracks.update(k * 0.2, whole if k % 3 else split)
        assert len(confirmed) <= 1, k
    assert len(confirmed) == 1
    track = confirmed[0]
    assert not track['moving']
    assert math.isclose(track['x'], 10.0, abs_tol=0.1) and math.isclose(track['y'], 7.8, abs_tol=0.05)


def test_a_person_next_to_a_pallet_is_a_separate_track_and_walks_away_as_one():
    from tugger_control.obstacle_monitor import OWN_MARGIN_M
    pallet = _box_piece(9.4, 10.6, 7.4, 8.2)
    tracks = Tracks()
    for k in range(30):
        x = 9.0 + 0.08 * k            # walks along the pallet, 0.4 m beside it, then on
        person = _box_piece(x - 0.1, x + 0.1, 6.85, 7.05)
        assert 7.4 - 7.05 > OWN_MARGIN_M
        confirmed = tracks.update(k * 0.2, [pallet, person])
    assert len(confirmed) == 2
    walker = max(confirmed, key=lambda t: abs(t['vx']))
    assert walker['moving'] and not min(confirmed, key=lambda t: abs(t['vx']))['moving']


def test_too_close_to_bend_out_backs_up_just_enough_then_goes_around():
    from tugger_control.obstacle_policy import BACKUP_MAX_M, plan_backup, plan_detour
    route = [(i * 0.05, 0.0) for i in range(400)]          # 20 m along x, driven from x = 0
    pallet = {'x': 9.9, 'y': 0.74, 'r': 0.67}
    robot = (6.83, 0.0)
    assert plan_detour(route, robot, pallet, ROBOT_HALF_WIDTH_M, lambda p: True) is None
    back, bent = plan_backup(route, robot, pallet, ROBOT_HALF_WIDTH_M, lambda p: True)
    length = sum(math.dist(a, b) for a, b in zip(back, back[1:]))
    assert 0 < length <= BACKUP_MAX_M
    assert all(abs(y) < 1e-9 for _, y in back) and back[-1][0] < robot[0]     # straight back
    assert math.dist(bent[0], back[-1]) < 0.06


def test_no_backup_when_even_two_metres_back_would_not_make_room():
    from tugger_control.obstacle_policy import plan_backup
    route = [(i * 0.05, 0.0) for i in range(400)]
    pallet = {'x': 9.9, 'y': 0.74, 'r': 0.67}
    assert plan_backup(route, (6.83, 0.0), pallet, ROBOT_HALF_WIDTH_M, lambda p: False) is None
    assert plan_backup(route, (1.0, 0.0), {'x': 3.5, 'y': 0.5, 'r': 0.67},
                       ROBOT_HALF_WIDTH_M, lambda p: True) is None    # only 1 m was driven


def test_a_curved_driven_stretch_is_backed_along_the_curve():
    from tugger_control.obstacle_policy import backup_line
    arc = [(math.sin(a), 1 - math.cos(a)) for a in [k * 0.05 for k in range(30)]]
    assert backup_line(arc) == arc
    line = [(5.0 - 0.1 * k, 7.0) for k in range(11)]
    straight = backup_line(line)
    assert straight[0] == (5.0, 7.0) and math.isclose(straight[-1][0], 4.0)


def test_a_long_pallet_along_the_aisle_is_passed_by_its_real_width_not_its_circle():
    from tugger_control.obstacle_policy import object_extent
    pallet = {'x': 9.9, 'y': 0.73, 'r': 0.70,
              'outline': [(9.3, 0.33), (10.5, 0.33), (10.5, 1.13), (9.3, 1.13)], 'outline_margin': 0.03}
    left, right, half_length = object_extent(pallet, (9.9, 0.0), 0.0)
    assert math.isclose(right, 0.30, abs_tol=1e-9) and math.isclose(left, 1.16, abs_tol=1e-9)
    assert math.isclose(half_length, 0.63, abs_tol=1e-9)
    # Racks 1.55 m either side of the route (the y = 7 aisle), 0.3 m kept to them.
    def clear(points):
        return all(1.55 - abs(y) - ROBOT_HALF_WIDTH_M >= 0.30 for _, y in points)
    route = [(i * 0.05, 0.0) for i in range(400)]
    assert plan_detour(route, (4.0, 0.0), pallet, ROBOT_HALF_WIDTH_M, clear) is not None
    circle = {k: v for k, v in pallet.items() if k not in ('outline', 'outline_margin')}
    assert plan_detour(route, (4.0, 0.0), circle, ROBOT_HALF_WIDTH_M, clear) is None


def test_an_object_without_an_outline_keeps_its_circle():
    from tugger_control.obstacle_policy import object_extent
    assert object_extent({'x': 1.0, 'y': 0.5, 'r': 0.3}, (1.0, 0.0), 0.0) == (0.8, 0.2, 0.3)


def test_the_back_up_goes_a_margin_further_so_stopping_15_cm_short_still_leaves_room():
    from tugger_control.obstacle_policy import BACKUP_MARGIN_M, plan_backup, plan_detour
    route = [(i * 0.05, 0.0) for i in range(400)]
    pallet = {'x': 9.9, 'y': 0.74, 'r': 0.67}
    robot = (6.83, 0.0)
    back, _ = plan_backup(route, robot, pallet, ROBOT_HALF_WIDTH_M, lambda p: True)
    planned = robot[0] - back[-1][0]
    stopped_short = (back[-1][0] + 0.15, 0.0)            # the goal checker's 15 cm
    assert plan_detour(route, stopped_short, pallet, ROBOT_HALF_WIDTH_M, lambda p: True) is not None
    no_margin, _ = plan_backup(route, robot, pallet, ROBOT_HALF_WIDTH_M, lambda p: True, margin=0.0)
    assert math.isclose(planned - (robot[0] - no_margin[-1][0]), BACKUP_MARGIN_M, abs_tol=1e-6)


def test_an_object_being_recognised_is_not_judged_yet():
    from tugger_control.obstacle_policy import awaiting_look
    assert awaiting_look(dict(obj(5, 0), looks=4, pending=True))
    assert not awaiting_look(dict(obj(5, 0), looks=4))
    assert not awaiting_look(dict(obj(5, 0), looks=4, pending=True, cls='pallet'))


def test_an_unnamed_object_keeps_the_full_gap_while_recognition_runs_as_it_could_be_a_person():
    from tugger_control.obstacle_policy import route_objects
    half = ROBOT_HALF_WIDTH_M
    beside = 0.6 + half             # 0.27 m from the robot's side: a slow pass for a box
    box_no_ai = obj(5, beside, r=0.1)
    in_way, slow = route_objects([box_no_ai], LINE, half)
    assert not in_way and slow == [box_no_ai]           # no recognition: as before
    unnamed = dict(obj(5, beside, r=0.1), looks=3)
    in_way, slow = route_objects([unnamed], LINE, half)
    assert in_way == [unnamed] and not slow             # recognition running: full gap
    pallet = dict(obj(5, beside, r=0.1), looks=3, cls='pallet')
    in_way, slow = route_objects([pallet], LINE, half)
    assert not in_way and slow == [pallet]              # named as a pallet: passed slowly
