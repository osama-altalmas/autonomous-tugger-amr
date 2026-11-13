"""Phase B estimators: marker localization, cargo estimate and scene."""

import json
import math

from tugger_control.cargo_estimate import CargoEstimate, scene_payload
from tugger_control.lift_geometry import LIFT_DOWN_M, LIFT_UP_M
from tugger_control.marker_localization import correction

MARKERS = {'qr_station_1': (-17.5, -7.0), 'qr_home': (0.0, 0.0)}


def test_marker_moves_the_estimate_onto_the_known_marker():
    # Estimate 4 cm east of the truth; the frame is seen 2 cm ahead.
    truth = (-17.52, -7.0, 0.0)
    estimate = (truth[0] + 0.04, truth[1], 0.0)
    name, corrected, size = correction(estimate, (0.02, 0.0), MARKERS)
    assert name == 'qr_station_1'
    assert math.isclose(corrected[0], truth[0], abs_tol=1e-9)
    assert math.isclose(size, 0.04, abs_tol=1e-9)


def test_heading_is_used_to_place_the_marker():
    # Facing north, a frame 3 cm ahead is 3 cm north of the robot.
    name, corrected, size = correction((0.0, -0.03, math.pi / 2), (0.03, 0.0), MARKERS)
    assert name == 'qr_home' and size < 1e-9


def test_a_red_object_far_from_any_marker_is_ignored():
    assert correction((5.0, 5.0, 0.0), (0.05, 0.0), MARKERS) is None


def test_cargo_is_attached_only_after_its_tag_is_read_and_the_lift_is_up():
    estimate = CargoEstimate({0: (-17.5, -7.0, 0.0, 0.0), 3: (17.5, 7.0, 0.0, 0.0)})
    robot = (-17.5, -7.0, 0.0)
    estimate.request('attach', 0)
    estimate.update(robot, LIFT_UP_M, 10.0)          # no tag read: not attached
    assert estimate.state(0) == 'detached'
    estimate.request('attach', 0)
    estimate.qr_read(0, 10.5)
    estimate.update(robot, LIFT_DOWN_M, 11.0)        # tag read, lift still down
    assert estimate.state(0) == 'detached'
    estimate.update(robot, LIFT_UP_M, 12.0)
    assert estimate.state(0) == 'attached'
    # It moves with the robot, raised by the lift.
    estimate.update((-15.0, -7.0, 0.0), LIFT_UP_M, 13.0)
    x, y, z, _ = estimate.poses[0]
    assert math.isclose(x, -15.0) and z > 0.0


def test_a_wrong_tag_does_not_attach():
    estimate = CargoEstimate({0: (-17.5, -7.0, 0.0, 0.0), 3: (17.5, 7.0, 0.0, 0.0)})
    estimate.request('attach', 0)
    estimate.qr_read(3, 10.0)
    estimate.update((-17.5, -7.0, 0.0), LIFT_UP_M, 10.5)
    assert estimate.state(0) == 'detached'


def test_detach_leaves_the_cargo_where_the_robot_set_it_down():
    estimate = CargoEstimate({0: (-17.5, -7.0, 0.0, 0.0), 3: (17.5, 7.0, 0.0, 0.0)})
    estimate.request('attach', 0)
    estimate.qr_read(0, 1.0)
    estimate.update((-17.5, -7.0, 0.0), LIFT_UP_M, 1.5)
    estimate.update((-17.5, 7.0, math.pi / 2), LIFT_UP_M, 30.0)
    estimate.request('detach', 0)
    estimate.update((-17.5, 7.0, math.pi / 2), LIFT_DOWN_M, 40.0)
    assert estimate.state(0) == 'detached'
    x, y, z, _ = estimate.poses[0]
    assert math.isclose(x, -17.5) and math.isclose(y, 7.0) and z == 0.0


def test_scene_has_static_boxes_and_cargo_at_its_estimated_pose():
    from tugger_control.clearance_2_5d import Box
    static = [Box('rack/c', 1.0, 2.0, 0.0, 0.5, 0.4, 0.0, 2.0)]
    shapes = {0: [('tabletop_collision', 0.295, [1.3, 1.3, 0.03])]}
    data = json.loads(scene_payload(static, shapes, {0: (3.0, 4.0, 0.0, 0.0)}, 12.0))
    assert data['valid'] and data['stamp'] == 12.0
    cargo = [b for b in data['boxes'] if b['model'] == 'cargo_station_0_stand'][0]
    assert (cargo['x'], cargo['y'], cargo['z'], cargo['hx']) == (3.0, 4.0, 0.295, 0.65)


def _leg_scan(robot, legs, half=0.04, step=0.01):
    """Scan points on the faces of square legs visible from the robot."""
    import numpy as np
    x, y, yaw = robot
    c, s = math.cos(yaw), math.sin(yaw)
    points = []
    for lx, ly, _ in legs:
        for t in np.arange(-half, half + 1e-9, step):
            for fx, fy in ((lx - half, ly + t), (lx + half, ly + t), (lx + t, ly - half),
                           (lx + t, ly + half)):
                # keep faces facing the robot
                if math.hypot(fx - x, fy - y) <= math.hypot(lx - x, ly - y):
                    dx, dy = fx - x, fy - y
                    points.append((c * dx + s * dy, -s * dx + c * dy))
    return points


def test_station_legs_give_the_robot_pose():
    from tugger_control.marker_localization import leg_fix
    legs = [(-18.06, -7.56, 0.04), (-18.06, -6.44, 0.04), (-16.94, -7.56, 0.04),
            (-16.94, -6.44, 0.04)]
    truth = (-15.5, -7.0, math.pi)
    estimate = (-15.45, -7.04, math.pi - math.radians(2))   # 6 cm and 2 degrees off
    (x, y, yaw), residual, used = leg_fix(estimate, _leg_scan(truth, legs), legs)
    assert used >= 3
    assert math.hypot(x - truth[0], y - truth[1]) < 0.02
    assert abs(math.atan2(math.sin(yaw - truth[2]), math.cos(yaw - truth[2]))) < math.radians(1)


def test_too_few_legs_give_no_fix():
    from tugger_control.marker_localization import leg_fix
    legs = [(-18.06, -7.56, 0.04), (-16.94, -7.56, 0.04)]
    assert leg_fix((-15.5, -7.0, math.pi), _leg_scan((-15.5, -7.0, math.pi), legs), legs) is None


def test_laser_updates_stay_on_when_a_route_only_passes_a_station():
    from tugger_control.marker_localization import suspend_laser_updates
    assert suspend_laser_updates(True, False)        # working at the station
    assert not suspend_laser_updates(True, True)     # a route passing through
    assert not suspend_laser_updates(False, False)   # open floor


def test_zone_actions():
    from tugger_control.marker_localization import zone_action
    assert zone_action(None, False) == 'resume'      # start-up on open floor
    assert zone_action(None, True) == 'suspend'
    assert zone_action(False, True) == 'suspend'     # arriving at a station
    assert zone_action(True, False) == 'reseed'      # leaving a station
    assert zone_action(True, True) is None
    assert zone_action(False, False) is None


def test_leaving_a_station_reseeds_before_updates_resume():
    """Issue 9: AMCL must not apply the whole departure as one motion step."""
    import math
    from types import SimpleNamespace
    from tugger_control import marker_localization as ml

    calls = []
    pose = {'x': -17.5, 'y': 6.9}

    def lookup(*_):
        return SimpleNamespace(header=SimpleNamespace(stamp='stamp'), transform=SimpleNamespace(
            translation=SimpleNamespace(x=pose['x'], y=pose['y']),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(math.pi / 4), w=math.cos(math.pi / 4))))

    node = SimpleNamespace(
        _tf=SimpleNamespace(lookup_transform=lookup), _centres=[(-17.5, 7.0)],
        _route_transit=False, _station_mode=None, _resume_pending=False,
        _amcl_params=SimpleNamespace(service_is_ready=lambda: True),
        get_logger=lambda: SimpleNamespace(info=lambda *_: None))
    node._send = lambda stamp, x, y, yaw, p_std, y_std: calls.append(('reseed', x, y, yaw, p_std))
    node._set_update_thresholds = lambda station: calls.append(('thresholds', station))
    tick = lambda: ml.MarkerLocalization._zone_tick(node)   # noqa: E731

    tick()                                     # docked: suspend
    assert calls == [('thresholds', True)]
    calls.clear()
    pose['y'] = 5.1
    node._route_transit = True                 # route driving begins
    tick()
    assert [c[0] for c in calls] == ['reseed']           # re-seed only, still suspended
    _, x, y, yaw, p_std = calls[0]
    assert (x, y) == (-17.5, 5.1) and abs(yaw - math.pi / 2) < 1e-9
    assert p_std == ml.POSITION_STD_M
    calls.clear()
    tick()                                     # next tick: normal thresholds
    assert calls == [('thresholds', False)]
    calls.clear()
    tick()
    assert calls == []
