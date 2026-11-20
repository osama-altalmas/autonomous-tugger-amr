"""Back up, then go around: the panel side (campaign 4 crash, 2026-11-20)."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

from tugger_control.direction_policy import prepare_dynamic_path
from tugger_control.obstacle_policy import ROBOT_HALF_WIDTH_M
from tugger_control.operator_panel import OperatorBridge


def test_a_detour_the_robot_could_not_start_along_is_not_clear_and_does_not_raise():
    bridge = SimpleNamespace(_obstacles=[])
    points = [(i * 0.1, 0.0) for i in range(30)]
    target = {'x': 1.5, 'y': 0.5, 'r': 0.3}
    assert OperatorBridge._detour_clear(bridge, points, math.radians(50), ROBOT_HALF_WIDTH_M,
                                        target) is False


def _bridge(pose):
    hold_path = SimpleNamespace(poses=[])
    bridge = SimpleNamespace(
        _obstacles=[], _backups_done=[], _obstacle_hold={'phase': 'route', 'path': hold_path},
        _obstacle_supervisor=Mock(), _report=Mock(), _follow=Mock(), _publish_speed_limit=Mock(),
        get_clock=Mock(return_value=SimpleNamespace(now=Mock(return_value=SimpleNamespace(
            to_msg=Mock(return_value=None))))),
        _pose=lambda p: SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=p[0], y=p[1]))))
    checked = []

    def detour_clear(points, heading, half, target):
        prepare_dynamic_path(points, heading)       # raises if the heading is not the path's
        checked.append(heading)
        return True
    bridge._detour_clear = detour_clear
    for name in ('_backed_up_for', '_points_path'):
        setattr(bridge, name, getattr(OperatorBridge, name).__get__(bridge))
    return bridge, checked


def test_backing_up_round_a_corner_checks_each_detour_with_the_heading_it_starts_with():
    # Driven north up x = 4.5, round a 1 m corner, then east along y = 7.
    route = [(4.5, 3.0 + 0.05 * i) for i in range(61)]
    route += [(5.5 - math.cos(a), 6.0 + math.sin(a)) for a in [k * math.pi / 40 for k in range(1, 21)]]
    route += [(5.5 + 0.05 * i, 7.0) for i in range(1, 300)]
    pose = (6.0, 7.0, 0.0)
    pallet = {'id': 1, 'x': 8.6, 'y': 7.74, 'r': 0.67}
    bridge, checked = _bridge(pose)
    started = OperatorBridge._try_backup(bridge, pose, route, pallet, ROBOT_HALF_WIDTH_M)
    assert checked                                # candidates were checked, none raised
    if started:
        bridge._follow.assert_called_once()
        assert bridge._backups_done == [(8.6, 7.74)]
