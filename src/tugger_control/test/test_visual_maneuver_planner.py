"""Motion-free tests of the experimental 4WS primitive search."""

import math

import pytest

from tugger_control.visual_maneuver_planner import (
    marker_visible, plan_visual_maneuver, propagate, replan_visual_maneuver,
)


@pytest.mark.parametrize('distance,curvature', [(.05, 3.), (-.05, 3.),
                                                 (.05, -3.), (-.05, -3.),
                                                 (.05, 0.)])
def test_4ws_primitive_reverses_exactly(distance, curvature):
    start = (.1, -.2, .3)
    end = propagate(propagate(start, distance, curvature), -distance, curvature)
    assert end == pytest.approx(start, abs=1e-12)


@pytest.mark.parametrize('target', [(-.043, .025, 0.), (-.043, -.025, 0.),
                                    (0., .03, 0.), (0., -.03, 0.)])
def test_primitive_search_finds_visible_maneuver_in_empty_scene(target):
    plan = plan_visual_maneuver(target, lambda pose: True, max_depth=12, beam_width=2000)
    assert plan is not None
    assert len({math.copysign(1, distance) for distance, _ in plan}) == 2
    pose = (0., 0., 0.)
    for distance, curvature in plan:
        assert abs(curvature) <= 2 / .58
        for step in range(1, 11):
            assert marker_visible(propagate(pose, distance * step / 10, curvature), target)
        pose = propagate(pose, distance, curvature)
    assert math.hypot(pose[0] - target[0], pose[1] - target[1]) <= .018
    assert abs(pose[2] - target[2]) <= .1


def test_clearance_callback_rejects_blocked_start():
    assert plan_visual_maneuver((0., .03, 0.), lambda pose: False) is None


def test_clearance_callback_rejects_every_path_out_of_the_start():
    assert plan_visual_maneuver((0., .03, 0.),
                                lambda pose: math.hypot(pose[0], pose[1]) < .001) is None


def test_clearance_callback_is_required():
    with pytest.raises(ValueError, match='clearance'):
        plan_visual_maneuver((0., .03, 0.), None)


def test_replanning_checks_clearance_in_world_coordinates():
    seen = []
    pose = (10., -4., math.pi / 2)
    target = (9.975, -4.043, math.pi / 2)

    def clearance(candidate):
        seen.append(candidate)
        return False

    assert replan_visual_maneuver(pose, target, clearance) is None
    assert seen[0] == pytest.approx(pose)


def test_replanning_rejects_unknown_or_nonfinite_pose():
    with pytest.raises(ValueError):
        replan_visual_maneuver((float('nan'), 0., 0.), (0., 0., 0.), lambda _: True)


@pytest.mark.parametrize('x', [-.06, .06])
def test_margin_rejects_marker_at_edge_of_modeled_camera_view(x):
    assert not marker_visible((0., 0., 0.), (x, .03, 0.))
    assert plan_visual_maneuver((x, .03, 0.), lambda pose: True) is None


def test_visual_clearance_includes_the_carried_cargo():
    from tugger_control.clearance_2_5d import Box
    from tugger_control.lift_geometry import LIFT_UP_M
    from tugger_control.visual_scene_clearance import PreparedVisualClearance
    # A beam at cargo height beside the robot: only the carried cargo reaches it.
    beam = Box('beam', 0.0, 0.62, 0.0, 0.05, 0.05, 0.33, 0.40)
    empty = PreparedVisualClearance([beam], LIFT_UP_M)
    loaded = PreparedVisualClearance([beam], LIFT_UP_M, carried=(0.0, 0.0, 0.065, 0.0))
    assert empty((0.0, 0.0, 0.0))
    assert not loaded((0.0, 0.0, 0.0))
