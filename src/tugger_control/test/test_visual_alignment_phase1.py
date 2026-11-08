"""ROS-free, deterministic regression probes for Problem 1 improvements."""

import math
import time
from pathlib import Path

import pytest

from tugger_control.clearance_2_5d import Box, collision_pairs
from tugger_control.visual_heading_gate import (angle_delta,
                                                relative_qr_heading_consistent)
from tugger_control.crab_motion import CRAB_MAX_ANGLE_RAD
from tugger_control.visual_maneuver_planner import (advance, plan_visual_maneuver,
                                                   propagate)
from tugger_control.visual_scene_clearance import PreparedVisualClearance
from tugger_control.warehouse_clearance import robot_boxes, static_world_boxes

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('forward,lateral', [
    (0.0, 0.03), (0.03, 0.03), (0.05, 0.05),
    (0.0, -0.04), (-0.043, 0.025),
])
def test_expanded_capture_has_exact_swept_clearance_and_10mm_1deg_goal(forward, lateral):
    boxes = static_world_boxes(ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf')
    checker = PreparedVisualClearance(boxes, 0.0)
    target = (forward, lateral, 0.0)
    start = time.monotonic()
    plan = plan_visual_maneuver(
        target, checker, require_full_visibility=False, max_depth=16,
        position_tolerance=.010, heading_tolerance=math.radians(1),
        time_budget=3.0)
    # Generous bound avoids noisy CI but detects the minutes-long old search.
    assert time.monotonic() - start < 5.0
    assert plan is not None
    for primitive in plan:
        if len(primitive) > 2 and primitive[2] != 0.0:
            # Crab: all wheels at one angle within the wheel limit, no arc.
            assert (0.005 <= abs(primitive[0]) <= 0.10 and primitive[1] == 0.0 and
                    abs(primitive[2]) <= CRAB_MAX_ANGLE_RAD)
        else:
            assert 0.005 <= abs(primitive[0]) <= 0.05 and abs(primitive[1]) <= 2/.58
    pose = (0.0, 0.0, 0.0)
    for primitive in plan:
        distance = primitive[0]
        count = math.ceil(abs(distance)/.005)
        for index in range(1, count + 1):
            assert checker(advance(pose, primitive, distance*index/count))
        pose = advance(pose, primitive)
    assert math.hypot(pose[0] - forward, pose[1] - lateral) <= .010
    assert abs(angle_delta(pose[2], 0.0)) <= math.radians(1)


def test_expanded_planner_cannot_ignore_collisions_even_with_macro_search():
    # Block *every* departure from the start while keeping the start itself free.
    checker = lambda p: math.hypot(p[0], p[1]) < .002
    assert plan_visual_maneuver((0., .03, 0.), checker,
                                require_full_visibility=False, max_depth=16) is None


def test_prepared_clearance_matches_original_exact_sat_across_real_scene():
    boxes = static_world_boxes(ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf')
    checker = PreparedVisualClearance(boxes, 0.0)
    locations = [(0.0, 0.0), (17.5, 7.0), (-17.5, -7.0),
                 (20., 0.), (15.5, 7.), (-17.5, 6.)]
    for x, y in locations:
        for dx in (-.2, 0, .2):
            for dy in (-.2, 0, .2):
                for yaw in (0., .35, 1.57, 3.14):
                    pose = (x+dx, y+dy, yaw)
                    assert checker(pose) == (not collision_pairs(
                        robot_boxes(pose, 0.0), boxes, margin=.01))


def test_prepared_clearance_still_rejects_obstacle_just_at_margin_boundary():
    obstacle = Box('test', .5, 0., 0., .05, .2, .0725, .2)
    checker = PreparedVisualClearance([obstacle], 0.0, margin=.01)
    assert not checker((0., 0., 0.))


def test_relative_qr_heading_crosscheck_detects_sudden_vision_disagreement():
    assert relative_qr_heading_consistent(.4, .3, .2, .1)
    assert relative_qr_heading_consistent(.2, .3, .2, .1)  # opposite optical handedness
    assert not relative_qr_heading_consistent(.8, .3, .2, .1)
    assert abs(angle_delta(-math.pi+.02, math.pi-.02)) == pytest.approx(.04)

@pytest.mark.parametrize('target', [(0.0, .03, 0.0), (.03, .03, 0.0)])
def test_stopped_replans_reduce_steering_slew_residual_in_ideal_4ws_model(target):
    """Kinematic/steering test, NOT a camera/Gazebo or safety certification."""
    from tugger_control.motion_math import approach, steering_for_twist
    from tugger_control.visual_maneuver_executor import VisualManeuverExecutor

    pose = (0., 0., 0.)
    speed = steering = 0.
    executor = None
    replans = 0
    dt = .05
    for step in range(4500):
        if (math.hypot(pose[0] - target[0], pose[1] - target[1]) <= .01 and
                abs(angle_delta(pose[2], target[2])) <= math.radians(1)):
            assert replans >= 1
            return
        if executor is None:
            c, s = math.cos(pose[2]), math.sin(pose[2])
            dx, dy = target[0] - pose[0], target[1] - pose[1]
            plan = plan_visual_maneuver(
                (c*dx + s*dy, -s*dx + c*dy,
                 angle_delta(target[2], pose[2])),
                lambda _: True, require_full_visibility=False,
                max_depth=16, position_tolerance=.01,
                heading_tolerance=math.radians(1))
            assert plan is not None
            replans += 1
            executor = VisualManeuverExecutor(plan)
        now = step * dt + 1
        command = executor.update(pose, speed, now, now, now,
                                  identity_ok=True, clearance_ok=True, visible=True)
        assert command['state'] != 'STOP', command.get('reason')
        if command['state'] == 'DONE':
            executor = None
        if command.get('frame') == 'crab':
            # Crab: the controller stops and re-steers first (modelled as
            # instantaneous once stopped), then rolls along the wheel angle.
            desired = math.copysign(math.hypot(command['linear_x'], command['linear_y']),
                                    command['linear_x'])
            beta = math.atan2(math.copysign(1, desired) * command['linear_y'],
                              math.copysign(1, desired) * command['linear_x'])
            speed = approach(speed, desired, .25 if speed*desired >= 0 else .35, dt)
            pose = (pose[0] + speed*dt*math.cos(pose[2] + beta),
                    pose[1] + speed*dt*math.sin(pose[2] + beta), pose[2])
            continue
        desired = command['linear_x']
        # The 4WS controller holds the wheel angle when commanded to stop.
        desired_steering = (steering if abs(desired) < 1e-4 else
                            steering_for_twist(desired, command['angular_z'], .58, math.pi / 4))
        speed = approach(speed, desired, .25 if speed*desired >= 0 else .35, dt)
        steering = approach(steering, desired_steering, .8, dt)
        pose = propagate(pose, speed*dt, 2*math.tan(steering)/.58)
    pytest.fail('offline steering-slew model did not converge within 225 seconds')
