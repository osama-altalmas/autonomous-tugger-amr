"""Motion-free tests of executor interlocks and measured progress."""

import pytest
import math

from tugger_control.visual_maneuver_executor import VisualManeuverExecutor
from tugger_control.visual_maneuver_planner import (propagate, plan_visual_maneuver,
                                                   replan_visual_maneuver, marker_visible)
from tugger_control.motion_math import approach, steering_for_twist


def update(executor, pose=(0., 0., 0.), speed=0., now=1., **kwargs):
    args = dict(identity_ok=True, clearance_ok=True, visible=True)
    args.update(kwargs)
    return executor.update(pose, speed, now, now, now, **args)


def test_forward_reverse_requires_stationary_feedback_between_arcs():
    executor = VisualManeuverExecutor(((.025, 3.), (-.025, -3.)))
    assert update(executor, now=.6)['state'] == 'WAIT_STOP'
    assert update(executor)['linear_x'] > 0
    end = propagate((0., 0., 0.), .025, 3.)
    assert update(executor, end, .01, 2.)['state'] == 'WAIT_STOP'
    assert update(executor, end, .01, 2.1)['state'] == 'WAIT_STOP'
    # Stopped, but not yet for 0.3 s: the next primitive still waits.
    assert update(executor, end, 0., 2.2)['state'] == 'WAIT_STOP'
    assert update(executor, end, 0., 2.6)['linear_x'] < 0
    final = propagate(end, -.025, -3.)
    assert update(executor, final, -.01, 3.)['state'] == 'WAIT_STOP'
    assert update(executor, final, 0., 3.1)['state'] == 'WAIT_STOP'
    assert update(executor, final, 0., 3.5)['state'] == 'DONE'


@pytest.mark.parametrize('field', ['identity_ok', 'clearance_ok', 'visible'])
def test_missing_safety_conditions_stop_permanently(field):
    executor = VisualManeuverExecutor(((.025, 0.),))
    assert update(executor, **{field: False})['state'] == 'STOP'
    assert update(executor)['state'] == 'STOP'


def test_stale_camera_and_odometry_stop():
    for odom, camera in ((0., 1.), (1., 0.)):
        executor = VisualManeuverExecutor(((.025, 0.),))
        result = executor.update((0., 0., 0.), 0., 1., odom, camera,
                                 identity_ok=True, clearance_ok=True, visible=True)
        assert result['state'] == 'STOP'


def test_cancellation_and_divergence_stop():
    executor = VisualManeuverExecutor(((.025, 0.),))
    assert update(executor, cancelled=True)['state'] == 'STOP'
    executor = VisualManeuverExecutor(((.025, 0.),))
    update(executor, now=.6)
    assert update(executor)['state'] == 'MOVING'
    assert update(executor, (0., .02, 0.), now=2.)['state'] == 'STOP'


def test_robot_that_does_not_turn_during_an_arc_is_stopped():
    # Regression: a 6 degree correction (-0.035 m at curvature -3) driven
    # straight used to pass the fixed 0.10 rad yaw check and loop forever.
    executor = VisualManeuverExecutor(((-.035, -3.),), speed=.005)
    update(executor, now=.6)
    assert update(executor)['state'] == 'MOVING'
    stopped = None
    for step in range(1, 40):
        result = update(executor, (-.001 * step, 0., 0.), -.005, 1. + .2 * step)
        if result['state'] == 'STOP':
            stopped = step
            break
    assert stopped is not None and stopped < 35
    assert result['reason'] == 'odometry diverged from planned primitive'


def test_straight_after_a_full_lock_arc_allows_the_wheels_to_unwind():
    # Seen in Gazebo: 0.020 rad yaw on the straight after a -3 curvature arc
    # while the wheels unwound from 41 degrees; that is not a divergence.
    executor = VisualManeuverExecutor(((-.035, -3.), (.028, 0.)), speed=.005)
    assert executor.slew_lag[1] == executor.slew_lag[0] > 0.02
    executor = VisualManeuverExecutor(((.028, 0.),), speed=.005)
    assert executor.slew_lag[0] == 0.0


def test_next_primitive_waits_until_the_robot_has_really_stopped():
    # Seen in Gazebo: the robot still rolled 2 mm back after the reverse arc
    # when the next primitive started at under 3 mm/s, which read as divergence.
    executor = VisualManeuverExecutor(((-.035, -3.), (.028, 0.)), speed=.005)
    update(executor, now=.6)
    assert update(executor)['state'] == 'MOVING'
    end = propagate((0., 0., 0.), -.035, -3.)
    assert update(executor, end, -.002, 9.)['state'] == 'WAIT_STOP'
    assert update(executor, end, -.002, 9.5)['state'] == 'WAIT_STOP'
    assert update(executor, end, 0., 9.6)['state'] == 'WAIT_STOP'
    assert update(executor, end, 0., 9.95)['linear_x'] > 0


def test_primitive_timeout_and_invalid_curvature_stop():
    with pytest.raises(ValueError):
        VisualManeuverExecutor(((.025, 4.),))
    executor = VisualManeuverExecutor(((.025, 0.),), segment_timeout=2.)
    update(executor, now=.6)
    update(executor)
    assert update(executor, now=4.)['state'] == 'STOP'


@pytest.mark.parametrize('target', [(-.043, .025, 0.), (-.043, -.025, 0.),
                                    (0., .03, 0.), (0., -.03, 0.)])
def test_planner_executor_with_acceleration_and_steering_slew(target):
    # A kinematic plan is only a proposal: steering and acceleration slew can
    # leave a measurable residual.  Exercise the production closed loop by
    # replanning from the measured pose instead of weakening the 20 mm gate.
    executor = None
    pose, speed, steering, dt = (0., 0., 0.), 0., 0., .05
    for step in range(2400):
        if executor is None:
            if (math.hypot(pose[0] - target[0], pose[1] - target[1]) <= .02 and
                    abs(pose[2] - target[2]) <= .1):
                break
            plan = replan_visual_maneuver(
                pose, target, lambda _: True, max_depth=12, beam_width=2000)
            assert plan is not None
            executor = VisualManeuverExecutor(plan)
        command = update(executor, pose, speed, 1 + step * dt,
                         visible=marker_visible(pose, target))
        assert command['state'] != 'STOP', command.get('reason')
        if command['state'] == 'DONE':
            executor = None
            continue
        desired = command['linear_x']
        desired_steering = steering_for_twist(desired, command['angular_z'], .58, math.pi / 4)
        speed = approach(speed, desired, .25 if speed * desired >= 0 else .35, dt)
        steering = approach(steering, desired_steering, .8, dt)
        pose = propagate(pose, speed * dt, 2 * math.tan(steering) / .58)
    else:
        pytest.fail('maneuver did not complete')
    assert math.hypot(pose[0] - target[0], pose[1] - target[1]) <= .02
    assert abs(pose[2] - target[2]) <= .1


@pytest.mark.parametrize('lateral', [.03, -.03])
def test_feedback_replanning_corrects_remaining_sideways_error(lateral):
    target = (0., lateral, 0.)
    pose, speed, steering, dt = (0., 0., 0.), 0., 0., .05
    executor = None
    for step in range(3600):
        if executor is None:
            dx, dy = target[0] - pose[0], target[1] - pose[1]
            local = (math.cos(pose[2]) * dx + math.sin(pose[2]) * dy,
                     -math.sin(pose[2]) * dx + math.cos(pose[2]) * dy,
                     target[2] - pose[2])
            if math.hypot(dx, dy) <= .02 and abs(local[2]) <= .1:
                break
            plan = replan_visual_maneuver(pose, target, lambda _: True,
                                          max_depth=12, beam_width=2000)
            assert plan is not None, f'no refinement from {local}'
            executor = VisualManeuverExecutor(plan)
        command = update(executor, pose, speed, 1 + step * dt,
                         visible=marker_visible(pose, target))
        assert command['state'] != 'STOP', command.get('reason')
        if command['state'] == 'DONE':
            executor = None
            continue
        desired = command['linear_x']
        desired_steering = steering_for_twist(desired, command['angular_z'], .58, math.pi / 4)
        speed = approach(speed, desired, .25 if speed * desired >= 0 else .35, dt)
        steering = approach(steering, desired_steering, .8, dt)
        pose = propagate(pose, speed * dt, 2 * math.tan(steering) / .58)
    else:
        pytest.fail('feedback replanning did not converge')
    assert math.hypot(pose[0] - target[0], pose[1] - target[1]) <= .02
