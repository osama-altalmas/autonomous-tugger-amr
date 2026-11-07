"""Offline reproduction of visual-servo convergence gaps; no ROS motion.

The static marker is expressed in the instantaneous robot body frame. This
idealized model does not test camera calibration, obstacles, or dynamics.
"""

import math

import pytest

from tugger_control.motion_math import approach, steering_for_twist
from tugger_control.visual_maneuver_executor import VisualManeuverExecutor
from tugger_control.visual_maneuver_planner import (
    marker_visible, replan_visual_maneuver,
)


def _simulate(forward, lateral, seconds=120.0):
    dt = 0.05
    target = (forward, lateral, 0.0)
    pose, speed, steering = (0.0, 0.0, 0.0), 0.0, 0.0
    executor = None
    for step in range(int(seconds / dt)):
        now = 1.0 + step * dt
        if executor is None:
            if (math.hypot(pose[0] - target[0], pose[1] - target[1]) <= .02 and
                    abs(pose[2] - target[2]) <= .1):
                return 'ALIGNED', math.hypot(pose[0] - target[0], pose[1] - target[1])
            plan = replan_visual_maneuver(
                pose, target, lambda _: True, max_depth=12, beam_width=2000)
            if plan is None:
                return 'STOP', math.hypot(pose[0] - target[0], pose[1] - target[1])
            executor = VisualManeuverExecutor(plan)
        command = executor.update(
            pose, speed, now, now, now,
            identity_ok=True, clearance_ok=True,
            visible=marker_visible(pose, target))
        if command['state'] == 'STOP':
            return 'STOP', math.hypot(pose[0] - target[0], pose[1] - target[1])
        if command['state'] == 'DONE':
            executor = None
            continue
        velocity = command['linear_x']
        # The 4WS controller holds the wheel angle when commanded to stop.
        desired_steering = (steering if abs(velocity) < 1e-4 else steering_for_twist(
            velocity, command['angular_z'], .58, math.pi / 4))
        speed = approach(speed, velocity,
                         .25 if speed * velocity >= 0 else .35, dt)
        steering = approach(steering, desired_steering, .8, dt)
        omega = 2.0 * speed * math.tan(steering) / 0.58
        pose = (pose[0] + dt * speed * math.cos(pose[2]),
                pose[1] + dt * speed * math.sin(pose[2]),
                pose[2] + dt * omega)
    return 'TIMEOUT', math.hypot(pose[0] - target[0], pose[1] - target[1])


@pytest.mark.parametrize('forward,lateral', [
    (0.06, 0.03), (0.06, -0.03), (-0.06, 0.03), (-0.06, -0.03),
    (-0.043, 0.025), (0.0, 0.03),
])
def test_camera_correction_converges_in_idealized_4ws_model(forward, lateral):
    state, distance = _simulate(forward, lateral)
    target = (forward, lateral, 0.0)
    if not marker_visible((0.0, 0.0, 0.0), target):
        # The camera gate must reject a target whose complete marker is
        # outside the verified image envelope; no blind recovery is allowed.
        assert state == 'STOP'
    else:
        assert state == 'ALIGNED' and distance <= 0.02


@pytest.mark.parametrize('forward', [-0.06, 0.06])
def test_straight_camera_correction_converges_in_idealized_4ws_model(forward):
    state, distance = _simulate(forward, 0.0)
    target = (forward, 0.0, 0.0)
    if not marker_visible((0.0, 0.0, 0.0), target):
        assert state == 'STOP'
    else:
        assert state == 'ALIGNED' and distance <= 0.02
