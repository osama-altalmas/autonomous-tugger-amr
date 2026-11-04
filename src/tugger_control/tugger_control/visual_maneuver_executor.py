"""Motion-free primitive execution state machine; not connected to ROS.

The caller must enforce marker identity/freshness, collision clearance,
visibility, command ownership and a watchdog independently. No ROS commands
are published here.
"""

import math

from tugger_control.visual_maneuver_planner import propagate


def _local_pose(origin, pose):
    x, y, yaw = origin
    dx, dy = pose[0] - x, pose[1] - y
    return (math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
            math.atan2(math.sin(pose[2] - yaw), math.cos(pose[2] - yaw)))


class VisualManeuverExecutor:
    """Track bounded signed arcs from measured odometry with stop interlocks."""

    def __init__(self, primitives, *, speed=0.01, observation_age=0.5,
                 segment_timeout=12.0, position_error=0.012, heading_error=0.10):
        self.primitives = tuple((float(d), float(k)) for d, k in primitives)
        if (not self.primitives or not 0 < speed <= 0.02 or
                not 0 < observation_age <= 0.5 or segment_timeout <= 0 or
                position_error <= 0 or heading_error <= 0):
            raise ValueError('invalid visual maneuver limits')
        if any(not math.isfinite(d) or not math.isfinite(k) or
               not 0.005 <= abs(d) <= 0.05 or abs(k) > 2.0 / 0.58
               for d, k in self.primitives):
            raise ValueError('primitive outside 4WS limits')
        self.speed = speed
        self.observation_age = observation_age
        self.segment_timeout = segment_timeout
        self.position_error = position_error
        self.heading_error = heading_error
        self.index = 0
        self.origin = None
        self.started = None
        self.stopping = False
        self.stopped = False

    def update(self, pose, speed, now, odom_stamp, camera_stamp, *, identity_ok,
               clearance_ok, visible, cancelled=False):
        """Return a request dictionary; fail closed on unavailable feedback."""
        stop = {'state': 'STOP', 'linear_x': 0.0, 'angular_z': 0.0}
        if self.stopped:
            return {**stop, 'reason': 'executor stopped'}
        try:
            numbers = (*pose, speed, now, odom_stamp, camera_stamp)
            valid = len(pose) == 3 and all(math.isfinite(v) for v in numbers)
        except (TypeError, ValueError):
            valid = False
        if (not valid or cancelled or not identity_ok or not clearance_ok or not visible or
                not 0 <= now - odom_stamp <= self.observation_age or
                not 0 <= now - camera_stamp <= self.observation_age):
            self.stopped = True
            return {**stop, 'reason': 'cancelled or unsafe/stale feedback'}
        if self.started is not None and (now - self.started > self.segment_timeout or now < self.started):
            self.stopped = True
            return {**stop, 'reason': 'primitive timed out'}
        if self.stopping:
            if abs(speed) > 0.003:
                return {**stop, 'state': 'WAIT_STOP', 'reason': 'waiting for stationary feedback'}
            self.index += 1
            self.origin = None
            self.started = None
            self.stopping = False
            if self.index == len(self.primitives):
                self.stopped = True
                return {**stop, 'state': 'DONE', 'reason': 'maneuver ended; verify camera target'}
        if self.origin is None:
            if abs(speed) > 0.003:
                return {**stop, 'state': 'WAIT_STOP', 'reason': 'waiting before maneuver'}
            self.origin = tuple(pose)
            self.started = now
        distance, curvature = self.primitives[self.index]
        local = _local_pose(self.origin, pose)
        # Short primitives have less than pi/2 rotation, so the longitudinal
        # arc coordinate remains monotonic. Yaw itself lags while steering
        # slews and therefore must not be used as the distance travelled.
        progress = (math.asin(max(-1.0, min(1.0, curvature * local[0]))) / curvature
                    if abs(curvature) > 1e-8 else local[0])
        expected = propagate((0., 0., 0.), progress, curvature)
        yaw_error = abs(math.atan2(math.sin(local[2] - expected[2]),
                                   math.cos(local[2] - expected[2])))
        if (progress * distance < -0.002 * abs(distance) or
                math.hypot(local[0] - expected[0], local[1] - expected[1]) > self.position_error or
                yaw_error > self.heading_error or abs(progress) > abs(distance) + 0.01):
            self.stopped = True
            return {**stop, 'reason': 'odometry diverged from planned primitive'}
        if abs(progress) >= abs(distance) - 0.002:
            self.stopping = True
            return {**stop, 'state': 'WAIT_STOP', 'reason': 'primitive complete'}
        velocity = math.copysign(self.speed, distance)
        return {'state': 'MOVING', 'linear_x': velocity,
                'angular_z': velocity * curvature, 'reason': 'guarded primitive request'}
