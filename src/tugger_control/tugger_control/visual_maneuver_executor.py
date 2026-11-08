"""Motion-free primitive execution state machine; not connected to ROS.

The caller must enforce marker identity/freshness, collision clearance,
visibility, command ownership and a watchdog independently. No ROS commands
are published here.
"""

import math

from tugger_control.crab_motion import CRAB_FRAME, CRAB_MAX_ANGLE_RAD, crab_propagate
from tugger_control.visual_maneuver_planner import propagate

# Crab primitives (distance, 0, wheel angle) may be longer than arcs: one
# diagonal move replaces a parallel-parking sequence.
CRAB_MAX_DISTANCE = 0.10


def _local_pose(origin, pose):
    x, y, yaw = origin
    dx, dy = pose[0] - x, pose[1] - y
    return (math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
            math.atan2(math.sin(pose[2] - yaw), math.cos(pose[2] - yaw)))


STILL_SPEED = 0.001
STILL_TIME = 0.3


class VisualManeuverExecutor:
    """Track bounded signed arcs and crab moves from measured odometry.

    A primitive is (distance, curvature) for a car-like arc or
    (distance, 0, wheel angle) for a crab move.
    """

    def __init__(self, primitives, *, speed=0.01, crab_speed=None, observation_age=0.5,
                 segment_timeout=12.0, position_error=0.012, heading_error=0.10,
                 steering_rate=0.8, wheelbase=0.58):
        self.primitives = tuple(
            (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
            for p in primitives)
        crab_speed = speed if crab_speed is None else crab_speed
        if (not self.primitives or not 0 < speed <= 0.02 or
                not 0 < crab_speed <= 0.02 or
                not 0 < observation_age <= 0.5 or segment_timeout <= 0 or
                position_error <= 0 or heading_error <= 0 or
                steering_rate <= 0 or wheelbase <= 0):
            raise ValueError('invalid visual maneuver limits')
        for d, k, beta in self.primitives:
            if not all(math.isfinite(v) for v in (d, k, beta)):
                raise ValueError('primitive outside 4WS limits')
            if beta != 0.0:
                if (k != 0.0 or abs(beta) > CRAB_MAX_ANGLE_RAD or
                        not 0.005 <= abs(d) <= CRAB_MAX_DISTANCE):
                    raise ValueError('crab primitive outside 4WS limits')
            elif not 0.005 <= abs(d) <= 0.05 or abs(k) > 2.0 / 0.58:
                raise ValueError('primitive outside 4WS limits')
        self.speed = speed
        self.crab_speed = crab_speed
        self.observation_age = observation_age
        self.segment_timeout = segment_timeout
        self.position_error = position_error
        self.heading_error = heading_error
        # Yaw the robot may gain or lose against the ideal arc while its wheels
        # slew: worst case a full swing between locks at the 4WS controller's
        # steering rate. A primitive starts with the wheels still at the
        # previous primitive's angle, so that angle counts too (a straight
        # segment after a full-lock arc keeps turning until they unwind).
        def lag(k):
            return abs(k) * speed * 2.0 * abs(math.atan(wheelbase * k / 2.0)) / steering_rate
        self.slew_lag = [
            max(lag(k), lag(self.primitives[i - 1][1]) if i else 0.0)
            for i, (_, k, _) in enumerate(self.primitives)]
        self.index = 0
        self.origin = None
        self.started = None
        self.stopping = False
        self.stopped = False
        self.still_since = None

    def update(self, pose, speed, now, odom_stamp, camera_stamp, *, identity_ok,
               clearance_ok, visible, cancelled=False):
        """Return a request dictionary; fail closed on unavailable feedback."""
        stop = {'state': 'STOP', 'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0,
                'frame': 'base_footprint'}
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
        # A primitive's start is measured only once the robot has really
        # stopped: below 1 mm/s for 0.3 s. Starting while it still rolled from
        # the previous arc counted that roll against the new primitive.
        if abs(speed) > STILL_SPEED:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = now
        settled = self.still_since is not None and now - self.still_since >= STILL_TIME
        if self.stopping:
            if not settled:
                return {**stop, 'state': 'WAIT_STOP', 'reason': 'waiting for stationary feedback'}
            self.index += 1
            self.origin = None
            self.started = None
            self.stopping = False
            if self.index == len(self.primitives):
                self.stopped = True
                return {**stop, 'state': 'DONE', 'reason': 'maneuver ended; verify camera target'}
        if self.origin is None:
            if not settled:
                return {**stop, 'state': 'WAIT_STOP', 'reason': 'waiting before maneuver'}
            self.origin = tuple(pose)
            self.started = now
        distance, curvature, beta = self.primitives[self.index]
        local = _local_pose(self.origin, pose)
        if beta != 0.0:
            # Crab: distance along the wheel direction; no turning expected.
            progress = local[0] * math.cos(beta) + local[1] * math.sin(beta)
            expected = crab_propagate((0., 0., 0.), progress, beta)
        else:
            # Short primitives have less than pi/2 rotation, so the
            # longitudinal arc coordinate remains monotonic. Yaw itself lags
            # while steering slews and must not be used as distance travelled.
            progress = (math.asin(max(-1.0, min(1.0, curvature * local[0]))) / curvature
                        if abs(curvature) > 1e-8 else local[0])
            expected = propagate((0., 0., 0.), progress, curvature)
        yaw_error = abs(math.atan2(math.sin(local[2] - expected[2]),
                                   math.cos(local[2] - expected[2])))
        # The yaw tolerance follows the physics of the arc: the steering-slew
        # lag plus 0.02 rad, growing with the turn made so far. A fixed 0.10 rad
        # would accept a robot that never turns during a 0.10 rad correction.
        yaw_tolerance = min(self.heading_error,
                            self.slew_lag[self.index] + 0.02 + 0.3 * abs(expected[2]))
        if (progress * distance < -0.002 * abs(distance) or
                math.hypot(local[0] - expected[0], local[1] - expected[1]) > self.position_error or
                yaw_error > yaw_tolerance or abs(progress) > abs(distance) + 0.01):
            self.stopped = True
            return {**stop, 'reason': 'odometry diverged from planned primitive',
                    'detail': (f'primitive {self.index + 1}/{len(self.primitives)}, '
                               f'progress {progress:+.3f}/{distance:+.3f} m, '
                               f'position error {math.hypot(local[0] - expected[0], local[1] - expected[1]):.3f} m, '
                               f'yaw error {yaw_error:.3f}/{yaw_tolerance:.3f} rad')}
        if abs(progress) >= abs(distance) - 0.002:
            self.stopping = True
            return {**stop, 'state': 'WAIT_STOP', 'reason': 'primitive complete'}
        if beta != 0.0:
            velocity = math.copysign(self.crab_speed, distance)
            return {'state': 'MOVING', 'linear_x': velocity * math.cos(beta),
                    'linear_y': velocity * math.sin(beta), 'angular_z': 0.0,
                    'frame': CRAB_FRAME, 'reason': 'guarded crab request'}
        velocity = math.copysign(self.speed, distance)
        return {'state': 'MOVING', 'linear_x': velocity, 'linear_y': 0.0,
                'angular_z': velocity * curvature, 'frame': 'base_footprint',
                'reason': 'guarded primitive request'}
