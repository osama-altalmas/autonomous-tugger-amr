"""Turn in place for the 4WS tugger: geometry, timing and wheel commands.

Turning in place is an optional capability on top of car-like driving. Each
wheel steers tangent to a circle around the robot centre, so the robot
rotates without moving its centre. For a wheelbase L and track W every wheel
needs atan((L/2) / (W/2)) = 52.8 degrees; the steering joints allow
STEERING_LIMIT_RAD (54 degrees), the least angle with a safety margin.
Car-like driving keeps its own 45 degree limit (steering_geometry).

Wheel order everywhere: front left, front right, rear left, rear right.
Pure functions; no robot commands.
"""

import math

from tugger_control.steering_geometry import (  # noqa: F401  (re-exported)
    TRACK_M, WHEEL_RADIUS_M, WHEELBASE_M, ackermann_steering, curvature_from_virtual)
# Each wheel tangent to the circle through the four wheel contact points.
PIVOT_STEERING_RAD = math.atan2(WHEELBASE_M / 2.0, TRACK_M / 2.0)
STEERING_LIMIT_RAD = math.radians(54.0)
PIVOT_RADIUS_M = math.hypot(WHEELBASE_M / 2.0, TRACK_M / 2.0)
# Steering pattern for counter-clockwise rotation (positive yaw rate) with
# the wheel rolling direction signs that go with it.
PIVOT_STEERING = (-PIVOT_STEERING_RAD, PIVOT_STEERING_RAD,
                  PIVOT_STEERING_RAD, -PIVOT_STEERING_RAD)
PIVOT_ROLL_SIGN = (-1.0, 1.0, -1.0, 1.0)

# Rotation limits. Loaded, the 1.3 m cargo square sweeps a 0.96 m radius, so
# it turns more slowly.
MAX_YAW_RATE_UNLOADED = 0.5
MAX_YAW_RATE_LOADED = 0.3
# Gentle, so the passive front wheels (low lateral friction) do not slip.
YAW_ACCELERATION = 0.3
# Wheels hold the pivot pattern this long before the turn starts, so the
# steering joints have settled on their targets.
PIVOT_SETTLE_S = 0.3
# A turn in place shifts the centre a little (steering a standing wheel
# scrubs it; measured in Gazebo about 3 cm per 90-degree turn with all four
# wheels driven). Up to this shift the robot leaves straight from the measured
# pose; beyond it the camera stage re-centres it on the station marker first.
TURN_SHIFT_TOLERANCE_M = 0.05
# Clearance kept around the whole swept circle.
ROTATION_MARGIN_M = 0.05
# Time to settle after each steering change before driving.
SETTLE_S = 0.5


def drive_steering(delta):
    """Car-like wheel angles for the virtual steering angle ``delta``.

    Counter-phase Ackermann: front and rear opposite, inner wheels steer more
    (steering_geometry.ackermann_steering). Straight ahead is all zeros.
    """
    return ackermann_steering(curvature_from_virtual(delta))


def steering_reached(measured, target, tolerance=0.02):
    """True when every wheel is within ``tolerance`` rad of its target."""
    return all(abs(a - b) <= tolerance for a, b in zip(measured, target))


def pivot_wheel_speeds(yaw_rate, wheel_radius=WHEEL_RADIUS_M):
    """Wheel angular speeds (rad/s) for turning in place at ``yaw_rate``."""
    surface = yaw_rate * PIVOT_RADIUS_M / wheel_radius
    return tuple(sign * surface for sign in PIVOT_ROLL_SIGN)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rotation_to_axis(yaw, axis):
    """Smallest rotation that lines the body up with ``axis`` (either end first).

    The 4WS robot drives forward and backward alike, so a body heading and its
    opposite serve the same straight line: the result is within +-90 degrees.
    """
    turn = wrap(axis - yaw)
    if turn > math.pi / 2:
        turn -= math.pi
    elif turn < -math.pi / 2:
        turn += math.pi
    return turn


def approach_axis(approach_poses):
    """Direction of travel at the end of an approach (staging -> dock)."""
    if len(approach_poses) < 2:
        raise ValueError('an approach needs at least two poses')
    end = approach_poses[-1]
    for pose in reversed(approach_poses[:-1]):
        if math.hypot(end[0] - pose[0], end[1] - pose[1]) >= 0.3:
            return math.atan2(end[1] - pose[1], end[0] - pose[0])
    first = approach_poses[0]
    return math.atan2(end[1] - first[1], end[0] - first[0])


def _profile_time(angle, rate, acceleration):
    """Trapezoidal (or triangular) yaw profile duration for ``angle`` rad."""
    angle = abs(angle)
    if angle < 1e-9:
        return 0.0
    ramp = rate * rate / acceleration
    if angle <= ramp:
        return 2.0 * math.sqrt(angle / acceleration)
    return angle / rate + rate / acceleration


def rotation_time_s(angle, loaded, steering_rate=2.2):
    """Time to turn in place by ``angle`` and be ready to drive again.

    Wheels steer to the pivot pattern and back (each wheel at most
    PIVOT_STEERING_RAD + 45 degrees from a car-like angle), the body turns on
    a trapezoidal profile, and the robot settles after each steering change.
    """
    if abs(angle) < math.radians(0.5):
        return 0.0
    steer = 2.0 * (PIVOT_STEERING_RAD + math.pi / 4) / steering_rate
    rate = MAX_YAW_RATE_LOADED if loaded else MAX_YAW_RATE_UNLOADED
    return steer + _profile_time(angle, rate, YAW_ACCELERATION) + 3.0 * SETTLE_S


def rotation_sweep(pose, angle=2.0 * math.pi, step=math.radians(2.0)):
    """Poses at the same centre, from ``pose``'s yaw through ``angle``."""
    count = max(1, math.ceil(abs(angle) / step))
    return [(pose[0], pose[1], wrap(pose[2] + angle * index / count))
            for index in range(count + 1)]


def pivot_odometry(steering, wheel_speeds, wheel_positions=None,
                   wheel_radius=WHEEL_RADIUS_M):
    """Body velocity (vx, vy, yaw rate) from measured wheels, least squares.

    Works for any steering pattern: each wheel at (x, y) rolling along its
    steering angle at its surface speed constrains the rigid-body motion
    (vx - w*y, vy + w*x). ``steering`` and ``wheel_speeds`` are per wheel
    (rad and rad/s); wheels with unknown speed may be None.
    """
    if wheel_positions is None:
        wheel_positions = ((WHEELBASE_M / 2, TRACK_M / 2), (WHEELBASE_M / 2, -TRACK_M / 2),
                           (-WHEELBASE_M / 2, TRACK_M / 2), (-WHEELBASE_M / 2, -TRACK_M / 2))
    rows, values = [], []
    for (x, y), angle, speed in zip(wheel_positions, steering, wheel_speeds):
        if speed is None:
            continue
        surface = speed * wheel_radius
        rows.append((1.0, 0.0, -y))
        values.append(surface * math.cos(angle))
        rows.append((0.0, 1.0, x))
        values.append(surface * math.sin(angle))
    if len(rows) < 4:
        raise ValueError('at least two wheels are needed')
    # Normal equations for the 3 unknowns.
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    atb = [sum(r[i] * v for r, v in zip(rows, values)) for i in range(3)]
    return tuple(_solve3(ata, atb))


def _solve3(a, b):
    """Solve a 3x3 linear system by Gaussian elimination with pivoting."""
    m = [row[:] + [value] for row, value in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError('wheel geometry does not determine the motion')
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r != col:
                factor = m[r][col] / m[col][col]
                m[r] = [x - factor * y for x, y in zip(m[r], m[col])]
    return [m[i][3] / m[i][i] for i in range(3)]
