"""Steering geometry of the 4WS tugger: the single source for its turning limit.

Car-like driving uses counter-phase Ackermann steering: front and rear axles
steer opposite ways, and on each axle the inner wheel steers more than the
outer one, so all four wheels point around one turning centre on the robot's
lateral centre line. No wheel slides sideways in a curve. (Until 2026-11-13
both wheels of an axle got the same angle; in a 1 m corner at 0.6 m/s the
worst wheel then slid sideways at about 4 cm/s, which wheel odometry cannot
see, so the pose drifted in every curve.)

The turning limit follows from the inner wheel's 45 degree limit: the
tightest curve is MIN_TURN_RADIUS_M (0.51 m) at the robot centre. Every module
that bounds curvature imports it from here; nav2.yaml and the C++ planner and
follower defaults carry the same numbers, checked by a test.

A "virtual" single steering angle delta (bicycle model, tan(delta) =
curvature * wheelbase / 2) is kept as the controller's state and in
diagnostics; ackermann_steering() turns it into the four wheel angles.
"""
import math

WHEELBASE_M = 0.58
TRACK_M = 0.44
WHEEL_RADIUS_M = 0.055

# Wheel contact positions (x forward, y left), in the steering controller's
# joint order: front left, front right, rear left, rear right.
WHEEL_POSITIONS = ((WHEELBASE_M / 2.0, TRACK_M / 2.0), (WHEELBASE_M / 2.0, -TRACK_M / 2.0),
                   (-WHEELBASE_M / 2.0, TRACK_M / 2.0), (-WHEELBASE_M / 2.0, -TRACK_M / 2.0))

# Car-like limit of any single wheel.
MAX_WHEEL_STEERING_RAD = math.radians(45.0)


def max_curvature(max_wheel_steering=MAX_WHEEL_STEERING_RAD, wheelbase=WHEELBASE_M,
                  track=TRACK_M):
    """Tightest path curvature with the inner wheels at their limit."""
    t = math.tan(max_wheel_steering)
    return t / (wheelbase / 2.0 + track / 2.0 * t)


MAX_CURVATURE = max_curvature()                  # 1.961 /m
MIN_TURN_RADIUS_M = 1.0 / MAX_CURVATURE          # 0.510 m


def curvature_from_virtual(delta, wheelbase=WHEELBASE_M):
    """Path curvature of the virtual single steering angle."""
    return 2.0 * math.tan(delta) / wheelbase


def virtual_from_curvature(curvature, wheelbase=WHEELBASE_M):
    """Virtual single steering angle for a path curvature."""
    return math.atan(wheelbase * curvature / 2.0)


MAX_VIRTUAL_STEERING_RAD = virtual_from_curvature(MAX_CURVATURE)   # 29.6 deg


def ackermann_steering(curvature, positions=WHEEL_POSITIONS):
    """Wheel angles (front left, front right, rear left, rear right).

    The turning centre is at (0, 1 / curvature); the wheel at (x, y) points
    perpendicular to the line from it: angle = atan2(x k, 1 - y k).
    """
    return tuple(math.atan2(x * curvature, 1.0 - y * curvature) for x, y in positions)


def rolling_speeds(v, omega, angles, wheel_radius=WHEEL_RADIUS_M, positions=WHEEL_POSITIONS):
    """Wheel speeds (rad/s) for body speed v and yaw rate omega.

    Each wheel rolls at the body velocity at its contact point projected on
    its own heading; with Ackermann angles that is the whole velocity.
    """
    return tuple(((v - omega * y) * math.cos(angle) + omega * x * math.sin(angle)) / wheel_radius
                 for (x, y), angle in zip(positions, angles))
