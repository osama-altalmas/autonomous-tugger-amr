"""Crab steering for the 4WS tugger: all four wheels at one angle.

With every wheel steered to the same angle ``beta`` the robot translates
along that direction without turning: a diagonal sideways move. Industrial
4WS and omnidirectional AGVs use it for final positioning, because a few
centimetres of lateral error is one short straight move instead of a
parallel-parking sequence of arcs.

Limits for a real machine: ``beta`` at most CRAB_MAX_ANGLE_RAD (50 degrees,
under the 54 degree joint limit), low speed only, and the robot is stopped
whenever the wheels change angle (the controller enforces both).

Wheel order everywhere: front left, front right, rear left, rear right.
Pure functions; no robot commands.
"""

import math

from tugger_control.turn_in_place import STEERING_LIMIT_RAD

# Crab direction relative to the robot's forward (or backward) axis.
CRAB_MAX_ANGLE_RAD = math.radians(50.0)
# Crab is a positioning move, not a travel mode.
CRAB_MAX_SPEED = 0.10
# Header frame of a crab command (linear.x, linear.y body velocity).
CRAB_FRAME = 'crab'

assert CRAB_MAX_ANGLE_RAD < STEERING_LIMIT_RAD


def crab_steering(beta):
    """Wheel angles for crab direction ``beta`` (rad)."""
    return (beta, beta, beta, beta)


def crab_command(vx, vy):
    """Body velocity -> (signed wheel speed m/s, wheel angle rad).

    The wheels can point within CRAB_MAX_ANGLE_RAD of the robot axis, so a
    backward-diagonal move rolls the wheels backwards at the mirrored angle.
    Raises ValueError for a direction the wheels cannot reach.
    """
    if not (math.isfinite(vx) and math.isfinite(vy)):
        raise ValueError('nonfinite crab command')
    speed = math.hypot(vx, vy)
    if speed < 1e-9:
        return 0.0, 0.0
    sign = 1.0 if vx >= 0.0 else -1.0
    beta = math.atan2(sign * vy, sign * vx)
    if abs(beta) > CRAB_MAX_ANGLE_RAD + 1e-9:
        raise ValueError(f'crab direction {math.degrees(beta):.0f} deg beyond the wheel limit')
    return sign * speed, beta


def crab_wheel_speeds(speed, wheel_radius):
    """Wheel angular speeds (rad/s): every wheel rolls at the body speed."""
    omega = speed / wheel_radius
    return (omega, omega, omega, omega)


def crab_propagate(pose, distance, beta):
    """Pose after a crab move of signed ``distance`` along wheel angle ``beta``."""
    x, y, yaw = pose
    return (x + distance * math.cos(yaw + beta),
            y + distance * math.sin(yaw + beta),
            yaw)
