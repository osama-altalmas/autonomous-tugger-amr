"""ROS-independent smooth 4WS command utilities."""
import math


def speed_profile(v, omega, *, straight_limit=1.2, curve_limit=0.6,
                  curve_threshold=0.25):
    """Limit route speed while preserving the requested path curvature.

    ``omega / v`` is the planar curvature commanded by Nav2.  Gentle heading
    corrections retain the straight-line limit; genuine curves use the lower
    limit and scale omega with v so the 4WS path does not change.
    """
    if not all(math.isfinite(value) for value in (v, omega, straight_limit,
                                                   curve_limit, curve_threshold)):
        raise ValueError('Non-finite speed-profile input')
    if straight_limit <= 0 or curve_limit <= 0 or curve_threshold < 0:
        raise ValueError('Invalid speed-profile limit')
    if abs(v) < 1e-6:
        return 0.0, omega
    is_curve = abs(omega / v) >= curve_threshold
    limit = curve_limit if is_curve else straight_limit
    limited = math.copysign(min(abs(v), limit), v)
    return limited, omega * limited / v


def approach(current, target, rate, dt):
    if rate < 0 or dt < 0:
        raise ValueError('rate and dt must be nonnegative')
    step = rate * dt
    return current + max(-step, min(step, target - current))


def steering_for_twist(v, omega, wheelbase, max_steering):
    if abs(v) < 1e-5:
        return 0.0
    return max(-max_steering, min(max_steering, math.atan(wheelbase * omega / (2 * v))))


def speed_limit_for_steering(delta, wheelbase, max_lateral_accel, max_speed):
    kappa = abs(2 * math.tan(delta) / wheelbase)
    return min(max_speed, math.sqrt(max_lateral_accel / kappa)) if kappa > 1e-6 else max_speed


def wheel_speeds(v, delta, wheelbase, track_width, wheel_radius):
    omega = 2 * v * math.tan(delta) / wheelbase
    c = math.cos(delta)
    return ((v - omega * track_width / 2) / (c * wheel_radius),
            (v + omega * track_width / 2) / (c * wheel_radius), omega)
