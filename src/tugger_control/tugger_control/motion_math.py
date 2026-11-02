"""ROS-independent smooth 4WS command utilities."""
import math


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
