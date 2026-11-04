"""Pure, bounded QR visual-servo proposal logic.

This module deliberately does not publish ROS messages.  A future motion
arbiter may consume its result only during the explicitly armed precision
docking phase.
"""

import math


def bounded_qr_correction(lateral_m, forward_m, yaw_rad, *,
                          max_speed=0.05, max_turn=0.25,
                          position_tolerance=0.02,
                          yaw_tolerance=math.radians(1.0)):
    """Return a slow correction proposal, hold state, or completion state."""
    values = (lateral_m, forward_m, yaw_rad)
    if not all(math.isfinite(value) for value in values):
        return {'state': 'STOP', 'reason': 'Nonfinite QR observation'}
    if math.hypot(lateral_m, forward_m) <= position_tolerance and \
            abs(yaw_rad) <= yaw_tolerance:
        return {'state': 'ALIGNED', 'linear_x': 0.0, 'angular_z': 0.0}
    if abs(forward_m) < 0.002:
        return {'state': 'STOP', 'reason':
                'target requires a lateral or heading maneuver outside the verified correction envelope'}
    # Signed body motion requests either direction; the 4WS controller's
    # reversal guard decelerates through zero. Convergence is not guaranteed.
    scale = min(1.0, max(abs(lateral_m), abs(forward_m)) / 0.10)
    linear_x = max(-max_speed, min(max_speed, forward_m * 0.5)) * scale
    # For a target behind the robot, yaw must reverse to reduce lateral error:
    # marker lateral velocity is approximately -yaw_rate * marker_forward.
    direction = 1.0 if forward_m > 0.0 else -1.0
    angular_z = max(-max_turn, min(max_turn, direction * 2.0 * lateral_m - yaw_rad))
    return {
        'state': 'CORRECT', 'linear_x': linear_x, 'angular_z': angular_z,
        'reason': 'Bounded QR alignment proposal; requires motion-arbiter approval',
    }
