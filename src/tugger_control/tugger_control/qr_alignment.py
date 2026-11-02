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
    if abs(lateral_m) <= position_tolerance and abs(forward_m) <= position_tolerance and \
            abs(yaw_rad) <= yaw_tolerance:
        return {'state': 'ALIGNED', 'linear_x': 0.0, 'angular_z': 0.0}
    # Signed body motion lets the 4WS controller converge from either side;
    # its reversal guard still decelerates through zero.
    scale = min(1.0, max(abs(lateral_m), abs(forward_m)) / 0.10)
    linear_x = max(-max_speed, min(max_speed, forward_m * 0.5)) * scale
    # Positive lateral error means the frame is to robot-left.  Positive yaw
    # curves a positive-speed 4WS tugger toward that side.
    angular_z = max(-max_turn, min(max_turn, 2.0 * lateral_m - yaw_rad))
    return {
        'state': 'CORRECT', 'linear_x': linear_x, 'angular_z': angular_z,
        'reason': 'Bounded QR alignment proposal; requires motion-arbiter approval',
    }
