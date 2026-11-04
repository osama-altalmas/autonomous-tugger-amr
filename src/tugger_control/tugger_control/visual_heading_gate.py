"""Pure angular consistency checks for simulator visual docking.

The stationary floor QR supplies *relative* image orientation; the approved
map-frame route supplies the absolute heading. Their initial offset is not
known without camera/texture calibration, so this deliberately NEVER treats
raw QR image yaw as an absolute world heading measurement.
"""

import math


def angle_delta(first, second):
    """Signed shortest angular change from second to first."""
    if not (math.isfinite(first) and math.isfinite(second)):
        raise ValueError('invalid heading')
    return math.atan2(math.sin(first - second), math.cos(first - second))


def relative_qr_heading_consistent(qr_yaw, reference_qr_yaw,
                                   map_yaw, reference_map_yaw,
                                   tolerance=math.radians(5)):
    """Check equal relative yaw changes (allowing the fixed camera handedness).

    This check cannot certify an absolute 1 degree QR yaw measurement. Absolute
    yaw still requires a verified map TF and an approved final docking heading.
    """
    observed = angle_delta(qr_yaw, reference_qr_yaw)
    measured = angle_delta(map_yaw, reference_map_yaw)
    return min(abs(angle_delta(observed, measured)),
               abs(angle_delta(observed, -measured))) <= tolerance
