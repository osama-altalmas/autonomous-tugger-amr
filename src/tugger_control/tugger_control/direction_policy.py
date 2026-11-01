"""Phase 15 front/rear policy for a dynamically planned Nav2 path.

This module is deliberately ROS-message independent so its geometry can be
tested without starting Gazebo.  It converts a Nav2 path's XY geometry into
body headings only after checking the live robot yaw.  Mission progress always
remains from the first point toward the last point.
"""

import math


def wrap(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _initial_heading(positions, minimum_displacement=0.01):
    if len(positions) < 2:
        raise ValueError('Path has fewer than two points')
    first_x, first_y = positions[0]
    for x, y in positions[1:]:
        dx, dy = x - first_x, y - first_y
        if math.hypot(dx, dy) >= minimum_displacement:
            return math.atan2(dy, dx)
    raise ValueError('Path has no measurable initial displacement')


def _tangent_headings(positions):
    """Give every path position the tangent toward its next nonzero point."""
    headings = []
    for index, (x, y) in enumerate(positions):
        heading = None
        for next_x, next_y in positions[index + 1:]:
            dx, dy = next_x - x, next_y - y
            if math.hypot(dx, dy) >= 1e-7:
                heading = math.atan2(dy, dx)
                break
        headings.append(heading)
    for index in range(len(headings) - 2, -1, -1):
        if headings[index] is None:
            headings[index] = headings[index + 1]
    if not headings or headings[0] is None:
        raise ValueError('Path has no measurable displacement')
    headings[-1] = headings[-1] if headings[-1] is not None else headings[-2]
    return headings


def prepare_dynamic_path(positions, robot_yaw, angular_tolerance_deg=20.0):
    """Select the leading end and produce execution-only body headings.

    ``positions`` is an iterable of ``(x, y)`` map coordinates.  The input is
    never changed.  A valid plan must leave the robot either front-aligned or
    rear-aligned with its first travel tangent; turning an ambiguous plan into
    an accidental direction change is prohibited.
    """
    if not 0 < angular_tolerance_deg < 90:
        raise ValueError('Angular tolerance must be between 0 and 90 degrees')
    if not math.isfinite(robot_yaw):
        raise ValueError('Robot heading must be finite')

    points = [(float(x), float(y)) for x, y in positions]
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError('Path contains a nonfinite position')
    travel_heading = _initial_heading(points)
    heading_difference = abs(wrap(travel_heading - robot_yaw))
    tolerance = math.radians(angular_tolerance_deg)

    if heading_difference <= tolerance:
        leading_end, body_direction, velocity_sign, offset = (
            'FRONT', 'FORWARD', 1, 0.0)
    elif abs(math.pi - heading_difference) <= tolerance:
        leading_end, body_direction, velocity_sign, offset = (
            'REAR', 'REVERSE', -1, math.pi)
    else:
        raise ValueError(
            'AMBIGUOUS START: live body and planned travel headings differ by '
            f'{math.degrees(heading_difference):.2f} degrees')

    tangents = _tangent_headings(points)
    body_headings = [wrap(tangent + offset) for tangent in tangents]
    start_error = abs(wrap(body_headings[0] - robot_yaw))
    if start_error > tolerance:
        raise ValueError('Adapted path does not match the live body heading')

    return {
        'positions': points,
        'body_headings': body_headings,
        'leading_end': leading_end,
        'route_progress_direction': 'FORWARD',
        'body_motion_direction': body_direction,
        'body_velocity_sign': velocity_sign,
        'travel_heading_deg': math.degrees(wrap(travel_heading)),
        'robot_heading_deg': math.degrees(wrap(robot_yaw)),
        'heading_difference_deg': math.degrees(heading_difference),
    }
