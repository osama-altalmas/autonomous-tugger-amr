"""Motion-free, facility-independent departure construction.

A departing robot retraces its *configured* docking approach in reverse XY
order, retaining its measured body orientation. The caller must independently
validate the loaded collision sweep, fresh scene, and departure motion owner.
The resulting path is a candidate, not authorization to drive.
"""

import math

from tugger_control.direction_policy import prepare_dynamic_path, wrap
from tugger_control.steering_geometry import MAX_VIRTUAL_STEERING_RAD


def plan_station_departure(current_pose, approach_poses, *,
                           maximum_dock_gap_m=0.20,
                           maximum_body_yaw_error_deg=20.0,
                           minimum_exit_distance_m=0.50,
                           wheelbase_m=0.58,
                           maximum_steering_rad=MAX_VIRTUAL_STEERING_RAD):
    """Return an execution-direction-consistent *candidate* to approach staging.

    ``approach_poses`` runs from the free-space staging point into the dock.
    Yaw values in that source may encode *travel*, not robot body orientation;
    departure body headings are derived solely from XY and measured body yaw.
    Neither the station name nor its coordinates are hardcoded here.
    """
    current = tuple(float(value) for value in current_pose)
    approach = [tuple(float(value) for value in pose) for pose in approach_poses]
    if (len(current) != 3 or len(approach) < 2 or
            any(len(pose) != 3 for pose in approach) or
            not all(math.isfinite(v) for pose in [current, *approach] for v in pose)):
        raise ValueError('Departure requires finite current and approach x/y/yaw poses')
    if (not 0 < maximum_dock_gap_m <= 0.50 or
            not 0 < maximum_body_yaw_error_deg < 45 or
            minimum_exit_distance_m <= 0 or
            not math.isfinite(wheelbase_m) or wheelbase_m <= 0 or
            not math.isfinite(maximum_steering_rad) or
            not 0 < maximum_steering_rad < math.pi / 2):
        raise ValueError('Invalid departure geometry tolerances')
    gap = math.hypot(current[0] - approach[-1][0],
                    current[1] - approach[-1][1])
    if gap > maximum_dock_gap_m:
        raise ValueError(f'Robot is {gap:.3f} m from the configured dock; cannot depart')

    # Start at the measured pose and join the reversed approach at its first
    # sample at least 2.5 cm away (a robot on the approach line: unchanged).
    # A robot slightly beside the line (for example after turning in place)
    # instead blends onto it with a smooth cosine offset whose length keeps
    # the curvature within the steering limit. Both pass the same checks.
    reversed_approach = list(reversed(approach))
    outbound = (approach[-2][0] - approach[-1][0],
                approach[-2][1] - approach[-1][1])
    curvature_limit = 2.0 * math.tan(maximum_steering_rad) / wheelbase_m
    next_index = next((index for index, point in enumerate(reversed_approach)
                       if math.hypot(point[0] - current[0], point[1] - current[1])
                       >= 0.025), None)
    if next_index is None:
        raise ValueError('Docking approach has no usable departure displacement')
    direct = [(current[0], current[1])] + [
        (point[0], point[1]) for point in reversed_approach[next_index:]]
    try:
        prepared, selected, length = _departure_candidate(
            current, direct, outbound, maximum_body_yaw_error_deg, curvature_limit)
    except ValueError as exc:
        blended = _blended_positions(current, reversed_approach, outbound, curvature_limit)
        if blended is None:
            raise exc
        prepared, selected, length = _departure_candidate(
            current, blended, outbound, maximum_body_yaw_error_deg, curvature_limit)
    if length < minimum_exit_distance_m:
        raise ValueError('Configured approach does not lead far enough out of the station')

    return {
        'poses': prepared,
        'staging_pose': prepared[-1],
        'leading_end': selected['leading_end'],
        'body_motion_direction': selected['body_motion_direction'],
        'length_m': length,
        'dock_gap_m': gap,
    }



def _blended_positions(current, reversed_approach, outbound, curvature_limit):
    """Positions easing from beside the approach line onto it, or None."""
    norm = math.hypot(*outbound)
    if norm < 1e-9:
        return None
    unit = (outbound[0] / norm, outbound[1] / norm)
    end = reversed_approach[0]
    along = (current[0] - end[0]) * unit[0] + (current[1] - end[1]) * unit[1]
    foot = (end[0] + along * unit[0], end[1] + along * unit[1])
    offset = (current[0] - foot[0], current[1] - foot[1])
    lateral = math.hypot(*offset)
    if lateral < 1e-4 or lateral > 0.10:
        return None
    # A cosine offset of amplitude e over length D has peak curvature
    # e*(pi/D)^2/2; keep it at half the steering limit.
    blend = max(0.3, math.pi * math.sqrt(lateral / curvature_limit))
    positions = [(current[0], current[1])]
    for point in reversed_approach:
        ahead = (point[0] - current[0]) * unit[0] + (point[1] - current[1]) * unit[1]
        if ahead < 0.01:
            continue
        weight = 0.5 * (1.0 + math.cos(math.pi * ahead / blend)) if ahead < blend else 0.0
        candidate = (point[0] + weight * offset[0], point[1] + weight * offset[1])
        if math.hypot(candidate[0] - positions[-1][0], candidate[1] - positions[-1][1]) >= 0.01:
            positions.append(candidate)
    return positions if len(positions) >= 3 else None


def _departure_candidate(current, positions, outbound, maximum_body_yaw_error_deg,
                         curvature_limit):
    """Validate a departure through ``positions``; return (poses, selection, length)."""
    # Avoid an initial backwards hop from a slightly overshot measured dock pose.
    toward = (positions[1][0] - positions[0][0],
              positions[1][1] - positions[0][1])
    if outbound[0] * toward[0] + outbound[1] * toward[1] <= 0:
        raise ValueError('Departure would initially travel back into the station')

    selected = prepare_dynamic_path(
        positions, current[2], angular_tolerance_deg=maximum_body_yaw_error_deg)
    prepared = [[x, y, heading] for (x, y), heading in
                zip(positions, selected['body_headings'])]
    # If dock-end registration is slightly offset, never command a sudden turn.
    initial_yaw_error = abs(wrap(prepared[0][2] - current[2]))
    if initial_yaw_error > math.radians(maximum_body_yaw_error_deg):
        raise ValueError('Departure body heading disagrees with the measured robot pose')
    length = 0.0
    for before, after in zip(prepared, prepared[1:]):
        travel = math.hypot(after[0] - before[0], after[1] - before[1])
        angle = abs(wrap(after[2] - before[2]))
        if travel < 1e-8 or angle / travel > curvature_limit + 1e-6:
            raise ValueError('Departure curvature exceeds the 4WS steering limits')
        length += travel
    return prepared, selected, length
