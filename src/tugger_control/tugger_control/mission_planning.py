"""Pure, motion-free selection of destination missions and saved-route rejoins."""

import math

from tugger_control.direction_policy import prepare_dynamic_path


DIRECT_REJOIN_DISTANCE_M = 0.25
MAX_CONNECTOR_DISTANCE_M = 3.0
TERMINAL_REJOIN_EXCLUSION_M = 0.50
PLAN_START_POSITION_TOLERANCE_M = 0.25
PLAN_START_YAW_TOLERANCE_RAD = math.radians(20.0)


def _distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _remaining_length(poses, index):
    return sum(_distance(first, second) for first, second in
               zip(poses[index:], poses[index + 1:]))


def select_destination_plan(destination, candidates, robot_pose,
                            direct_distance=DIRECT_REJOIN_DISTANCE_M,
                            max_connector_distance=MAX_CONNECTOR_DISTANCE_M,
                            destination_poses=()):
    """Select a forward-only saved-route suffix for a requested destination.

    ``candidates`` contains ``(key, route, poses)`` records.  The result is a
    preview decision only: it never authorizes execution or publishes motion.
    A route may be joined at an interior pose, but never after its last pose.
    """
    if direct_distance <= 0 or max_connector_distance < direct_distance:
        raise ValueError('Invalid rejoin distance limits')
    x, y, yaw = (float(value) for value in robot_pose)
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ValueError('Robot pose must be finite')
    if any(_distance((x, y), pose) <= direct_distance
           for pose in destination_poses):
        return {
            'mode': 'ALREADY_AT_DESTINATION',
            'destination': destination,
            'reason': 'Current pose is within the destination arrival tolerance',
        }

    valid = []
    for key, route, poses in candidates:
        if route.get('target') != destination or len(poses) < 2:
            continue
        eligible = [index for index in range(len(poses) - 1)
                    if _remaining_length(poses, index) >= TERMINAL_REJOIN_EXCLUSION_M]
        if not eligible:
            continue
        index = min(eligible,
                    key=lambda i: _distance((x, y), poses[i]))
        gap = _distance((x, y), poses[index])
        if gap > max_connector_distance:
            continue
        suffix = [list(pose) for pose in poses[index:]]
        try:
            direction = prepare_dynamic_path(
                [(pose[0], pose[1]) for pose in suffix], yaw)
        except ValueError:
            continue
        mode = 'SAVED_REJOIN' if gap <= direct_distance else 'SHORT_CONNECTOR'
        valid.append((gap, index, key, route, suffix, direction, mode))

    if not valid:
        return {
            'mode': 'LIVE_RECOVERY_REQUIRED',
            'destination': destination,
            'reason': 'No route to destination has a safe, heading-compatible rejoin',
        }

    gap, index, key, route, suffix, direction, mode = min(
        valid, key=lambda item: (item[0], -item[1], item[2]))
    return {
        'mode': mode,
        'destination': destination,
        'route_key': key,
        'route': route,
        'poses': suffix,
        'rejoin_index': index,
        'gap_m': gap,
        'direction': direction,
    }


def assess_route_deviation(poses, robot_pose, previous_index=0,
                           tolerance_m=DIRECT_REJOIN_DISTANCE_M):
    """Report whether a moving mission remains on a monotonic route suffix.

    The result is intentionally a supervisory decision, not a command.  A
    runtime mission owner can use ``REJOIN_REQUIRED`` to request a bounded
    connector, while never moving route progress backwards.
    """
    if len(poses) < 2:
        raise ValueError('Route needs at least two poses')
    if not 0 <= previous_index < len(poses) - 1:
        raise ValueError('Previous route index is outside the forward route')
    index = min(range(previous_index, len(poses) - 1),
                key=lambda i: _distance(robot_pose, poses[i]))
    gap = _distance(robot_pose, poses[index])
    return {
        'state': 'ON_ROUTE' if gap <= tolerance_m else 'REJOIN_REQUIRED',
        'rejoin_index': index,
        'cross_track_error_m': gap,
        'remaining_poses': [list(pose) for pose in poses[index:]],
    }


def plan_is_current(planned_pose, current_pose,
                    position_tolerance_m=PLAN_START_POSITION_TOLERANCE_M,
                    yaw_tolerance_rad=PLAN_START_YAW_TOLERANCE_RAD):
    """Check whether a preview can still be dispatched without replanning.

    A mission preview is tied to the live pose and leading-end decision used
    during preflight. It must be recomputed when either differs materially at
    RUN time; this function makes that decision without ROS side effects.
    """
    if position_tolerance_m <= 0 or not 0 < yaw_tolerance_rad < math.pi:
        raise ValueError('Invalid plan-currentness tolerances')
    planned = tuple(float(value) for value in planned_pose)
    current = tuple(float(value) for value in current_pose)
    if len(planned) != 3 or len(current) != 3:
        raise ValueError('Poses must contain x, y, and yaw')
    if not all(math.isfinite(value) for value in (*planned, *current)):
        raise ValueError('Poses must be finite')
    position_error = _distance(planned, current)
    yaw_error = abs(math.atan2(
        math.sin(planned[2] - current[2]),
        math.cos(planned[2] - current[2])))
    return {
        'current': (position_error <= position_tolerance_m and
                    yaw_error <= yaw_tolerance_rad),
        'position_error_m': position_error,
        'yaw_error_rad': yaw_error,
    }
