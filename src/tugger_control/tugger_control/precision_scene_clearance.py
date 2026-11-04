"""Offline building block for Gazebo precision-approach clearance.

This is not wired into mission dispatch and does not authorize motion.  It
checks sampled robot boxes against a supplied scene; callers must establish
the scene's completeness and freshness and provide continuous runtime safety.
"""

import math

from tugger_control.clearance_2_5d import Box, boxes_overlap, collision_pairs
from tugger_control.warehouse_clearance import robot_boxes


def sampled_precision_clearance(poses, obstacles, lift, *, margin=0.01):
    """Check interpolated unloaded-robot poses against height-aware boxes.

    The rectangular wheel bounds conservatively cover steering positions.
    This sampled test is not a continuous-sweep or sensor-coverage guarantee.
    Returns ``(clear, sample_index, first_collision)``.
    """
    points = [tuple(float(value) for value in pose) for pose in poses]
    if len(points) < 2 or any(len(point) != 3 or
                              not all(math.isfinite(value) for value in point)
                              for point in points):
        raise ValueError('A precision path needs at least two finite x, y, yaw poses')
    if not math.isfinite(lift) or not -0.005 <= lift <= 0.005:
        raise ValueError('Precision approach requires a lowered lift')
    if not math.isfinite(margin) or margin < 0.01:
        raise ValueError('Precision clearance margin must be at least 0.01 m')
    obstacles = list(obstacles)
    if not obstacles or any(not isinstance(obstacle, Box) for obstacle in obstacles):
        raise ValueError('A nonempty, validated collision scene is required')

    sample_index = 0
    for start, end in zip(points, points[1:]):
        rotation = math.atan2(math.sin(end[2] - start[2]),
                              math.cos(end[2] - start[2]))
        count = max(1, math.ceil(math.hypot(end[0] - start[0], end[1] - start[1]) / 0.01),
                    math.ceil(abs(rotation) / 0.02))
        for index in range(count):
            fraction = index / count
            pose = (start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                    start[2] + rotation * fraction)
            hits = collision_pairs(robot_boxes(pose, lift), obstacles, margin=margin)
            if hits:
                return False, sample_index, hits[0]
            sample_index += 1
    hits = collision_pairs(robot_boxes(points[-1], lift), obstacles, margin=margin)
    if hits:
        return False, sample_index, hits[0]
    return True, sample_index + 1, None


def precision_stopping_envelope(pose, lift, obstacles, current_speed, requested_speed,
                                *, reaction_s=0.5, deceleration=0.35):
    """Conservative simulator-only unloaded stopping envelope, not safety-rated.

    The circular expansion bounds translation plus a worst-case 4WS heading
    change over the stopping distance. Scene freshness and completeness must
    be established by the caller on every cycle.
    """
    values = (*pose, lift, current_speed, requested_speed, reaction_s, deceleration)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Non-finite precision guard input')
    if (len(pose) != 3 or not -0.005 <= lift <= 0.005 or
            abs(current_speed) > 0.21 or abs(requested_speed) > 0.051 or
            reaction_s <= 0 or deceleration <= 0):
        raise ValueError('Precision motion outside guarded operating limits')
    obstacles = list(obstacles)
    if not obstacles or any(not isinstance(box, Box) for box in obstacles):
        raise ValueError('Complete validated collision scene required')
    speed = max(abs(current_speed), abs(requested_speed))
    travel = speed * reaction_s + speed * speed / (2 * deceleration)
    rotation = min(math.pi, 2 * travel / 0.58)
    extra = 0.02 + travel + 2 * math.hypot(0.44, 0.29) * math.sin(rotation / 2)
    parts = robot_boxes(pose, lift)
    for obstacle in obstacles:
        expanded = Box(obstacle.name, obstacle.x, obstacle.y, obstacle.yaw,
                       obstacle.half_x + extra, obstacle.half_y + extra,
                       obstacle.z_min - 0.01, obstacle.z_max + 0.01)
        for part in parts:
            if boxes_overlap(part, expanded):
                return False, (part.name, obstacle.name)
    return True, None
