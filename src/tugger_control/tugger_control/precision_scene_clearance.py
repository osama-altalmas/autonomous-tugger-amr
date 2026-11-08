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

    # Spatial index: obstacles bucketed by 1 m cell over their bounding
    # circles, so each sampled pose is tested only against nearby boxes.
    # Exact: a box outside the robot's reach cannot touch it.
    buckets = {}
    for obstacle in obstacles:
        radius = math.hypot(obstacle.half_x, obstacle.half_y) + margin
        for cx in range(math.floor(obstacle.x - radius), math.floor(obstacle.x + radius) + 1):
            for cy in range(math.floor(obstacle.y - radius), math.floor(obstacle.y + radius) + 1):
                buckets.setdefault((cx, cy), []).append(obstacle)

    def nearby(parts):
        found, seen = [], set()
        for part in parts:
            reach = math.hypot(part.half_x, part.half_y)
            for cx in range(math.floor(part.x - reach), math.floor(part.x + reach) + 1):
                for cy in range(math.floor(part.y - reach), math.floor(part.y + reach) + 1):
                    for obstacle in buckets.get((cx, cy), ()):
                        if id(obstacle) not in seen:
                            seen.add(id(obstacle))
                            found.append(obstacle)
        return found

    def hits_at(pose):
        parts = robot_boxes(pose, lift)
        return collision_pairs(parts, nearby(parts), margin=margin)

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
            hits = hits_at(pose)
            if hits:
                return False, sample_index, hits[0]
            sample_index += 1
    hits = hits_at(points[-1])
    if hits:
        return False, sample_index, hits[0]
    return True, sample_index + 1, None


# Counter-phase 4WS at 45 degrees and a 0.58 m wheelbase: |curvature| <= 2/0.58.
FULL_LOCK_CURVATURE = 2.0 / 0.58


def stopping_curvature(curvature_bound):
    """Clamp a curvature bound to [0, full lock]; invalid values mean full lock."""
    if not math.isfinite(curvature_bound) or curvature_bound < 0:
        return FULL_LOCK_CURVATURE
    return min(curvature_bound, FULL_LOCK_CURVATURE)


def precision_stopping_envelope(pose, lift, obstacles, current_speed, requested_speed,
                                *, reaction_s=0.5, deceleration=1.5,
                                max_speed=0.4, max_current_speed=None,
                                curvature_bound=FULL_LOCK_CURVATURE):
    """Conservative simulator-only unloaded stopping envelope, not safety-rated.

    The circular expansion bounds translation plus a worst-case 4WS heading
    change over the stopping distance. Scene freshness and completeness must
    be established by the caller on every cycle.
    """
    values = (*pose, lift, current_speed, requested_speed, reaction_s, deceleration)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Non-finite precision guard input')
    if max_current_speed is None:
        max_current_speed = max_speed
    if not math.isfinite(max_current_speed) or max_current_speed <= 0:
        raise ValueError('Invalid precision current-speed limit')
    if (len(pose) != 3 or not -0.005 <= lift <= 0.005 or
            abs(current_speed) > max_current_speed or abs(requested_speed) > max_speed or
            reaction_s <= 0 or deceleration <= 0):
        raise ValueError('Precision motion outside guarded operating limits')
    obstacles = list(obstacles)
    if not obstacles or any(not isinstance(box, Box) for box in obstacles):
        raise ValueError('Complete validated collision scene required')
    speed = max(abs(current_speed), abs(requested_speed))
    travel = speed * reaction_s + speed * speed / (2 * deceleration)
    # Heading change while stopping: by default the robot is assumed to swing
    # at full steering lock. The gate passes a measured bound instead.
    rotation = min(math.pi, stopping_curvature(curvature_bound) * travel)
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


def choose_precision_speed(pose, lift, obstacles, current_speed, requested_speed,
                           *, max_speed=0.4, minimum_speed=0.0,
                           curvature_bound=FULL_LOCK_CURVATURE,
                           max_current_speed=0.4):
    """Select the fastest precision speed whose stopping envelope is clear.

    The approach controller is allowed to request 0.4 m/s away from the
    station, but the final metres may require a lower speed near the cargo
    legs. Unknown or completely blocked geometry still returns zero/failure.
    """
    if (not math.isfinite(requested_speed) or not math.isfinite(max_speed) or
            not math.isfinite(minimum_speed)):
        raise ValueError('Invalid precision velocity')
    if max_speed <= 0 or minimum_speed < 0 or minimum_speed > max_speed:
        raise ValueError('Invalid precision speed limit')
    limited = max(-max_speed, min(max_speed, requested_speed))
    magnitude = abs(limited)
    fallback_minimum = minimum_speed if magnitude >= minimum_speed else 0.0
    candidates = sorted({magnitude, *(min(magnitude, value) for value in
                                      (0.3, 0.2, 0.1, 0.05, 0.02)
                                      if value >= fallback_minimum), 0.0},
                        reverse=True)
    last_hit = None
    for candidate in candidates:
        value = math.copysign(candidate, limited) if candidate else 0.0
        clear, last_hit = precision_stopping_envelope(
            pose, lift, obstacles, current_speed, value,
            max_speed=max_speed, max_current_speed=max_current_speed,
            curvature_bound=curvature_bound)
        if clear:
            return value, None
    return 0.0, last_hit


def precision_approach_speed_limit(pose, obstacles, *, approach_speed=0.4):
    """Taper the docking approach before the robot enters cargo-leg space.

    The Nav2 docking controller may approach at 0.4 m/s in open space.  Once
    the actual collision scene places the robot near fixed station legs, the
    request is capped before the stopping-envelope guard has to reject it.
    HOME has no ``*_legs`` obstacle, so it keeps the approach limit until the
    camera controller takes ownership.
    """
    if len(pose) != 3 or not all(math.isfinite(value) for value in pose):
        raise ValueError('Invalid precision approach pose')
    if not math.isfinite(approach_speed) or approach_speed <= 0:
        raise ValueError('Invalid precision approach speed')
    leg_boxes = [(model, box) for model, box in obstacles
                 if '_legs' in model or '_legs' in box.name]
    if not leg_boxes:
        return approach_speed

    def distance_to_box(box):
        dx = max(abs(pose[0] - box.x) - box.half_x, 0.0)
        dy = max(abs(pose[1] - box.y) - box.half_y, 0.0)
        return math.hypot(dx, dy)

    distance = min(distance_to_box(box) for _, box in leg_boxes)
    if distance <= 0.65:
        return min(0.10, approach_speed)
    if distance <= 1.0:
        return min(0.10, approach_speed)
    if distance <= 1.4:
        return min(0.20, approach_speed)
    return approach_speed
