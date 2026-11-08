"""Conservative, simulator-only height-aware command guard primitives.

These checks use Gazebo collision ground truth; they are not safety-rated and
do not replace real-world perception. Unknown scene geometry fails closed.
"""

import json
import math
from collections import Counter
import xml.etree.ElementTree as ET

from tugger_control.lift_geometry import lift_is_raised
from tugger_control.precision_scene_clearance import FULL_LOCK_CURVATURE, stopping_curvature
from tugger_control.clearance_2_5d import Box, boxes_overlap
from tugger_control.warehouse_clearance import cargo_box, robot_boxes, static_world_boxes


REQUIRED_MODELS = frozenset(('north_wall', 'south_wall', 'east_wall',
                             'west_wall_north', 'west_wall_south',
                             'cargo_station_0_stand', 'cargo_station_3_stand',
                             'cargo_station_0_legs', 'cargo_station_3_legs',
                             'cargo_dropoff_northwest_legs', 'cargo_dropoff_southeast_legs'))


def scene_expected_counts(world_path):
    """Minimum collision count per known world model, including both payloads."""
    counts = Counter(box.name.split('/', 1)[0] for box in static_world_boxes(world_path))
    root = ET.parse(world_path).getroot()
    for station in (0, 3):
        name = f'cargo_station_{station}_stand'
        model = root.find(f"./world/model[@name='{name}']")
        if model is None:
            raise ValueError(f'Missing cargo model: {name}')
        count = len(model.findall('./link/collision'))
        if count < 1:
            raise ValueError(f'Missing cargo collision: {name}')
        counts[name] = count
    return dict(counts)


def decode_scene(payload, now, *, max_age=0.4, expected_counts=None):
    """Decode a complete fresh simulator scene, failing closed on bad data."""
    if not math.isfinite(now) or not math.isfinite(max_age) or max_age <= 0:
        raise ValueError('Invalid scene clock')
    data = json.loads(payload)
    if not isinstance(data, dict) or data.get('valid') is not True:
        raise ValueError('Simulator scene is incomplete or unsupported')
    stamp = data.get('stamp')
    if not isinstance(stamp, (float, int)) or not math.isfinite(stamp):
        raise ValueError('Simulator scene timestamp is invalid')
    if now - stamp < 0.0:
        raise ValueError(f'Simulator scene is from a future clock epoch '
                         f'({stamp - now:.3f} s ahead)')
    if now - stamp > max_age:
        raise ValueError(f'Simulator scene is stale ({now - stamp:.3f} s old)')
    raw_boxes = data.get('boxes')
    if not isinstance(raw_boxes, list) or len(raw_boxes) > 10000:
        raise ValueError('Simulator collision scene is invalid')
    names, models, boxes = set(), Counter(), []
    for raw in raw_boxes:
        if not isinstance(raw, dict):
            raise ValueError('Invalid simulator obstacle')
        name, model = raw.get('name'), raw.get('model')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Invalid or duplicate collision name')
        if not isinstance(model, str) or not model or model == 'autonomous_tugger':
            raise ValueError('Invalid collision model')
        try:
            x, y, z, hx, hy, hz = (float(raw[k]) for k in ('x', 'y', 'z', 'hx', 'hy', 'hz'))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid obstacle geometry') from exc
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError('Invalid obstacle height')
        boxes.append((model, Box(name, x, y, 0.0, hx, hy, z - hz, z + hz)))
        names.add(name)
        models[model] += 1
    if not REQUIRED_MODELS.issubset(models):
        raise ValueError('Simulator scene is missing required warehouse or cargo models')
    if expected_counts is not None and any(
            models[name] < count for name, count in expected_counts.items()):
        raise ValueError('Simulator scene is missing modeled collision geometry')
    return boxes


# Highest believable measured speed: 1.0 m/s top speed plus overshoot and
# odometry noise. Anything above means a runaway or a sensor fault: stop.
MEASURED_SPEED_LIMIT = 1.2


def loaded_envelope_clear(robot_pose, attached_cargo_pose, lift, cargo_id, obstacles,
                          current_speed, requested_speed, *, reaction_s=0.5,
                          deceleration=1.5, xy_clearance=0.02, z_clearance=0.01,
                          max_speed=1.0, curvature_bound=FULL_LOCK_CURVATURE,
                          curvature=None, steering_curvature=None,
                          max_current_speed=MEASURED_SPEED_LIMIT):
    """Check the robot's and cargo's stopping envelope.

    With ``curvature`` (the commanded path curvature, signed, 1/m) the robot
    and its cargo are swept along the arcs it can follow while stopping: from
    the commanded and the current steering curvature, each widened by
    STOPPING_CURVATURE_MARGIN, over the reaction and braking distance. Without
    it, a deliberately conservative all-directions inflation is used, which
    near rack corners refuses even creeping speeds (the loaded near-rack
    stall). Height is kept separate so cargo can pass above fixed legs.
    Returns ``(clear, collision_or_none)``.
    """
    numbers = (*robot_pose, *attached_cargo_pose, lift, current_speed, requested_speed,
               reaction_s, deceleration, xy_clearance, z_clearance)
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError('Non-finite motion guard input')
    if (cargo_id not in (0, 3) or reaction_s <= 0 or deceleration <= 0 or
            xy_clearance < 0 or z_clearance < 0 or max_speed <= 0 or
            # The measured speed is judged against what the robot can
            # physically do, not against the new ceiling: when the ceiling
            # drops (route -> station -> camera) or speed overshoots by a few
            # mm/s, the robot is still above it for a moment. The envelope
            # below already uses the higher of measured and requested speed.
            abs(current_speed) > max_current_speed or abs(requested_speed) > max_speed or
            not lift_is_raised(lift)):
        raise ValueError('Loaded motion is outside guarded operating limits')
    if math.hypot(attached_cargo_pose[0] - robot_pose[0],
                  attached_cargo_pose[1] - robot_pose[1]) > 0.04:
        raise ValueError('Cargo has shifted outside the attachment envelope')
    speed = max(abs(current_speed), abs(requested_speed))
    travel = speed * reaction_s + speed * speed / (2.0 * deceleration)
    if curvature is not None and math.isfinite(curvature):
        return _arc_envelope_clear(robot_pose, attached_cargo_pose, lift, cargo_id, obstacles,
                                   current_speed, requested_speed, travel, curvature,
                                   steering_curvature, xy_clearance, z_clearance)
    # Counter-phase 4WS at 45 degrees has |curvature| <= 2 / wheelbase.
    # A square corner can sweep this far as its heading changes.
    angular_change = min(math.pi, stopping_curvature(curvature_bound) * travel)
    rotation = 2.0 * math.hypot(0.68, 0.68) * math.sin(angular_change / 2.0)
    horizontal_margin = xy_clearance + travel + rotation
    parts = robot_boxes(robot_pose, lift) + [cargo_box('cargo', attached_cargo_pose)]
    selected = f'cargo_station_{cargo_id}_stand'
    for model, obstacle in obstacles:
        if model == selected:
            continue
        inflated = Box(obstacle.name, obstacle.x, obstacle.y, obstacle.yaw,
                       obstacle.half_x + horizontal_margin,
                       obstacle.half_y + horizontal_margin,
                       obstacle.z_min - z_clearance, obstacle.z_max + z_clearance)
        for part in parts:
            if boxes_overlap(part, inflated):
                return False, (part.name, model, obstacle.name)
    return True, None


# Steering uncertainty around the commanded and current curvature (1/m).
STOPPING_CURVATURE_MARGIN = 0.3
ARC_SAMPLE_M = 0.02


def _arc_envelope_clear(robot_pose, cargo_pose, lift, cargo_id, obstacles, current_speed,
                        requested_speed, travel, curvature, steering_curvature,
                        xy_clearance, z_clearance):
    """Sweep robot and cargo along the arcs reachable while stopping."""
    direction = requested_speed if abs(requested_speed) > 1e-9 else current_speed
    sign = 1.0 if direction >= 0 else -1.0
    known = [curvature] + ([steering_curvature] if steering_curvature is not None and
                           math.isfinite(steering_curvature) else [])
    low = max(-FULL_LOCK_CURVATURE, min(known) - STOPPING_CURVATURE_MARGIN)
    high = min(FULL_LOCK_CURVATURE, max(known) + STOPPING_CURVATURE_MARGIN)
    arcs = sorted({low, high, *[max(low, min(high, k)) for k in known]})
    selected = f'cargo_station_{cargo_id}_stand'
    margin = xy_clearance + ARC_SAMPLE_M
    reach = travel + math.hypot(0.68, 0.68) + margin + 0.1
    near = []
    for model, obstacle in obstacles:
        if model == selected:
            continue
        if math.hypot(obstacle.x - robot_pose[0], obstacle.y - robot_pose[1]) > reach + math.hypot(
                obstacle.half_x, obstacle.half_y):
            continue
        near.append((model, Box(obstacle.name, obstacle.x, obstacle.y, obstacle.yaw,
                                obstacle.half_x + margin, obstacle.half_y + margin,
                                obstacle.z_min - z_clearance, obstacle.z_max + z_clearance)))
    x0, y0, yaw0 = robot_pose
    dx, dy = cargo_pose[0] - x0, cargo_pose[1] - y0
    count = max(1, math.ceil(travel / ARC_SAMPLE_M))
    for kappa in arcs:
        for index in range(count + 1):
            s = sign * travel * index / count
            turn = kappa * s
            if abs(kappa) < 1e-9:
                fx, fy = s, 0.0
            else:
                fx, fy = math.sin(turn) / kappa, (1.0 - math.cos(turn)) / kappa
            c, n = math.cos(yaw0), math.sin(yaw0)
            pose = (x0 + c * fx - n * fy, y0 + n * fx + c * fy, yaw0 + turn)
            ct, st = math.cos(turn), math.sin(turn)
            cargo = (pose[0] + ct * dx - st * dy, pose[1] + st * dx + ct * dy,
                     cargo_pose[2], cargo_pose[3] + turn)
            parts = robot_boxes(pose, lift) + [cargo_box('cargo', cargo)]
            for model, obstacle in near:
                for part in parts:
                    if boxes_overlap(part, obstacle):
                        return False, (part.name, model, obstacle.name)
    return True, None


def choose_guarded_speed(robot_pose, attached_cargo_pose, lift, cargo_id, obstacles,
                         current_speed, requested_speed, *, max_speed=1.0,
                         curvature_bound=FULL_LOCK_CURVATURE, curvature=None,
                         steering_curvature=None):
    """Reduce loaded speed near tight geometry; return zero if none is safe."""
    if not math.isfinite(requested_speed):
        raise ValueError('Invalid requested velocity')
    if not math.isfinite(max_speed) or max_speed <= 0:
        raise ValueError('Invalid loaded speed limit')
    limited = max(-max_speed, min(max_speed, requested_speed))
    magnitude = abs(limited)
    candidates = sorted({magnitude, *(min(magnitude, v) for v in
                                      (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)), 0.0},
                        reverse=True)
    last_hit = None
    for candidate in candidates:
        value = math.copysign(candidate, limited) if candidate else 0.0
        clear, last_hit = loaded_envelope_clear(
            robot_pose, attached_cargo_pose, lift, cargo_id, obstacles,
            current_speed, value, max_speed=max_speed,
            curvature_bound=curvature_bound, curvature=curvature,
            steering_curvature=steering_curvature)
        if clear:
            return value, None
    return 0.0, last_hit
