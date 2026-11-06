"""Simulation-only sampled checks against modeled warehouse box collisions.

The checks require fresh simulator poses for both movable cargo pieces. They do
not discover unmodeled obstacles or replace Nav2's sensor collision checking.
"""

import math
import xml.etree.ElementTree as ET

from tugger_control.clearance_2_5d import Box, boxes_overlap, collision_pairs
from tugger_control.lift_geometry import LIFT_TOLERANCE_M, LIFT_UP_M, cargo_lift_offset


def validate_modeled_geometry(world_path, robot_path):
    """Reject model changes that would invalidate the configured bounds."""
    robot = ET.parse(robot_path).getroot()
    properties = {item.get('name'): item.get('value')
                  for item in robot.findall('./{*}property')}
    expected_properties = {'wheelbase': 0.58, 'wheel_y': 0.22,
                           'wheel_radius': 0.055, 'wheel_width': 0.04}
    for name, expected in expected_properties.items():
        if not math.isclose(float(properties.get(name, 'nan')), expected, abs_tol=1e-9):
            raise ValueError(f'Robot geometry changed: {name}')
    if len(robot.findall('.//collision')) != 3:
        raise ValueError('Robot collision inventory changed')
    wheel_macro = next((item for item in robot.findall('./{*}macro')
                        if item.get('name') == 'steered_wheel'), None)
    if wheel_macro is None:
        raise ValueError('Missing wheel geometry macro')
    wheel_origin = wheel_macro.find("./joint[@name='${prefix}_steering_joint']/origin")
    wheel_collision = wheel_macro.find("./link[@name='${prefix}_wheel_link']/collision")
    wheel_shape = wheel_collision.find('geometry/cylinder') if wheel_collision is not None else None
    wheel_offset = wheel_collision.find('origin') if wheel_collision is not None else None
    if (wheel_origin is None or wheel_origin.get('xyz') != '${x} ${y} -0.045' or
            wheel_origin.get('rpy') != '0 0 0' or wheel_shape is None or
            wheel_shape.get('radius') != '${wheel_radius}' or
            wheel_shape.get('length') != '${wheel_width}' or wheel_offset is None or
            wheel_offset.get('xyz') != '0 0 0' or wheel_offset.get('rpy') != '1.5708 0 0'):
        raise ValueError('Wheel collision geometry changed')
    wheel_calls = {item.get('prefix'): (item.get('x'), item.get('y'))
                   for item in robot.findall('./{*}steered_wheel')}
    expected_wheels = {
        'front_left': ('${wheelbase/2}', '${wheel_y}'),
        'front_right': ('${wheelbase/2}', '${-wheel_y}'),
        'rear_left': ('${-wheelbase/2}', '${wheel_y}'),
        'rear_right': ('${-wheelbase/2}', '${-wheel_y}'),
    }
    if wheel_calls != expected_wheels or len(robot.findall('./{*}steered_wheel')) != 4:
        raise ValueError('Wheel placement changed')
    for link, size, origin in (
            ('base_link', (0.88, 0.58, 0.15), (0.0, 0.0, 0.0475)),
            ('lift_platform_link', (0.5, 0.32, 0.015), (0.0, 0.0, 0.0))):
        collision = robot.find(f"./link[@name='{link}']/collision")
        if collision is None:
            raise ValueError(f'Missing collision: {link}')
        box = collision.find('geometry/box')
        values = tuple(float(v) for v in box.get('size', '').split()) if box is not None else ()
        origin_node = collision.find('origin')
        xyz = tuple(float(v) for v in origin_node.get('xyz', '0 0 0').split()) if origin_node is not None else (0.0, 0.0, 0.0)
        rpy = tuple(float(v) for v in origin_node.get('rpy', '0 0 0').split()) if origin_node is not None else (0.0, 0.0, 0.0)
        if values != size or xyz != origin or rpy != (0.0, 0.0, 0.0):
            raise ValueError(f'Robot collision geometry changed: {link}')
    for joint, expected in (('base_footprint_joint', (0.0, 0.0, 0.1)),
                            ('lift_joint', (0.0, 0.0, 0.13))):
        node = robot.find(f"./joint[@name='{joint}']/origin")
        if node is None or tuple(float(v) for v in node.get('xyz', '').split()) != expected:
            raise ValueError(f'Robot joint geometry changed: {joint}')
        if node.get('rpy', '0 0 0') != '0 0 0':
            raise ValueError(f'Robot joint rotation changed: {joint}')
    lift_axis = robot.find("./joint[@name='lift_joint']/axis")
    if lift_axis is None or lift_axis.get('xyz') != '0 0 1':
        raise ValueError('Lift joint axis changed')
    world = ET.parse(world_path).getroot()
    for station in (0, 3):
        collision = world.find(
            f"./world/model[@name='cargo_station_{station}_stand']/link/collision[@name='tabletop_collision']")
        if collision is None or collision.findtext('geometry/box/size') != '1.3 1.3 0.03':
            raise ValueError(f'Cargo collision geometry changed: station {station}')
        if _pose(collision) != [0.0, 0.0, 0.295, 0.0, 0.0, 0.0]:
            raise ValueError(f'Cargo collision pose changed: station {station}')
        model = world.find(f"./world/model[@name='cargo_station_{station}_stand']")
        link = model.find('link') if model is not None else None
        if link is None or _pose(link) != [0.0] * 6 or len(link.findall('collision')) != 1:
            raise ValueError(f'Cargo link geometry changed: station {station}')


def _pose(element):
    node = element.find('pose')
    if node is not None and node.get('relative_to'):
        raise ValueError('Relative SDF collision poses are not supported')
    values = [float(value) for value in node.text.split()] if node is not None else [0.0] * 6
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError('Invalid SDF pose')
    if abs(values[3]) > 1e-8 or abs(values[4]) > 1e-8:
        raise ValueError('Non-planar SDF pose requires a full 3D checker')
    return values


def _compose(first, second):
    c, s = math.cos(first[5]), math.sin(first[5])
    return [first[0] + c * second[0] - s * second[1],
            first[1] + s * second[0] + c * second[1], first[2] + second[2],
            0.0, 0.0, first[5] + second[5]]


def static_world_boxes(path):
    """Parse all static SDF box collisions, refusing unsupported geometry."""
    root = ET.parse(path).getroot()
    world = root.find('world')
    if world is None or world.findall('include') or world.findall('model/model'):
        raise ValueError('Unsupported world structure')
    boxes = []
    for model in world.findall('model'):
        name = model.get('name')
        if model.findtext('static', 'false').strip().lower() != 'true':
            if name not in ('cargo_station_0_stand', 'cargo_station_3_stand'):
                raise ValueError(f'Unknown dynamic model: {name}')
            continue
        model_pose = _pose(model)
        for link in model.findall('link'):
            link_pose = _compose(model_pose, _pose(link))
            for collision in link.findall('collision'):
                geometry = collision.find('geometry')
                box = geometry.find('box/size') if geometry is not None else None
                if box is None:
                    if name == 'ground_plane' and geometry is not None and geometry.find('plane') is not None:
                        continue
                    raise ValueError(f'Unsupported collision geometry: {name}')
                size = [float(value) for value in box.text.split()]
                if len(size) != 3 or not all(math.isfinite(value) and value > 0 for value in size):
                    raise ValueError(f'Invalid collision dimensions: {name}')
                pose = _compose(link_pose, _pose(collision))
                z_min, z_max = pose[2] - size[2] / 2, pose[2] + size[2] / 2
                if z_max <= 1e-8:
                    continue
                boxes.append(Box(f"{name}/{collision.get('name', 'collision')}",
                                 pose[0], pose[1], pose[5], size[0] / 2, size[1] / 2,
                                 z_min, z_max))
    return boxes


def robot_boxes(pose, lift):
    """Conservative chassis/lift/wheel boxes for the current 4WS model."""
    x, y, yaw = pose
    if (not all(math.isfinite(value) for value in (x, y, yaw, lift)) or
            not -LIFT_TOLERANCE_M <= lift <= LIFT_UP_M + LIFT_TOLERANCE_M):
        raise ValueError('Invalid robot pose or lift position')
    boxes = [Box('chassis', x, y, yaw, 0.44, 0.29, 0.0725, 0.2225),
             Box('lift', x, y, yaw, 0.25, 0.16, 0.2225 + lift, 0.2375 + lift)]
    c, s = math.cos(yaw), math.sin(yaw)
    for dx in (-0.29, 0.29):
        for dy in (-0.22, 0.22):
            boxes.append(Box('wheel_bound', x + c * dx - s * dy, y + s * dx + c * dy,
                             0, 0.075, 0.075, 0.0, 0.11))
    return boxes


def cargo_box(name, pose, extra_xy=0.03):
    """Cargo is a 1.3 m square, 30 mm thick, with 30 mm capture allowance."""
    x, y, z, yaw = pose
    return Box(name, x, y, yaw, 0.65 + extra_xy, 0.65 + extra_xy,
               z + 0.28, z + 0.31)


# Same band as the lift's "down" check. A measured cargo sits a little below the
# commanded lift height (0.7 mm in Gazebo), so landing on the support legs must
# not be reported as penetration.
SUPPORT_CONTACT_TOLERANCE_M = 0.005


def lift_sweep_clear(robot_pose, cargo_pose, other_cargo_pose, static_boxes,
                     lift_position, target_lift, *, margin=0.01):
    """Sample lift travel and allow support-leg contact without penetration.

    All poses are fresh world-frame observations. This checks only the modeled
    box scene; it does not validate unmodeled dynamic or overhead obstacles.
    """
    if not all(math.isfinite(value) for value in (*robot_pose, *cargo_pose,
                                                   *other_cargo_pose, lift_position,
                                                   target_lift, margin)) or margin < 0:
        raise ValueError('Invalid lift-clearance input')
    other = cargo_box('other_cargo', other_cargo_pose)
    obstacles = list(static_boxes) + [other]
    # The cargo only moves while the lift is above contact height; below it the
    # cargo rests on its support and the platform travels on its own.
    resting_z = cargo_pose[2] - cargo_lift_offset(lift_position)
    steps = max(1, math.ceil(abs(target_lift - lift_position) / 0.005))
    for index in range(steps + 1):
        lift = lift_position + (target_lift - lift_position) * index / steps
        parts = robot_boxes(robot_pose, lift)
        hits = collision_pairs(parts, obstacles, margin=margin)
        if hits:
            return False, hits[0]
        moved = (cargo_pose[0], cargo_pose[1], resting_z + cargo_lift_offset(lift),
                 cargo_pose[3])
        cargo = cargo_box('cargo', moved)
        for obstacle in obstacles:
            if (('_legs/' in obstacle.name or '_legs::' in obstacle.name) and
                    cargo.z_min >= obstacle.z_max - SUPPORT_CONTACT_TOLERANCE_M):
                continue  # Intended support contact, not penetration.
            if boxes_overlap(cargo, obstacle, margin=margin):
                return False, (cargo.name, obstacle.name)
    return True, None


def loaded_path_clear(poses, initial_robot_pose, attached_cargo_pose,
                      other_cargo_pose, static_boxes, lift_position, *, margin=0.01):
    """Sample the known-world path with independent robot and cargo envelopes.

    The carried cargo's relative planar pose is held fixed. Unknown or newly
    spawned obstacles are not represented; callers must not interpret a pass
    as complete runtime collision protection.
    """
    points = [tuple(float(v) for v in pose) for pose in poses]
    if len(points) < 2 or any(len(p) != 3 or not all(math.isfinite(v) for v in p)
                              for p in points):
        raise ValueError('A path needs at least two finite x, y, yaw poses')
    if len(initial_robot_pose) != 3 or len(attached_cargo_pose) != 4:
        raise ValueError('Invalid robot or cargo pose')
    rx, ry, heading = initial_robot_pose
    dx, dy = attached_cargo_pose[0] - rx, attached_cargo_pose[1] - ry
    c, s = math.cos(heading), math.sin(heading)
    offset_x, offset_y = c * dx + s * dy, -s * dx + c * dy
    relative_yaw = attached_cargo_pose[3] - heading
    if math.hypot(offset_x, offset_y) > 0.04:
        raise ValueError('Attached cargo offset exceeds the modeled capture envelope')
    obstacles = list(static_boxes) + [cargo_box('other_cargo', other_cargo_pose)]
    samples = []
    for start, end in zip(points, points[1:]):
        rotation = math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2]))
        count = max(1, math.ceil(math.hypot(end[0] - start[0], end[1] - start[1]) / 0.01),
                    math.ceil(abs(rotation) / 0.02))
        for index in range(count):
            ratio = index / count
            samples.append((start[0] + (end[0] - start[0]) * ratio,
                            start[1] + (end[1] - start[1]) * ratio,
                            start[2] + rotation * ratio))
    samples.append(points[-1])
    for index, pose in enumerate(samples):
        hits = collision_pairs(robot_boxes(pose, lift_position), obstacles, margin=margin)
        if hits:
            return False, index, hits[0]
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        moved = (x + c * offset_x - s * offset_y,
                 y + s * offset_x + c * offset_y,
                 attached_cargo_pose[2], yaw + relative_yaw)
        cargo = cargo_box('cargo', moved)
        for obstacle in obstacles:
            if (('_legs/' in obstacle.name or '_legs::' in obstacle.name) and
                    cargo.z_min >= obstacle.z_max + margin):
                continue
            if boxes_overlap(cargo, obstacle, margin=margin):
                return False, index, ('cargo', obstacle.name)
    return True, len(samples), None
