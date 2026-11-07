"""Clearance of a turn in place against the height-aware collision scene.

The robot's chassis, lift and wheel bounds (and, when loaded, the attached
1.3 m cargo square, which turns with the robot about its centre) are sampled
every 2 degrees through the turn and checked against every scene box except
the attached cargo itself. Heights are kept, so a raised cargo passes above
fixed stand legs while the chassis between the legs must clear them.
Sampled, not continuous; the sample step bounds the miss to under 2 cm at
the cargo corners, which the margin covers. No robot commands.
"""

import math

from tugger_control.clearance_2_5d import Box, collision_pairs
from tugger_control.turn_in_place import rotation_sweep, wrap
from tugger_control.warehouse_clearance import cargo_box, robot_boxes


def _turned_cargo(robot_pose, cargo_pose, turn):
    """The attached cargo pose after the robot turns ``turn`` about its centre."""
    dx, dy = cargo_pose[0] - robot_pose[0], cargo_pose[1] - robot_pose[1]
    c, s = math.cos(turn), math.sin(turn)
    return (robot_pose[0] + c * dx - s * dy, robot_pose[1] + s * dx + c * dy,
            cargo_pose[2], wrap(cargo_pose[3] + turn))


# Vertical clearance, as for driving under a stand (precision_scene_clearance).
Z_MARGIN_M = 0.01


def rotation_clear(robot_pose, lift, obstacles, angle, *, cargo_id=None, cargo_pose=None,
                   margin=0.02, step=math.radians(2.0)):
    """Return ``(clear, (sample_yaw, collision) or None)`` for a turn by ``angle``.

    ``obstacles`` is the decoded scene: (model, Box) pairs. ``angle`` may be
    2*pi for the whole circle. With ``cargo_id`` the attached cargo turns too.
    ``margin`` is horizontal clearance around the turning parts; vertically
    Z_MARGIN_M applies, so an unloaded robot may turn under a stand it fits
    beneath, exactly as it drives under it.
    """
    if not all(math.isfinite(v) for v in (*robot_pose, lift, angle, margin)) or margin < 0:
        raise ValueError('invalid turn-in-place clearance input')
    attached = None
    if cargo_id is not None:
        if cargo_pose is None or len(cargo_pose) != 4:
            raise ValueError('a loaded turn needs the attached cargo pose')
        if math.hypot(cargo_pose[0] - robot_pose[0], cargo_pose[1] - robot_pose[1]) > 0.04:
            raise ValueError('cargo has shifted outside the attachment envelope')
        attached = f'cargo_station_{cargo_id}_stand'
    boxes = [Box(box.name, box.x, box.y, box.yaw, box.half_x + margin, box.half_y + margin,
                 box.z_min, box.z_max)
             for model, box in obstacles if model != attached]
    if not boxes:
        raise ValueError('a nonempty collision scene is required')
    # Only boxes within reach of the widest turning part can be touched.
    reach = (math.hypot(0.68, 0.68) if attached else math.hypot(0.44, 0.29)) + margin + 0.1
    near = [box for box in boxes
            if math.hypot(box.x - robot_pose[0], box.y - robot_pose[1]) <=
            reach + math.hypot(box.half_x, box.half_y)]
    for pose in rotation_sweep(robot_pose, angle, step):
        parts = robot_boxes(pose, lift)
        if attached:
            parts.append(cargo_box('cargo', _turned_cargo(
                robot_pose, cargo_pose, wrap(pose[2] - robot_pose[2]))))
        hits = collision_pairs(parts, near, margin=Z_MARGIN_M)
        if hits:
            return False, (round(pose[2], 3), hits[0])
    return True, None


def stopping_turn(yaw_rate, deceleration, *, reaction_s=0.3, margin_rad=0.05):
    """Angle turned before a turn in place stops, including reaction time."""
    rate = abs(yaw_rate)
    return rate * reaction_s + rate * rate / (2.0 * deceleration) + margin_rad
