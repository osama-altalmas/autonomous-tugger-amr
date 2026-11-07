"""Height-aware box collision primitives; not a motion authorization.

Robot, cargo, and obstacles are represented as independently placed upright
oriented boxes. This module does not acquire obstacle data, validate swept
motion, update Nav2 costmaps, or certify a physical scene.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Box:
    name: str
    x: float
    y: float
    yaw: float
    half_x: float
    half_y: float
    z_min: float
    z_max: float

    def __post_init__(self):
        values = (self.x, self.y, self.yaw, self.half_x, self.half_y,
                  self.z_min, self.z_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Box geometry must be finite')
        if self.half_x <= 0 or self.half_y <= 0 or self.z_min >= self.z_max:
            raise ValueError('Box dimensions and height interval must be positive')


def boxes_overlap(first, second, *, margin=0.0):
    """Test conservative 2.5D overlap, including touching surfaces.

    ``margin`` is an extra clearance required between each pair of objects;
    it applies to vertical separation and each tested horizontal axis.
    """
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Margin must be finite and nonnegative')
    if first.z_max + margin < second.z_min or second.z_max + margin < first.z_min:
        return False
    # Broad phase: each box lies inside a circle of radius hypot(half_x,
    # half_y); disjoint circles mean disjoint boxes. Exact, and it skips the
    # trigonometry for the far-away pairs that make up most of a scene.
    reach = (math.hypot(first.half_x, first.half_y) +
             math.hypot(second.half_x, second.half_y) + margin)
    dx, dy = second.x - first.x, second.y - first.y
    if dx * dx + dy * dy > reach * reach:
        return False
    first_axes = ((math.cos(first.yaw), math.sin(first.yaw)),
                  (-math.sin(first.yaw), math.cos(first.yaw)))
    second_axes = ((math.cos(second.yaw), math.sin(second.yaw)),
                   (-math.sin(second.yaw), math.cos(second.yaw)))
    offset = (second.x - first.x, second.y - first.y)
    for axis in first_axes + second_axes:
        distance = abs(offset[0] * axis[0] + offset[1] * axis[1])
        first_radius = (first.half_x * abs(first_axes[0][0] * axis[0] + first_axes[0][1] * axis[1]) +
                        first.half_y * abs(first_axes[1][0] * axis[0] + first_axes[1][1] * axis[1]))
        second_radius = (second.half_x * abs(second_axes[0][0] * axis[0] + second_axes[0][1] * axis[1]) +
                         second.half_y * abs(second_axes[1][0] * axis[0] + second_axes[1][1] * axis[1]))
        if distance > first_radius + second_radius + margin:
            return False
    return True


def collision_pairs(robot_parts, obstacles, *, margin=0.0):
    """Return intersecting part/obstacle names for the supplied scene only."""
    return [(part.name, obstacle.name)
            for part in robot_parts for obstacle in obstacles
            if boxes_overlap(part, obstacle, margin=margin)]
