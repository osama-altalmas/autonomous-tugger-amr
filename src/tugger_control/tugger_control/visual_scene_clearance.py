"""Exact static-scene broad phase for local visual alignment (simulation only).

Candidate obstacles are culled with conservative horizontal bounding circles and
vertical intervals; survivors use the *unchanged* 2.5D separating-axis test.
This does not add dynamic perception or replace the runtime motion guard.
"""

import math
from tugger_control.clearance_2_5d import boxes_overlap
from tugger_control.warehouse_clearance import cargo_box, robot_boxes


class PreparedVisualClearance:
    """Robot (and carried cargo, if any) against the scene at sampled poses.

    ``carried`` is the attached cargo in the robot frame as
    ``(forward, left, height, relative_yaw)``; it moves with the robot.
    """

    def __init__(self, obstacles, lift, *, margin=0.01, carried=None):
        self.lift = float(lift)
        self.carried = carried
        self.margin = float(margin)
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError('invalid clearance margin')
        # Materialize once. Changing the live scene requires a NEW instance.
        self.obstacles = tuple(obstacles)
        self.bounds = tuple((box, math.hypot(box.half_x, box.half_y))
                            for box in self.obstacles)

    def _parts(self, pose):
        parts = list(robot_boxes(pose, self.lift))
        if self.carried is not None:
            forward, left, height, relative_yaw = self.carried
            c, s = math.cos(pose[2]), math.sin(pose[2])
            parts.append(cargo_box('cargo', (pose[0] + c * forward - s * left,
                                             pose[1] + s * forward + c * left,
                                             height, pose[2] + relative_yaw)))
        return parts

    def __call__(self, pose):
        for part in self._parts(pose):
            radius = math.hypot(part.half_x, part.half_y)
            for obstacle, other_radius in self.bounds:
                if (part.z_max + self.margin < obstacle.z_min or
                        obstacle.z_max + self.margin < part.z_min):
                    continue
                reach = radius + other_radius + self.margin
                if abs(part.x - obstacle.x) > reach or abs(part.y - obstacle.y) > reach:
                    continue
                if boxes_overlap(part, obstacle, margin=self.margin):
                    return False
        return True
