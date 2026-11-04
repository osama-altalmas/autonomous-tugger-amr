"""Exact static-scene broad phase for local visual alignment (simulation only).

Candidate obstacles are culled with conservative horizontal bounding circles and
vertical intervals; survivors use the *unchanged* 2.5D separating-axis test.
This does not add dynamic perception or replace the runtime motion guard.
"""

import math
from tugger_control.clearance_2_5d import boxes_overlap
from tugger_control.warehouse_clearance import robot_boxes


class PreparedVisualClearance:
    def __init__(self, obstacles, lift, *, margin=0.01):
        self.lift = float(lift)
        self.margin = float(margin)
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError('invalid clearance margin')
        # Materialize once. Changing the live scene requires a NEW instance.
        self.obstacles = tuple(obstacles)
        self.bounds = tuple((box, math.hypot(box.half_x, box.half_y))
                            for box in self.obstacles)

    def __call__(self, pose):
        for part in robot_boxes(pose, self.lift):
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
