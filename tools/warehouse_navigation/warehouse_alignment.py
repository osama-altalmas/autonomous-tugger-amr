#!/usr/bin/env python3

import math


def normalize(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class WarehouseAlignment:
    """Rigid 2D transform from Gazebo warehouse coordinates to ROS map."""

    def __init__(self, world_pose, map_pose):
        wx, wy, wyaw = world_pose
        mx, my, myaw = map_pose

        self.rotation = normalize(myaw - wyaw)

        c = math.cos(self.rotation)
        s = math.sin(self.rotation)

        self.tx = mx - (c * wx - s * wy)
        self.ty = my - (s * wx + c * wy)

    def world_to_map(self, x, y, yaw):
        c = math.cos(self.rotation)
        s = math.sin(self.rotation)

        return (
            c * x - s * y + self.tx,
            s * x + c * y + self.ty,
            normalize(yaw + self.rotation),
        )

    def map_to_world(self, x, y, yaw):
        dx = x - self.tx
        dy = y - self.ty

        c = math.cos(self.rotation)
        s = math.sin(self.rotation)

        return (
            c * dx + s * dy,
            -s * dx + c * dy,
            normalize(yaw - self.rotation),
        )
