#!/usr/bin/env python3

import math
import re
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import Buffer, TransformListener, TransformException

from warehouse_alignment import WarehouseAlignment


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def gazebo_pose():
    result = subprocess.run(
        ["gz", "model", "-m", "autonomous_tugger", "-p"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True
    )

    output = result.stdout + result.stderr

    pattern = (
        r"Pose \[ XYZ \(m\) \] \[ RPY \(rad\) \]:"
        r"\s*\[([^\]]+)\]\s*\[([^\]]+)\]"
    )

    match = re.search(pattern, output)

    if not match:
        raise RuntimeError(
            "Could not parse Gazebo robot pose:\n" + output
        )

    xyz = [float(v) for v in match.group(1).split()]
    rpy = [float(v) for v in match.group(2).split()]

    return xyz[0], xyz[1], rpy[2]


class AlignmentChecker(Node):

    def __init__(self):
        super().__init__(
            "warehouse_alignment_checker",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ]
        )

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

    def map_pose(self):
        transform = self.buffer.lookup_transform(
            "map", "base_link", rclpy.time.Time()
        )

        p = transform.transform.translation
        q = transform.transform.rotation

        return p.x, p.y, yaw_from_quaternion(q)


def main():
    rclpy.init()
    node = AlignmentChecker()

    try:
        deadline = time.monotonic() + 30.0

        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)

            try:
                map_pose = node.map_pose()
                break
            except TransformException:
                pass
        else:
            raise RuntimeError("map -> base_link TF unavailable")

        world_pose = gazebo_pose()

        alignment = WarehouseAlignment(world_pose, map_pose)

        print("\n========== WAREHOUSE ALIGNMENT ==========")
        print("Gazebo pose:", world_pose)
        print("ROS map pose:", map_pose)
        print(
            "World-to-map rotation:",
            math.degrees(alignment.rotation),
            "degrees"
        )
        print(
            "World-to-map translation:",
            alignment.tx,
            alignment.ty
        )

        converted = alignment.world_to_map(*world_pose)
        print("Converted robot pose:", converted)
        print("========================================\n")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
