#!/usr/bin/env python3

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from action_msgs.msg import GoalStatus

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from nav2_msgs.action import FollowPath
from tf2_ros import Buffer, TransformListener, TransformException
from warehouse_alignment import WarehouseAlignment
from check_warehouse_alignment import gazebo_pose



def normalize(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(q):
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z)
    )


class WarehouseRouteExecutor(Node):

    def __init__(self, args):
        super().__init__(
            "warehouse_route_executor",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ]
        )

        self.args = args

        self.publisher = self.create_publisher(
            NavPath, "/warehouse/planned_path", 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.client = ActionClient(
            self, FollowPath, "/follow_path"
        )

        data = json.loads(Path(args.route).read_text())
        route = data.get("selected_route")

        if not route:
            raise RuntimeError("No selected_route in route JSON")

        self.direction = route.get(
            "direction",
            data.get("driving_direction", "forward")
        )

        self.route_poses = route["poses"]

        if data.get("coordinate_frame") == "warehouse_world":
            if args.frame != "map":
                raise RuntimeError(
                    "Warehouse route conversion currently requires --frame map"
                )

            self.warehouse_route_poses = self.route_poses
            self.route_poses = None
        else:
            self.warehouse_route_poses = None

        poses_to_check = (
            self.warehouse_route_poses
            if self.warehouse_route_poses is not None
            else self.route_poses
        )

        if len(poses_to_check) < 2:
            raise RuntimeError("Route requires at least two poses")

        if self.warehouse_route_poses is not None:
            self.initialize_alignment()

        self.path = NavPath()
        self.path.header.frame_id = args.frame

        for x, y, yaw in self.route_poses:
            pose = PoseStamped()
            pose.header.frame_id = args.frame
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)

            pose.pose.orientation.z = math.sin(float(yaw) / 2)
            pose.pose.orientation.w = math.cos(float(yaw) / 2)

            self.path.poses.append(pose)

        self.timer = self.create_timer(
            0.5, self.publish_path
        )

    def initialize_alignment(self):
        if not self.wait_for_tf():
            raise RuntimeError("Cannot align warehouse route without TF")

        map_pose = self.get_robot_pose()
        world_pose = gazebo_pose()
        alignment = WarehouseAlignment(world_pose, map_pose)

        self.route_poses = [
            alignment.world_to_map(float(x), float(y), float(yaw))
            for x, y, yaw in self.warehouse_route_poses
        ]

        print(
            "Warehouse route converted to map coordinates: "
            f"rotation={math.degrees(alignment.rotation):.3f} deg, "
            f"translation=({alignment.tx:.3f}, {alignment.ty:.3f})"
        )

    def publish_path(self):
        stamp = self.get_clock().now().to_msg()
        self.path.header.stamp = stamp

        for pose in self.path.poses:
            pose.header.stamp = stamp

        self.publisher.publish(self.path)

    def get_robot_pose(self):
        tf = self.tf_buffer.lookup_transform(
            self.args.frame,
            self.args.base_frame,
            rclpy.time.Time()
        )

        p = tf.transform.translation
        q = tf.transform.rotation

        return p.x, p.y, quaternion_to_yaw(q)

    def wait_for_tf(self):
        print("\n========== TF INITIALIZATION ==========")
        print("Waiting for stable map -> base_link TF...")

        deadline = time.monotonic() + 60.0
        consecutive = 0
        last_error = None

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)

            try:
                self.get_robot_pose()
                consecutive += 1

                if consecutive >= 5:
                    print(
                        "TF READY: 5 consecutive successful lookups"
                    )
                    return True

            except TransformException as exc:
                consecutive = 0
                error = str(exc)

                if error != last_error:
                    print("Waiting:", error)
                    last_error = error

            time.sleep(0.2)

        print("ERROR: TF initialization timed out.")
        return False

    def check_start(self):
        try:
            x, y, yaw = self.get_robot_pose()
        except TransformException as exc:
            print("TF unavailable:", exc)
            return False

        sx, sy, syaw = self.route_poses[0]

        distance = math.hypot(x - sx, y - sy)
        heading_error = abs(normalize(yaw - syaw))

        print("\n========== START ALIGNMENT ==========")
        print(
            f"Robot: x={x:.3f}, y={y:.3f}, "
            f"yaw={math.degrees(yaw):.1f} deg"
        )
        print(
            f"Route: x={sx:.3f}, y={sy:.3f}, "
            f"yaw={math.degrees(syaw):.1f} deg"
        )
        print(f"Position error: {distance:.3f} m")
        print(
            f"Heading error: "
            f"{math.degrees(heading_error):.1f} deg"
        )

        aligned = (
            distance <= self.args.position_tolerance
            and heading_error <= math.radians(
                self.args.heading_tolerance
            )
        )

        print("Alignment:", "PASS" if aligned else "FAIL")
        return aligned

    def execute(self):
        print("\nWaiting for Nav2 FollowPath action...")

        if not self.client.wait_for_server(timeout_sec=20.0):
            print("ERROR: FollowPath action unavailable.")
            return

        if not self.check_start():
            print("ERROR: Robot not aligned. No goal sent.")
            return

        goal = FollowPath.Goal()
        goal.path = self.path
        goal.controller_id = ""
        goal.goal_checker_id = ""

        # Use a current simulation timestamp.
        self.publish_path()

        print("\nSending FollowPath goal...")

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        handle = send_future.result()

        if handle is None or not handle.accepted:
            print("ERROR: FollowPath goal rejected.")
            return

        print("FollowPath goal ACCEPTED.")

        result_future = handle.get_result_async()

        try:
            while rclpy.ok() and not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.2)

            if result_future.done():
                result = result_future.result()
                status = result.status

                print("\n========== NAVIGATION RESULT ==========")
                print("Status code:", status)
                print(
                    "Status:",
                    "SUCCEEDED"
                    if status == GoalStatus.STATUS_SUCCEEDED
                    else "NOT SUCCEEDED"
                )
                print("Result:", result.result)

        except KeyboardInterrupt:
            print("\nCancelling FollowPath goal...")
            cancel_future = handle.cancel_goal_async()

            deadline = time.monotonic() + 5.0

            while (
                rclpy.ok()
                and not cancel_future.done()
                and time.monotonic() < deadline
            ):
                rclpy.spin_once(self, timeout_sec=0.1)

            print("Cancellation requested.")
            raise


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--route", required=True)
    parser.add_argument("--frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--execute", action="store_true")

    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--heading-tolerance",
        type=float,
        default=15.0
    )

    args = parser.parse_args()

    rclpy.init()
    node = None

    try:
        node = WarehouseRouteExecutor(args)

        print("\n========== WAREHOUSE ROUTE ==========")
        print("Route:", args.route)
        print("Frame:", args.frame)
        print("Direction:", node.direction)
        print("Number of poses:", len(node.route_poses))
        print("Execution requested:", args.execute)

        if not node.wait_for_tf():
            print("No motion command was sent.")
            return

        if not node.check_start():
            print("Robot not aligned. No motion command sent.")
            return

        if args.execute:
            node.execute()
        else:
            print("\nDRY RUN — route publishing only.")
            print("Press Ctrl+C to exit.")

            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.2)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
