#!/usr/bin/env python3
"""Phase 14C: read-only predocking checkpoint verification."""

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


STEERING_JOINTS = (
    "front_left_steering_joint",
    "front_right_steering_joint",
    "rear_left_steering_joint",
    "rear_right_steering_joint",
)


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class PredockMonitor(Node):
    def __init__(self, target):
        super().__init__("phase14_predock_monitor")
        # ROS 2 Lyrical may declare use_sim_time automatically.
        # Set the existing parameter instead of declaring it twice.
        from rclpy.parameter import Parameter

        self.set_parameters([
            Parameter(
                "use_sim_time",
                Parameter.Type.BOOL,
                True,
            )
        ])

        if not self.get_parameter("use_sim_time").value:
            raise RuntimeError(
                "Phase 14C requires Gazebo simulation time"
            )

        self.target = target
        self.odom = None
        self.joints = None
        self.odom_received = None
        self.joints_received = None

        self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self.on_odom,
            10,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joints,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self
        )

    def on_odom(self, msg):
        self.odom = msg
        self.odom_received = time.monotonic()

    def on_joints(self, msg):
        self.joints = msg
        self.joints_received = time.monotonic()

    def inspect(self):
        now = time.monotonic()

        if self.odom is None:
            return False, "WAITING FOR /odometry/filtered"

        if self.joints is None:
            return False, "WAITING FOR /joint_states"

        if now - self.odom_received > 1.0:
            return False, "STALE ODOMETRY"

        if now - self.joints_received > 1.0:
            return False, "STALE JOINT STATES"

        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_footprint",
                Time(),
                timeout=Duration(seconds=0.15),
            )
        except Exception as exc:
            return False, f"MAP TF UNAVAILABLE: {exc}"

        # Ensure the transform timestamp is reasonably recent
        # relative to the current simulation time.
        stamp = transform.header.stamp
        stamp_ns = stamp.sec * 10**9 + stamp.nanosec
        now_ns = self.get_clock().now().nanoseconds

        if now_ns <= 0 or stamp_ns <= 0:
            return False, "SIMULATION CLOCK / TF NOT READY"

        tf_age = (now_ns - stamp_ns) / 1e9

        if tf_age < -0.1 or tf_age > 0.5:
            return False, f"STALE MAP TF: {tf_age:.3f} s"

        position = transform.transform.translation
        orientation = transform.transform.rotation

        x, y = position.x, position.y
        yaw = yaw_from_quaternion(orientation)

        tx, ty, tyaw = self.target

        position_error = math.hypot(x - tx, y - ty)
        heading_error = abs(angle_difference(yaw, tyaw))

        twist = self.odom.twist.twist

        speed = math.hypot(
            twist.linear.x,
            twist.linear.y,
        )
        angular_speed = abs(twist.angular.z)

        measured_joints = dict(
            zip(self.joints.name, self.joints.position)
        )

        missing = [
            joint for joint in STEERING_JOINTS
            if joint not in measured_joints
        ]

        if missing:
            return False, (
                "MISSING STEERING FEEDBACK: "
                + ", ".join(missing)
            )

        steering_angles = [
            measured_joints[joint]
            for joint in STEERING_JOINTS
        ]

        max_steering = max(
            abs(a) for a in steering_angles
        )

        # The wheels must be straight for the planned
        # 1 m precision-entry centerline.
        tests = {
            "position": position_error <= 0.02,
            "heading": heading_error <= math.radians(2),
            "linear_speed": speed <= 0.02,
            "angular_speed": angular_speed <= 0.02,
            "steering": max_steering <= math.radians(2),
        }

        report = (
            f"pose=({x:.3f}, {y:.3f}, "
            f"{math.degrees(yaw):.2f} deg) | "
            f"position_error={position_error:.3f} m | "
            f"heading_error={math.degrees(heading_error):.2f} deg | "
            f"speed={speed:.3f} m/s | "
            f"angular_speed={angular_speed:.3f} rad/s | "
            f"max_steering={math.degrees(max_steering):.2f} deg | "
            f"TF_age={tf_age:.3f} s | "
            f"checks={tests}"
        )

        return all(tests.values()), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    args, ros_args = parser.parse_known_args()

    # Only ROS 2 arguments may follow our script arguments.
    # rclpy.init() will validate them.
    if ros_args and ros_args[0] != "--ros-args":
        parser.error("Unrecognized arguments: " + " ".join(ros_args))


    if args.duration <= 0:
        parser.error("--duration must be positive")

    route = json.loads(Path(args.route).read_text())

    if route.get("schema") != "amr_phase14_two_segment_refined_v1":
        raise SystemExit("ERROR: Not a Phase 14B refined route")

    if route.get("execution_approved") is not False:
        raise SystemExit(
            "ERROR: Expected motion-disabled route"
        )

    target = route["checkpoint"]["target_pose"]

    rclpy.init()
    node = PredockMonitor(target)

    print("=" * 65)
    print("PHASE 14C — READ-ONLY CHECKPOINT MONITOR")
    print("Destination:", route["destination"])
    print("Target:", target)
    print("Motion commands: DISABLED")
    print("Docking permission: DISABLED")
    print("=" * 65, flush=True)

    start = time.monotonic()
    next_report = start
    consecutive_passes = 0
    ever_ready = False

    try:
        while (
            rclpy.ok()
            and time.monotonic() - start < args.duration
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()

            if now < next_report:
                continue

            next_report = now + 1.0
            passed, report = node.inspect()

            if passed:
                consecutive_passes += 1
            else:
                consecutive_passes = 0

            # Require five consecutive passing observations
            # rather than a single instantaneous measurement.
            if consecutive_passes >= 5:
                ever_ready = True

            status = (
                "CHECKPOINT READY (READ-ONLY)"
                if consecutive_passes >= 5
                else "CHECKPOINT NOT READY"
            )

            print(status)
            print(report, flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        print()
        print("Checkpoint observed ready:", ever_ready)
        print("Precision docking authorized: NO")
        print("Motion commands published: 0")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
