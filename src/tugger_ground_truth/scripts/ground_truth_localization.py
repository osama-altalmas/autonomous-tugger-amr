#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool, Float64MultiArray
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_ros import TransformException


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def quaternion_from_yaw(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class GroundTruthLocalization(Node):

    def __init__(self):
        super().__init__("tugger_ground_truth_localization")


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.declare_parameter("position_tolerance_m", 0.03)
        self.declare_parameter("yaw_tolerance_rad", 0.05236)

        self.position_tolerance_m = float(
            self.get_parameter("position_tolerance_m").value
        )
        self.yaw_tolerance_rad = float(
            self.get_parameter("yaw_tolerance_rad").value
        )

        self.latest_pose = None
        self.last_warning_time = 0.0

        self.subscription = self.create_subscription(
            PoseStamped,
            "/tugger/ground_truth_pose",
            self.pose_callback,
            10
        )

        self.error_publisher = self.create_publisher(
            Float64MultiArray,
            "/tugger/localization_error",
            10
        )
        self.health_publisher = self.create_publisher(
            Bool,
            "/tugger/localization_healthy",
            10
        )

        self.timer = self.create_timer(0.05, self.publish_localization)

        self.get_logger().info(
            "Ground-truth localization started. "
            "Publishing map -> odom."
        )

    def pose_callback(self, msg):
        if msg.header.frame_id != "map":
            self.get_logger().error(
                "Expected ground-truth pose in map frame."
            )
            return

        self.latest_pose = msg

    def publish_localization(self):

        if self.latest_pose is None:
            return

        current_time = self.get_clock().now()

        pose_time = Time.from_msg(self.latest_pose.header.stamp)

        age = (current_time - pose_time).nanoseconds / 1e9

        if age < -0.2 or age > 0.5:
            return

        try:
            odom_to_base = self.tf_buffer.lookup_transform(
                "odom",
                "base_footprint",
                Time(),
                timeout=Duration(seconds=0.05)
            )

        except TransformException:
            return

        ground_truth = self.latest_pose.pose

        gx = ground_truth.position.x
        gy = ground_truth.position.y
        gyaw = yaw_from_quaternion(ground_truth.orientation)

        ox = odom_to_base.transform.translation.x
        oy = odom_to_base.transform.translation.y
        oyaw = yaw_from_quaternion(
            odom_to_base.transform.rotation
        )

        dx = gx - ox
        dy = gy - oy
        position_error = math.hypot(dx, dy)
        yaw_error = math.atan2(
            math.sin(gyaw - oyaw),
            math.cos(gyaw - oyaw)
        )

        error = Float64MultiArray()
        # [position error (m), yaw error (rad), truth minus odom x (m), y (m)]
        error.data = [position_error, yaw_error, dx, dy]
        self.error_publisher.publish(error)

        healthy = Bool()
        healthy.data = (
            position_error <= self.position_tolerance_m
            and abs(yaw_error) <= self.yaw_tolerance_rad
        )
        self.health_publisher.publish(healthy)

        # T_map_odom = T_map_base * inverse(T_odom_base)
        map_odom_yaw = yaw_error

        c = math.cos(map_odom_yaw)
        s = math.sin(map_odom_yaw)

        tx = gx - (c * ox - s * oy)
        ty = gy - (s * ox + c * oy)

        qz, qw = quaternion_from_yaw(map_odom_yaw)

        transform = TransformStamped()

        transform.header.stamp = current_time.to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = "odom"

        transform.transform.translation.x = tx
        transform.transform.translation.y = ty
        transform.transform.translation.z = (
            ground_truth.position.z -
            odom_to_base.transform.translation.z
        )

        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(transform)


def main():
    rclpy.init()
    node = GroundTruthLocalization()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
