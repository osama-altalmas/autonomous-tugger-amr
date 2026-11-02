#!/usr/bin/env python3
"""Distance-based motion using the existing ROS 2 four-wheel-steering controller."""

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


class DistanceDriver(Node):

    def __init__(self, distance, speed):
        super().__init__('tugger_distance_driver')

        self.target = abs(distance)
        self.direction = 1.0 if distance >= 0 else -1.0
        self.speed = speed

        self.tolerance = 0.005
        self.start_pose = None
        self.current_pose = None
        self.last_odom_time = None

        self.last_progress = 0.0
        self.last_progress_time = time.monotonic()

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.create_subscription(
            Odometry,
            '/wheel/odom',
            self.odom_callback,
            10
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        self.current_pose = (p.x, p.y)
        self.last_odom_time = time.monotonic()

        if self.start_pose is None:
            self.start_pose = self.current_pose

    def command(self, linear_speed):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = linear_speed
        msg.twist.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def stop(self):
        for _ in range(10):
            self.command(0.0)
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self):
        self.get_logger().info(
            f'Distance target: {self.direction * self.target:.3f} m'
        )

        try:
            # Wait for fresh wheel odometry.
            deadline = time.monotonic() + 10.0

            while rclpy.ok() and self.current_pose is None:
                rclpy.spin_once(self, timeout_sec=0.1)

                if time.monotonic() > deadline:
                    self.get_logger().error(
                        'No wheel odometry received.'
                    )
                    return False

            if self.target <= self.tolerance:
                self.get_logger().info('Already at target.')
                return True

            start_time = time.monotonic()
            self.last_progress_time = start_time

            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)

                now = time.monotonic()

                if (
                    self.last_odom_time is None
                    or now - self.last_odom_time > 1.0
                ):
                    self.get_logger().error(
                        'Wheel odometry lost. Stopping.'
                    )
                    return False

                dx = self.current_pose[0] - self.start_pose[0]
                dy = self.current_pose[1] - self.start_pose[1]

                traveled = math.hypot(dx, dy)
                remaining = self.target - traveled

                if remaining <= self.tolerance:
                    self.get_logger().info(
                        f'Target reached: {traveled:.3f} m'
                    )
                    return True

                # Reduce speed as the robot approaches its target.
                linear_speed = min(
                    self.speed,
                    max(0.005, remaining * 0.5)
                )

                self.command(self.direction * linear_speed)

                if traveled > self.last_progress + 0.005:
                    self.last_progress = traveled
                    self.last_progress_time = now

                if now - self.last_progress_time > 8.0:
                    self.get_logger().error(
                        'Movement stalled. Stopping.'
                    )
                    return False

                if now - start_time > max(
                    30.0,
                    4.0 * self.target / self.speed + 15.0
                ):
                    self.get_logger().error(
                        'Motion safety timeout.'
                    )
                    return False

        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(
        description='Drive the tugger a specified distance.'
    )

    parser.add_argument(
        '--distance',
        type=float,
        required=True,
        help='Signed distance in metres'
    )

    parser.add_argument(
        '--speed',
        type=float,
        default=0.05,
        help='Maximum requested linear speed in m/s'
    )

    args = parser.parse_args()

    if not math.isfinite(args.distance):
        parser.error('Distance must be finite')

    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('Speed must be positive and finite')

    rclpy.init()
    node = DistanceDriver(args.distance, args.speed)

    try:
        success = node.run()
    except KeyboardInterrupt:
        node.stop()
        success = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
