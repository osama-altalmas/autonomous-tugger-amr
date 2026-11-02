#!/usr/bin/env python3
"""AMR V1 velocity recorder. Records telemetry; never commands motion."""

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.clock import Clock, ClockType

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


class Recorder(Node):

    def __init__(self, filename, rate):
        super().__init__("amr_velocity_recorder")

        self.set_parameters([
            Parameter(
                "use_sim_time",
                Parameter.Type.BOOL,
                True
            )
        ])

        self.filename = Path(filename)
        self.filename.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.file = self.filename.open(
            "w",
            newline="",
            buffering=1
        )

        self.columns = [
            "wall_time",
            "wall_elapsed_s",
            "sim_time_s",
            "rtf",
            "requested_v_mps",
            "requested_omega_rps",
            "controller_target_v_mps",
            "controller_commanded_v_mps",
            "target_steering_rad",
            "commanded_steering_rad",
            "controller_omega_rps",
            "measured_vx_mps",
            "measured_vy_mps",
            "measured_speed_mps",
            "measured_omega_rps",
            "odom_x_m",
            "odom_y_m",
            "cmd_age_s",
            "controller_age_s",
            "odom_age_s",
        ]

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.columns
        )

        self.writer.writeheader()

        self.start_wall = time.monotonic()
        self.previous_wall = None
        self.previous_sim = None

        self.requested_v = None
        self.requested_omega = None
        self.controller = [None] * 5
        self.vx = None
        self.vy = None
        self.omega = None
        self.x = None
        self.y = None

        self.cmd_received = None
        self.controller_received = None
        self.odom_received = None

        self.count = 0

        self.create_subscription(
            TwistStamped,
            "/cmd_vel",
            self.cmd_callback,
            20
        )

        self.create_subscription(
            Float64MultiArray,
            "/four_ws/motion_state",
            self.controller_callback,
            20
        )

        self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self.odom_callback,
            20
        )

        # Use a steady wall clock for sampling.
        # Gazebo simulation time is recorded separately.
        self.timer = self.create_timer(
            1.0 / rate,
            self.sample,
            clock=Clock(
                clock_type=ClockType.STEADY_TIME
            )
        )

        self.get_logger().info(
            f"Recording at {rate} Hz -> {self.filename}"
        )

    def cmd_callback(self, msg):
        self.requested_v = msg.twist.linear.x
        self.requested_omega = msg.twist.angular.z
        self.cmd_received = time.monotonic()

    def controller_callback(self, msg):
        if len(msg.data) >= 5:
            self.controller = list(msg.data[:5])
            self.controller_received = time.monotonic()

    def odom_callback(self, msg):
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.omega = msg.twist.twist.angular.z

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        self.odom_received = time.monotonic()

    @staticmethod
    def age(now, received):
        if received is None:
            return None
        return round(now - received, 4)

    def sample(self):
        wall = time.monotonic()
        sim = self.get_clock().now().nanoseconds / 1e9

        rtf = None

        if (
            self.previous_wall is not None
            and self.previous_sim is not None
        ):
            wall_dt = wall - self.previous_wall
            sim_dt = sim - self.previous_sim

            if wall_dt > 0 and sim_dt >= 0:
                rtf = sim_dt / wall_dt

        measured_speed = None

        if self.vx is not None and self.vy is not None:
            measured_speed = math.hypot(
                self.vx, self.vy
            )

        row = {
            "wall_time":
                datetime.now().astimezone().isoformat(),
            "wall_elapsed_s":
                round(wall - self.start_wall, 4),
            "sim_time_s": round(sim, 4),
            "rtf": rtf,
            "requested_v_mps":
                self.requested_v,
            "requested_omega_rps":
                self.requested_omega,
            "controller_target_v_mps":
                self.controller[0],
            "controller_commanded_v_mps":
                self.controller[1],
            "target_steering_rad":
                self.controller[2],
            "commanded_steering_rad":
                self.controller[3],
            "controller_omega_rps":
                self.controller[4],
            "measured_vx_mps": self.vx,
            "measured_vy_mps": self.vy,
            "measured_speed_mps": measured_speed,
            "measured_omega_rps": self.omega,
            "odom_x_m": self.x,
            "odom_y_m": self.y,
            "cmd_age_s":
                self.age(wall, self.cmd_received),
            "controller_age_s":
                self.age(wall, self.controller_received),
            "odom_age_s":
                self.age(wall, self.odom_received),
        }

        self.writer.writerow(row)
        self.count += 1

        if self.count % 20 == 0:
            self.file.flush()

        self.previous_wall = wall
        self.previous_sim = sim

    def finish(self):
        self.file.flush()
        self.file.close()

        self.get_logger().info(
            f"Saved {self.count} samples to {self.filename}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rate", type=float, default=10)

    args = parser.parse_args()

    if not 1 <= args.rate <= 50:
        parser.error("Rate must be between 1 and 50 Hz")

    rclpy.init()

    node = Recorder(args.output, args.rate)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
