#!/usr/bin/env python3
"""Mask LaserScan rays that hit the robot's own body before SLAM sees them.

/front_left/scan  ->  /front_left/scan_slam

The sensor pose in the base frame is looked up from TF (not hardcoded).
Rays whose endpoint falls inside the (padded) body box are set to +inf.
If TF is not available yet, the scan is DROPPED rather than forwarded,
so unfiltered self-hits can never reach SLAM Toolbox.
"""

import math

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def mask_scan(ranges, angle_min, angle_increment,
              sensor_x, sensor_y, sensor_yaw,
              x_min, x_max, y_min, y_max):
    """Return (new_ranges, n_masked). Pure function, no ROS needed."""
    r = np.asarray(ranges, dtype=np.float64)
    angles = sensor_yaw + angle_min + np.arange(r.size) * angle_increment
    finite = np.isfinite(r)
    rr = np.where(finite, r, 0.0)
    x = sensor_x + rr * np.cos(angles)
    y = sensor_y + rr * np.sin(angles)
    inside = finite & (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    out = r.copy()
    out[inside] = np.inf
    return out, int(inside.sum())


class SlamScanFilter(Node):

    def __init__(self):
        super().__init__('slam_scan_filter')

        self.declare_parameter('input_topic', '/front_left/scan')
        self.declare_parameter('output_topic', '/front_left/scan_slam')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('body_half_length', 0.45)
        self.declare_parameter('body_half_width', 0.30)
        self.declare_parameter('padding', 0.03)

        self.base_frame = self.get_parameter('base_frame').value
        hl = self.get_parameter('body_half_length').value
        hw = self.get_parameter('body_half_width').value
        pad = self.get_parameter('padding').value
        self.box = (-hl - pad, hl + pad, -hw - pad, hw + pad)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sensor_pose = None  # (x, y, yaw) once known
        self.count = 0

        self.pub = self.create_publisher(
            LaserScan, self.get_parameter('output_topic').value,
            qos_profile_sensor_data)
        self.sub = self.create_subscription(
            LaserScan, self.get_parameter('input_topic').value,
            self.scan_callback, qos_profile_sensor_data)

        self.get_logger().info(
            f'Masking box in {self.base_frame}: '
            f'x=[{self.box[0]:.2f},{self.box[1]:.2f}] '
            f'y=[{self.box[2]:.2f},{self.box[3]:.2f}]')

    def _lookup_sensor_pose(self, frame_id):
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, frame_id, rclpy.time.Time(),
                timeout=Duration(seconds=0.5))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(
                f'No TF {self.base_frame} <- {frame_id} yet: {e}',
                throttle_duration_sec=5.0)
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        pose = (t.transform.translation.x, t.transform.translation.y, yaw)
        self.get_logger().info(
            f'Sensor pose in {self.base_frame}: '
            f'x={pose[0]:.3f} y={pose[1]:.3f} yaw={math.degrees(yaw):.1f} deg')
        return pose

    def scan_callback(self, msg):
        if self.sensor_pose is None:
            self.sensor_pose = self._lookup_sensor_pose(msg.header.frame_id)
            if self.sensor_pose is None:
                return  # drop: never forward unfiltered scans

        new_ranges, n = mask_scan(
            msg.ranges, msg.angle_min, msg.angle_increment,
            *self.sensor_pose, *self.box)

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = new_ranges.astype(np.float32).tolist()
        out.intensities = list(msg.intensities)
        self.pub.publish(out)

        self.count += 1
        if self.count % 45 == 0:  # about every 3 s at 15 Hz
            self.get_logger().info(f'masked {n}/{len(msg.ranges)} rays')


def main(args=None):
    rclpy.init(args=args)
    node = SlamScanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
